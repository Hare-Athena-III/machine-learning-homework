"""
猫个体重识别 —— 推理 + 聚类 + CSV输出
======================================
用法:  python inference_test.py --model_path <path> [--test_dir <dir>] [--search]
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import hdbscan
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import glob
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# ======================== 配置 ========================
DEFAULT_MODEL_PATH = r"torch_models/best_ari_model_20260713_095153.pth"
TEST_DIR = r"data/cat_test_final"
OUTPUT_CSV = r"test_predictions.csv"
BATCH_SIZE = 64
NUM_WORKERS = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = "hf_hub:BVRA/MegaDescriptor-B-224"
EMBEDDING_DIM = 256


# ======================== 模型定义（必须与训练一致） ========================
class MetricLearningModel(nn.Module):
    """
    骨干网络 + 投影头（含残差连接）
    必须与 final_exam_2.py 中的定义完全一致
    """
    def __init__(self, backbone, embedding_dim=256):
        super().__init__()
        self.backbone = backbone
        in_features = backbone.num_features  # 1024 for MegaDescriptor-B

        self.projection = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

        self.residual = nn.Linear(in_features, embedding_dim)

    def forward(self, x):
        features = self.backbone(x)
        proj = self.projection(features)
        skip = self.residual(features)
        embeddings = F.normalize(skip + 0.1 * proj, p=2, dim=1)
        return embeddings


# ======================== 数据集 ========================
class TestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        self.filenames = [os.path.basename(p) for p in self.img_paths]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        if self.transform:
            img = self.transform(img)
        return img, idx


test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ======================== 聚类参数搜索 ========================
def search_hdbscan_params(embeddings):
    """
    自动搜索 HDBSCAN 最佳参数组合
    返回: (最佳聚类标签, 最佳参数字典)
    """
    print("搜索 HDBSCAN 参数...")
    n_samples = len(embeddings)
    best_score = -float('inf')
    best_pred = None
    best_params = {}

    # 根据数据量设定期望簇数范围
    expected_clusters = max(50, n_samples // 25)

    param_grid = []
    for min_size in [2, 3, 4, 5, 6, 8, 10]:
        for min_samples in [None, 2, 3, 5, 8]:
            for eps in [0.0, 0.05, 0.1, 0.2]:
                param_grid.append((min_size, min_samples, eps))

    for min_size, min_samples, eps in tqdm(param_grid, desc="参数搜索"):
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_size,
            min_samples=min_samples,
            cluster_selection_epsilon=eps,
            metric='euclidean'
        )
        pred = clusterer.fit_predict(embeddings)
        n_clusters = len(set(pred) - {-1})
        noise_ratio = (pred == -1).mean()

        if n_clusters < 5 or n_clusters > n_samples * 0.5:
            continue
        if noise_ratio > 0.7:
            continue

        cluster_score = 1.0 - min(abs(n_clusters - expected_clusters) / expected_clusters, 1.0)
        noise_score = 1.0 - min(abs(noise_ratio - 0.15) / 0.5, 1.0)
        total_score = cluster_score * 0.4 + noise_score * 0.3

        if n_clusters >= 2:
            try:
                sil = silhouette_score(embeddings, pred)
                total_score = total_score * 0.3 + (sil + 1.0) / 2.0 * 0.7
            except:
                total_score *= 0.3
        else:
            total_score *= 0.3

        if total_score > best_score:
            best_score = total_score
            best_pred = pred.copy()
            best_params = {
                'min_cluster_size': min_size,
                'min_samples': min_samples,
                'cluster_selection_epsilon': eps,
                'score': total_score,
                'n_clusters': n_clusters,
                'noise_ratio': noise_ratio,
            }

    if best_pred is None:
        print("参数搜索失败，使用默认参数")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=3, metric='euclidean')
        best_pred = clusterer.fit_predict(embeddings)
        best_params = {'min_cluster_size': 3, 'min_samples': None, 'cluster_selection_epsilon': 0.0}

    n_clusters = len(set(best_pred) - {-1})
    n_noise = (best_pred == -1).sum()
    print(f"\n最佳参数: {best_params}")
    print(f"簇数: {n_clusters}, 噪声: {n_noise}/{n_samples} ({n_noise/n_samples:.1%})")
    return best_pred


# ======================== 噪声后处理 ========================
def assign_noise_to_nearest_cluster(embeddings, cluster_labels):
    """
    将 HDBSCAN 的噪声点 (-1) 分配到最近的簇中心。
    因为评估时将 -1 视为一个普通标签，把所有不同个体混在一起会严重拉低 ARI。
    """
    unique_clusters = set(cluster_labels) - {-1}
    if not unique_clusters:
        return np.zeros_like(cluster_labels)

    centroids = []
    cluster_ids = sorted(unique_clusters)
    for cid in cluster_ids:
        mask = cluster_labels == cid
        centroid = embeddings[mask].mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        centroids.append(centroid)
    centroids = np.array(centroids)

    new_labels = cluster_labels.copy()
    noise_mask = cluster_labels == -1
    if noise_mask.any():
        noise_embeddings = embeddings[noise_mask]
        sims = np.dot(noise_embeddings, centroids.T)
        nearest = sims.argmax(axis=1)
        new_labels[noise_mask] = np.array([cluster_ids[n] for n in nearest])

    n_assigned = noise_mask.sum()
    if n_assigned > 0:
        print(f"  噪声后处理: {n_assigned} 个噪声点已分配到最近簇")

    return new_labels


# ======================== 主流程 ========================
def main():
    parser = argparse.ArgumentParser(description='猫个体重识别推理')
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH,
                        help='模型权重路径')
    parser.add_argument('--test_dir', type=str, default=TEST_DIR,
                        help='测试集目录')
    parser.add_argument('--output', type=str, default=OUTPUT_CSV,
                        help='输出CSV路径')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--no_search', action='store_true',
                        help='跳过HDBSCAN参数搜索，使用默认参数')
    parser.add_argument('--min_cluster_size', type=int, default=3,
                        help='HDBSCAN min_cluster_size（不搜索时生效）')
    args = parser.parse_args()

    print(f"设备: {DEVICE}")
    print(f"模型: {args.model_path}")
    print(f"测试集: {args.test_dir}")
    print(f"输出: {args.output}")

    # ---- 1. 加载测试集 ----
    print("\n[1/4] 加载测试集...")
    test_dataset = TestDataset(args.test_dir, transform=test_transform)
    print(f"  测试集: {len(test_dataset)} 张图片")
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=NUM_WORKERS,
    )

    # ---- 2. 加载模型 ----
    print("\n[2/4] 加载模型...")
    backbone = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
    model = MetricLearningModel(backbone, embedding_dim=EMBEDDING_DIM)

    state_dict = torch.load(args.model_path, map_location=DEVICE)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  ⚠ 缺少的key ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"  ⚠ 多余的key ({len(unexpected)}): {unexpected[:5]}...")

    model = model.to(DEVICE)
    model.eval()
    print(f"  模型加载成功")

    # ---- 3. 提取嵌入 ----
    print("\n[3/4] 提取嵌入向量...")
    all_embeddings = []
    with torch.no_grad():
        for imgs, _ in tqdm(test_loader, desc="推理"):
            imgs = imgs.to(DEVICE)
            embeddings = model(imgs)
            all_embeddings.append(embeddings.cpu().numpy())
    all_embeddings = np.concatenate(all_embeddings)
    print(f"  嵌入形状: {all_embeddings.shape}")

    # ---- 4. 聚类 ----
    print("\n[4/4] 聚类...")
    if args.no_search:
        print("使用默认参数 (不搜索)")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=args.min_cluster_size,
            metric='euclidean'
        )
        cluster_labels = clusterer.fit_predict(all_embeddings)
    else:
        cluster_labels = search_hdbscan_params(all_embeddings)

    # 噪声后处理：将 -1 分配到最近簇
    cluster_labels = assign_noise_to_nearest_cluster(all_embeddings, cluster_labels)

    n_clusters = len(set(cluster_labels))
    n_noise = 0  # 后处理后无噪声
    print(f"  后处理后簇数: {n_clusters} (无噪声)")

    # ---- 5. 输出 CSV ----
    filenames = [os.path.basename(p).replace('.jpg', '')
                 for p in test_dataset.img_paths]
    df = pd.DataFrame({
        'new_filename': filenames,
        'class_id': cluster_labels
    })
    df = df.sort_values('new_filename').reset_index(drop=True)
    df.to_csv(args.output, index=False)
    print(f"\n预测结果已保存: {args.output}")
    print(df.head(10))

    # ---- 6. 可视化 ----
    try:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        n_samples = len(all_embeddings)
        if n_samples > 3000:
            rng = np.random.RandomState(42)
            idx = rng.choice(n_samples, 3000, replace=False)
            vis_emb = all_embeddings[idx]
            vis_lbl = cluster_labels[idx]
        else:
            vis_emb = all_embeddings
            vis_lbl = cluster_labels

        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        vis_2d = tsne.fit_transform(vis_emb)

        plt.figure(figsize=(14, 10))
        scatter = plt.scatter(vis_2d[:, 0], vis_2d[:, 1], c=vis_lbl,
                              cmap='tab20', s=8, alpha=0.7, edgecolors='none')
        plt.colorbar(scatter, label='Cluster ID')
        plt.title(f't-SNE Cluster Visualization ({n_clusters} clusters, {n_noise} noise)',
                  fontsize=14)
        plt.xlabel('t-SNE dim 1')
        plt.ylabel('t-SNE dim 2')
        plt.tight_layout()
        plot_path = args.output.replace('.csv', '_vis.png')
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"可视化已保存: {plot_path}")
    except ImportError as e:
        print(f"可视化跳过（缺少依赖: {e}")


if __name__ == '__main__':
    main()
