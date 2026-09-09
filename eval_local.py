"""模拟老师评估：在验证集上跑 CSV + 算 ARI/NMI，提前知道分数"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
import timm
import numpy as np
import hdbscan
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from tqdm import tqdm

# ========== 配置 ==========
MODEL_PATH = r"torch_models/training_model.pth"
VAL_DIR = r"data/cat_train_val/cat_val"
BATCH_SIZE = 64
NUM_WORKERS = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = "hf_hub:BVRA/MegaDescriptor-B-224"

# ========== 模型定义（与训练一致） ==========
class MetricLearningModel(nn.Module):
    def __init__(self, backbone, embedding_dim=256):
        super().__init__()
        self.backbone = backbone
        in_features = backbone.num_features
        self.projection = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        self.residual = nn.Linear(in_features, embedding_dim)

    def forward(self, x):
        features = self.backbone(x)
        proj = self.projection(features)
        residual = self.residual(features)
        embeddings = nn.functional.normalize(residual + 0.05 * proj, p=2, dim=1)
        return embeddings

# ========== 验证集数据集 ==========
class ValDataset(Dataset):
    def __init__(self, root_dir, mapping_csv, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        df = pd.read_csv(mapping_csv, dtype={'new_filename': str})
        self.class_ids, self.unique_labels = pd.factorize(df['class_id'])
        self.image_paths = [os.path.join(root_dir, f"{fname}.jpg") for fname in df['new_filename']]
        self.filenames = df['new_filename'].tolist()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        if self.transform:
            img = self.transform(img)
        return img, self.class_ids[idx], idx

# ========== 预处理 ==========
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ========== 主流程 ==========
if __name__ == '__main__':
    print(f"Device: {DEVICE}")

    # 1. 加载验证集
    val_dataset = ValDataset(
        root_dir=VAL_DIR,
        mapping_csv=os.path.join(VAL_DIR, "mapping.csv"),
        transform=val_transform
    )
    print(f"Val: {len(val_dataset)} images, {val_dataset.unique_labels} individuals")

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # 2. 加载模型
    backbone = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
    model = MetricLearningModel(backbone)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    print(f"Model loaded from: {MODEL_PATH}")

    # 3. 提取 embedding
    all_embeddings, all_labels, all_filenames = [], [], []
    with torch.no_grad():
        for imgs, labels, idxs in tqdm(val_loader, desc="Extracting"):
            imgs = imgs.to(DEVICE)
            emb = model(imgs)
            all_embeddings.append(emb.cpu().numpy())
            all_labels.append(labels.numpy())
            all_filenames.extend([val_dataset.filenames[i] for i in idxs.numpy()])

    all_embeddings = np.concatenate(all_embeddings)
    all_labels = np.concatenate(all_labels)

    # ==================== 模拟老师评估 ====================

    print("\n" + "=" * 60)
    print("方法 A: HDBSCAN 带噪声（提交版本）")
    print("=" * 60)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, metric='euclidean')
    pred_hdb = clusterer.fit_predict(all_embeddings)
    valid = pred_hdb != -1
    if valid.sum() > 0:
        ari_hdb = adjusted_rand_score(all_labels[valid], pred_hdb[valid])
        nmi_hdb = normalized_mutual_info_score(all_labels[valid], pred_hdb[valid])
    else:
        ari_hdb, nmi_hdb = 0, 0
    noise_hdb = 1 - valid.sum() / len(pred_hdb)
    print(f"  ARI={ari_hdb:.4f} | NMI={nmi_hdb:.4f} | 噪声={noise_hdb:.2%}")

    # 方法 B: AgglomerativeClustering 全身评估
    print("\n" + "=" * 60)
    print("方法 B: AgglomerativeClustering 全身评估")
    print("=" * 60)
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    best_n, best_score = None, -1
    for n in [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]:
        pred = AgglomerativeClustering(n_clusters=n, metric='euclidean', linkage='ward').fit_predict(all_embeddings)
        score = silhouette_score(all_embeddings, pred)
        if score > best_score:
            best_score, best_n = score, n
    pred_agg = AgglomerativeClustering(n_clusters=best_n, metric='euclidean', linkage='ward').fit_predict(all_embeddings)
    ari_agg = adjusted_rand_score(all_labels, pred_agg)
    nmi_agg = normalized_mutual_info_score(all_labels, pred_agg)
    print(f"  n={best_n}: ARI={ari_agg:.4f} | NMI={nmi_agg:.4f} | 噪声=0%")

    # 方法 C: 内部 HDBSCAN 评估（等同于训练时的指标）
    print("\n" + "=" * 60)
    print("方法 C: 内部 HDBSCAN 评估（训练指标）")
    print("=" * 60)
    print(f"  ARI={ari_hdb:.4f} | NMI={nmi_hdb:.4f} | 噪声={noise_hdb:.2%}")
    print(f"  （方法与 A 相同，仅作对比参考）")

    # ==================== 输出 CSV ====================
    print("\n" + "=" * 60)
    print("生成提交用 CSV（HDBSCAN 带噪声版本）")
    print("=" * 60)
    df = pd.DataFrame({
        'new_filename': all_filenames,
        'class_id': pred_hdb
    })
    df.to_csv("val_predictions_debug.csv", index=False)
    print(f"已保存: val_predictions_debug.csv")
    print(df.head(10))

    print("\n✅ 模拟完成！方法 A 的分数最接近老师实际结果。")
