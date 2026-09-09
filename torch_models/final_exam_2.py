import os
import sys
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..'))
os.chdir(PROJECT_ROOT)  # 确保工作目录始终是项目根目录

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import timm
import numpy as np
import hdbscan
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from pytorch_metric_learning import losses, samplers, distances
from pytorch_metric_learning.utils import common_functions as c_f
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from datetime import datetime
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

CFG = {
    'data_root_train': os.path.join(PROJECT_ROOT, "data/cat_train_val/cat_train"),
    'data_root_val': os.path.join(PROJECT_ROOT, "data/cat_train_val/cat_val"),
    'data_root_test': os.path.join(PROJECT_ROOT, "data/cat_test_final"),
    'output_dir': PROJECT_ROOT,
    'batch_size': 32,
    'epochs_stage1': 5,        
    'epochs_stage2': 45,       
    'embedding_dim': 256,
    'lr_head': 5e-4,
    'lr_backbone': 2e-5,       
    'weight_decay': 5e-4,
    'margin_arcface': 0.50,    
    'gamma_arcface': 64,       
    'margin_circle': 0.25,
    'gamma_circle': 16,        
    'm_per_class': 8,          
    'num_workers': 4,          
    'prefetch_factor': 4,      
    'persistent_workers': True, 
    'seed': 42,
    'val_interval': 3,
    'model_name': "hf_hub:BVRA/MegaDescriptor-B-224",
    'amp': True,
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CFG['device'] = DEVICE
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"设备: {DEVICE}")
print(f"时间戳: {timestamp}")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class AnimalReIDDataset(Dataset):
    def __init__(self, root_dir, mapping_csv, transform=None, return_filename=False):
        self.root_dir = root_dir
        self.transform = transform
        self.return_filename = return_filename
        df = pd.read_csv(mapping_csv, dtype={'new_filename': str})
        class_ids, unique_labels = pd.factorize(df['class_id'])
        self.image_paths = []
        self.class_ids = []
        self.filenames = []
        for i, fname in enumerate(df['new_filename']):
            img_path = os.path.join(root_dir, f"{fname}.jpg")
            if os.path.exists(img_path):
                self.image_paths.append(img_path)
                self.class_ids.append(int(class_ids[i]))
                self.filenames.append(fname)
        self.num_classes = len(pd.unique(pd.array(self.class_ids)))
        missing = len(df) - len(self.image_paths)
        if missing:
            print(f"  {missing} 张图片缺失，实际使用 {len(self.image_paths)} 张")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        if self.transform:
            img = self.transform(img)
        label = self.class_ids[idx]
        if self.return_filename:
            return img, label, self.filenames[idx]
        return img, label


class TestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        import glob
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
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        if self.transform:
            img = self.transform(img)
        return img, idx


#数据增强
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.25, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomRotation(15),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
])

val_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


#模型
class MetricLearningModel(nn.Module):
    """
    骨干网络 + 投影头（含残差连接）
    embedding = normalize(residual_proj + 0.1 * mlp_proj)
    """
    def __init__(self, backbone, embedding_dim=256):
        super().__init__()
        self.backbone = backbone
        in_features = backbone.num_features

        # MLP 投影头
        self.projection = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

        # 残差连接：直接将骨干特征投影到嵌入空间
        self.residual = nn.Linear(in_features, embedding_dim)

    def forward(self, x):
        features = self.backbone(x)
        proj = self.projection(features)
        skip = self.residual(features)
        embeddings = F.normalize(skip + 0.1 * proj, p=2, dim=1)
        return embeddings

    def get_param_groups(self, lr_head, lr_backbone):
        backbone_params = []
        head_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        return [
            {'params': head_params, 'lr': lr_head},
            {'params': backbone_params, 'lr': lr_backbone},
        ]


#评估函数
@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    all_embeddings, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        embeddings = model(imgs)
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_embeddings), np.concatenate(all_labels)


def evaluate_clustering(embeddings, labels, search_params=True):
    results = {}

    #HDBSCAN 评估
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, metric='euclidean')
    pred = clusterer.fit_predict(embeddings)
    # 噪声后处理后再评估
    pred_clean = assign_noise_to_nearest_cluster(embeddings, pred)
    results['ari_hdb'] = adjusted_rand_score(labels, pred_clean)
    results['nmi_hdb'] = normalized_mutual_info_score(labels, pred_clean)
    results['noise_hdb'] = 1.0 - (pred != -1).sum() / len(pred)
    results['n_clusters_hdb'] = len(set(pred_clean))

    #AgglomerativeClustering 全身评估
    n_candidates = [30, 50, 70, 90, 110, 130, 150]
    if len(embeddings) > 5000:
        n_candidates = [50, 80, 110, 140, 170, 200, 250, 300]
    best_n, best_score = None, -1
    for n in n_candidates:
        if n >= len(embeddings):
            continue
        pred_agg = AgglomerativeClustering(
            n_clusters=n, metric='euclidean', linkage='ward'
        ).fit_predict(embeddings)
        score = silhouette_score(embeddings, pred_agg)
        if score > best_score:
            best_score, best_n = score, n
    if best_n is not None:
        pred_agg = AgglomerativeClustering(
            n_clusters=best_n, metric='euclidean', linkage='ward'
        ).fit_predict(embeddings)
        results['ari_agg'] = adjusted_rand_score(labels, pred_agg)
        results['nmi_agg'] = normalized_mutual_info_score(labels, pred_agg)
        results['best_n'] = best_n
    else:
        results['ari_agg'], results['nmi_agg'], results['best_n'] = 0.0, 0.0, 0

    return results


#噪声后处理
def assign_noise_to_nearest_cluster(embeddings, cluster_labels):
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


#测试集推理与聚类
@torch.no_grad()
def predict_test(model, test_loader, device, output_path, search_params=True):
    model.eval()

    # 提取嵌入
    all_embeddings = []
    for imgs, _ in tqdm(test_loader, desc="测试推理"):
        imgs = imgs.to(device)
        embeddings = model(imgs)
        all_embeddings.append(embeddings.cpu().numpy())
    all_embeddings = np.concatenate(all_embeddings)
    print(f"测试集嵌入形状: {all_embeddings.shape}")

    # 获取文件名
    import glob
    test_img_dir = CFG['data_root_test']
    filenames = sorted([
        os.path.basename(p).replace('.jpg', '')
        for p in glob.glob(os.path.join(test_img_dir, "*.jpg"))
    ])

    #参数搜索
    if search_params:
        print("\n搜索 HDBSCAN 参数...")
        best_score = -1
        best_pred = None
        best_params = {}
        n_samples = len(all_embeddings)

        for min_size in [2, 3, 4, 5, 6]:
            for min_samples in [None, 2, 3, 5]:
                for eps in [0.0, 0.1, 0.2]:
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=min_size,
                        min_samples=min_samples,
                        cluster_selection_epsilon=eps,
                        metric='euclidean'
                    )
                    pred = clusterer.fit_predict(all_embeddings)
                    n_clusters = len(set(pred) - {-1})
                    noise_ratio = (pred == -1).mean()

                    expected_clusters = max(50, n_samples // 25)
                    cluster_score = 1.0 - abs(n_clusters - expected_clusters) / expected_clusters
                    noise_score = 1.0 - abs(noise_ratio - 0.15) / 0.5
                    total_score = cluster_score * 0.5 + noise_score * 0.5

                    if noise_ratio < 0.6 and n_clusters >= 10:
                        if n_clusters >= 2 and n_clusters < n_samples:
                            try:
                                sil = silhouette_score(all_embeddings, pred)
                                total_score = total_score * 0.3 + (sil + 1) / 2 * 0.7
                            except:
                                pass

                    if total_score > best_score:
                        best_score = total_score
                        best_pred = pred.copy()
                        best_params = {
                            'min_cluster_size': min_size,
                            'min_samples': min_samples,
                            'cluster_selection_epsilon': eps,
                        }

        pred_labels = best_pred
        n_clusters = len(set(pred_labels) - {-1})
        n_noise = (pred_labels == -1).sum()
        print(f"最佳参数: {best_params}")
        print(f"簇数: {n_clusters}, 噪声点: {n_noise}/{n_samples} ({n_noise/n_samples:.1%})")
    else:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=3, metric='euclidean')
        pred_labels = clusterer.fit_predict(all_embeddings)

    # 噪声后处理：将 -1 分配到最近簇（避免被算作一个巨大噪声簇）
    pred_labels = assign_noise_to_nearest_cluster(all_embeddings, pred_labels)

    # 输出 CSV
    df = pd.DataFrame({
        'new_filename': filenames,
        'class_id': pred_labels
    })
    df = df.sort_values('new_filename').reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"\n预测结果已保存: {output_path}")
    print(df.head(10))

    return df, all_embeddings


#主训练循环
def main():
    import argparse
    parser = argparse.ArgumentParser(description='猫个体重识别训练')
    parser.add_argument('--cropped', action='store_true',
                        help='使用预处理裁剪后的数据 (data/cat_train_val_cropped)')
    args = parser.parse_args()

    set_seed(CFG['seed'])
    print("=" * 60)
    print("猫个体重识别 —— 改进版训练")
    if args.cropped:
        print("  [使用预处理裁剪数据]")
    print("=" * 60)

    #1. 加载数据
    data_root_train = CFG['data_root_train']
    data_root_val = CFG['data_root_val']
    if args.cropped:
        cropped_base = os.path.join(PROJECT_ROOT, "data/cat_train_val_cropped")
        data_root_train = os.path.join(cropped_base, "cat_train")
        data_root_val = os.path.join(cropped_base, "cat_val")
        print(f"  数据目录: {cropped_base}")

    print("\n[1/6] 加载数据集...")
    train_dataset = AnimalReIDDataset(
        root_dir=data_root_train,
        mapping_csv=os.path.join(data_root_train, "mapping.csv"),
        transform=train_transform,
    )
    val_dataset = AnimalReIDDataset(
        root_dir=data_root_val,
        mapping_csv=os.path.join(data_root_val, "mapping.csv"),
        transform=val_test_transform,
    )
    print(f"  训练集: {len(train_dataset)} 张, {train_dataset.num_classes} 个个体")
    print(f"  验证集: {len(val_dataset)} 张, {val_dataset.num_classes} 个个体")

    #2. 构建模型
    print("\n[2/6] 构建模型...")
    backbone = timm.create_model(CFG['model_name'], pretrained=True, num_classes=0)
    model = MetricLearningModel(backbone, embedding_dim=CFG['embedding_dim'])
    model = model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total_params/1e6:.2f}M, 可训练: {trainable_params/1e6:.2f}M")

    #3. 损失函数
    print("\n[3/6] 配置损失与优化器...")
    distance = distances.CosineSimilarity()
    loss_arcface = losses.ArcFaceLoss(
        num_classes=train_dataset.num_classes,
        embedding_size=CFG['embedding_dim'],
        margin=CFG['margin_arcface'],
        scale=CFG['gamma_arcface'],
        distance=distance,
    )
    loss_circle = losses.CircleLoss(
        m=CFG['margin_circle'],
        gamma=CFG['gamma_circle'],
        distance=distance,
    )
    # ArcFace为主，CircleLoss微量辅助
    loss_weights = {'arcface': 1.0, 'circle': 0.2}

    #4. 数据加载器
    train_labels = train_dataset.class_ids
    sampler = samplers.MPerClassSampler(
        labels=train_labels,
        m=CFG['m_per_class'],
        length_before_new_iter=len(train_dataset),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=CFG['batch_size'],
        sampler=sampler, num_workers=CFG['num_workers'],
        pin_memory=True, drop_last=True,
        prefetch_factor=CFG.get('prefetch_factor', 4),
        persistent_workers=CFG.get('persistent_workers', True),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CFG['batch_size'],
        shuffle=False, num_workers=CFG['num_workers'],
        pin_memory=True,
        prefetch_factor=CFG.get('prefetch_factor', 4),
        persistent_workers=CFG.get('persistent_workers', True),
    )

    #5. 优化器与调度器
    #Stage 1: 只训练投影头
    param_groups = model.get_param_groups(
        lr_head=CFG['lr_head'], lr_backbone=0
    )
    optimizer = optim.AdamW(param_groups, weight_decay=CFG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CFG['epochs_stage1'] + CFG['epochs_stage2'],
        eta_min=1e-6,
    )

    #6. 训练循环
    print("\n[4/6] 开始训练...")
    best_val_ari = 0.0
    best_val_nmi = 0.0
    model_save_path_ari = os.path.join(
        CFG['output_dir'], f"best_ari_model_{timestamp}.pth"
    )
    model_save_path_nmi = os.path.join(
        CFG['output_dir'], f"best_nmi_model_{timestamp}.pth"
    )
    history = {'loss': [], 'ari': [], 'nmi': [], 'noise': []}
    scaler = torch.amp.GradScaler('cuda', enabled=CFG['amp'])

    for epoch in range(1, CFG['epochs_stage1'] + CFG['epochs_stage2'] + 1):
        if epoch == CFG['epochs_stage1'] + 1:
            print(f"\n{'='*40}")
            print(f"Stage 2: 开始微调骨干网络 (lr_backbone={CFG['lr_backbone']})")
            print(f"{'='*40}")
            # 解冻骨干所有参数
            for param in model.backbone.parameters():
                param.requires_grad = True
            # 重建优化器
            param_groups = model.get_param_groups(
                lr_head=CFG['lr_head'], lr_backbone=CFG['lr_backbone']
            )
            optimizer = optim.AdamW(param_groups, weight_decay=CFG['weight_decay'])
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=CFG['epochs_stage2'],
                eta_min=1e-6,
            )
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * 0.1

        model.train()
        total_loss = 0.0
        valid_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CFG['epochs_stage1']+CFG['epochs_stage2']}")

        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            with torch.amp.autocast('cuda', enabled=CFG['amp']):
                embeddings = model(imgs)
                loss_a = loss_arcface(embeddings, labels)
                loss_c = loss_circle(embeddings, labels)
                loss = loss_weights['arcface'] * loss_a + loss_weights['circle'] * loss_c

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            valid_batches += 1
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                arc=f"{loss_a.item():.2f}",
                cir=f"{loss_c.item():.2f}",
            )

        avg_loss = total_loss / max(valid_batches, 1)
        history['loss'].append(avg_loss)
        scheduler.step()

        if epoch <= CFG['epochs_stage1']:
            warmup_ratio = epoch / CFG['epochs_stage1']
            for g in optimizer.param_groups:
                if g['lr'] > 0:
                    g['lr'] = CFG['lr_head'] * warmup_ratio

        if epoch % CFG['val_interval'] == 0 or epoch == 1:
            val_emb, val_labels = extract_embeddings(model, val_loader, DEVICE)
            metrics = evaluate_clustering(val_emb, val_labels)

            history['ari'].append(metrics['ari_hdb'])
            history['nmi'].append(metrics['nmi_hdb'])
            history['noise'].append(metrics['noise_hdb'])

            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"  Val | ARI(hdb)={metrics['ari_hdb']:.4f} "
                f"NMI(hdb)={metrics['nmi_hdb']:.4f} "
                f"ARI(agg)={metrics['ari_agg']:.4f} "
                f"NMI(agg)={metrics['nmi_agg']:.4f} "
                f"噪声={metrics['noise_hdb']:.1%} "
                f"簇={metrics['n_clusters_hdb']} "
                f"LR={current_lr:.2e}"
            )

            # 保存最佳模型
            if metrics['ari_hdb'] > best_val_ari:
                best_val_ari = metrics['ari_hdb']
                torch.save(model.state_dict(), model_save_path_ari)
                print(f"  ★ 新最佳 ARI 模型: {best_val_ari:.4f}")

            if metrics['nmi_hdb'] > best_val_nmi:
                best_val_nmi = metrics['nmi_hdb']
                torch.save(model.state_dict(), model_save_path_nmi)
                print(f"  ★ 新最佳 NMI 模型: {best_val_nmi:.4f}")

    print(f"\n训练完成！")
    print(f"最佳验证 ARI: {best_val_ari:.4f}")
    print(f"最佳验证 NMI: {best_val_nmi:.4f}")

    #最终验证集评估
    print("\n[5/6] 最佳模型在验证集上的最终评估...")
    model.load_state_dict(torch.load(model_save_path_ari, map_location=DEVICE))
    val_emb, val_labels = extract_embeddings(model, val_loader, DEVICE)
    final_metrics = evaluate_clustering(val_emb, val_labels)
    print(f"  HDBSCAN:  ARI={final_metrics['ari_hdb']:.4f}  NMI={final_metrics['nmi_hdb']:.4f}  噪声={final_metrics['noise_hdb']:.1%}")
    print(f"  Agglomerative(n={final_metrics['best_n']}):  ARI={final_metrics['ari_agg']:.4f}  NMI={final_metrics['nmi_agg']:.4f}")

    #测试集推理与输出
    print("\n[6/6] 测试集推理 + 聚类 + 输出CSV...")
    test_dataset = TestDataset(CFG['data_root_test'], transform=val_test_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=CFG['batch_size'] * 2,
        shuffle=False, num_workers=CFG['num_workers'],
        pin_memory=True, persistent_workers=CFG.get('persistent_workers', True),
    )
    print(f"  测试集: {len(test_dataset)} 张图片")

    output_csv = os.path.join(CFG['output_dir'], f"test_predictions_{timestamp}.csv")
    df_test, test_embeddings = predict_test(model, test_loader, DEVICE, output_csv)
    latest_csv = os.path.join(CFG['output_dir'], "test_predictions.csv")
    df_test.to_csv(latest_csv, index=False)
    print(f"最新预测已覆盖: {latest_csv}")
    np.save(os.path.join(CFG['output_dir'], f"val_features_{timestamp}.npy"), val_emb)
    np.save(os.path.join(CFG['output_dir'], f"val_labels_{timestamp}.npy"), val_labels)
    np.save(os.path.join(CFG['output_dir'], f"test_features_{timestamp}.npy"), test_embeddings)

    print("\n" + "=" * 60)
    print("全部完成！")
    print(f"最佳 ARI 模型: {model_save_path_ari}")
    print(f"最佳 NMI 模型: {model_save_path_nmi}")
    print(f"测试预测: {output_csv}")
    print(f"验证集特征: val_features_{timestamp}.npy")
    print("=" * 60)


if __name__ == '__main__':
    main()
