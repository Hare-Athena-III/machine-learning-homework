import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
import torch.optim as optim
import timm
import numpy as np
import hdbscan
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from pytorch_metric_learning import losses, samplers, distances
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import AgglomerativeClustering
from datetime import datetime
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

# 设置随机种子（可复现）
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"随机种子已设置为 {seed}")

# 1. 参数配置
DATA_ROOT_TRAIN = r"data/cat_train_val/cat_train"
DATA_ROOT_VAL = r"data/cat_train_val/cat_val"
BATCH_SIZE = 32
EPOCHS = 60
EMBEDDING_DIM = 64
LEARNING_RATE = 1e-4
MARGIN = 0.2
NUM_WORKERS = 4
USE_AMP = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
VAL_INTERVAL = 2          # 每 2 轮验证一次，更精确捕获最佳模型

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_DIR = r"torch_models"
os.makedirs(SAVE_DIR, exist_ok=True)
model_save_path = os.path.join(SAVE_DIR, f"best_metric_model_{timestamp}.pth")
MODEL_NAME = "hf_hub:BVRA/MegaDescriptor-B-224"


# 2. 自定义数据集（读取 mapping.csv）
class AnimalReIDDataset(Dataset):
    def __init__(self, root_dir, mapping_csv, transform=None):
        """
        root_dir: 图片所在目录（如 cat_train）
        mapping_csv: mapping.csv 的完整路径
        transform: 图像预处理
        """
        self.root_dir = root_dir
        self.transform = transform
        df = pd.read_csv(mapping_csv, dtype={'new_filename': str})
        # 将 class_id 映射为 0 开始的连续整数
        class_ids, unique_labels = pd.factorize(df['class_id'])
        # 构建图片路径列表，并过滤掉缺失的文件
        self.image_paths = []
        self.class_ids = []
        for i, fname in enumerate(df['new_filename']):
            img_path = os.path.join(root_dir, f"{fname}.jpg")
            if os.path.exists(img_path):
                self.image_paths.append(img_path)
                self.class_ids.append(class_ids[i])
        self.unique_labels = pd.unique(pd.array(self.class_ids))
        self.num_classes = len(self.unique_labels)
        missing = len(df) - len(self.image_paths)
        if missing > 0:
            print(f"{missing} 张图片缺失，已跳过。实际使用 {len(self.image_paths)} 张图片")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        if self.transform:
            img = self.transform(img)
        label = self.class_ids[idx]
        return img, label


# 3. 数据增强（更丰富的增强策略）
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.15, hue=0.05),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
])

val_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 5. 模型定义
class MetricLearningModel(nn.Module):
    def __init__(self, backbone, embedding_dim=64, num_classes=400):
        super().__init__()
        self.backbone = backbone
        in_features = backbone.num_features
        self.projection = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.projection(features)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        if self.training:
            logits = self.classifier(embeddings)
            return embeddings, logits
        return embeddings

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("骨干网络已冻结，仅训练投影头与分类头")

    def get_trainable_param_groups(self, lr_base):
        return [
            {'params': list(self.projection.parameters()) + list(self.classifier.parameters()),
             'lr': lr_base},
        ]


# 7. 训练与评估函数
def train_epoch(model, loader, optimizer, loss_metric, loss_cls, device, epoch=None, total_epochs=None):
    model.train()
    total_loss = 0.0
    total_metric = 0.0
    total_cls = 0.0
    valid_batches = 0
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}" if epoch else "训练")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.amp.autocast('cuda', enabled=USE_AMP):
            embeddings, logits = model(imgs)
            l_metric = loss_metric(embeddings, labels)
            l_cls = loss_cls(logits, labels)
            loss = l_metric + l_cls
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        total_metric += l_metric.item()
        total_cls += l_cls.item()
        valid_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", metric=f"{l_metric.item():.2f}", cls=f"{l_cls.item():.2f}")
    return total_loss / max(valid_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_embeddings, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        embeddings = model(imgs)  # 推理时 model.eval() 只返回 embedding
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_embeddings), np.concatenate(all_labels)


def compute_recall_at_k(embeddings, labels, k=1):
    n = len(embeddings)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sim_matrix = np.dot(embeddings, embeddings.T)
    correct = 0
    for i in range(n):
        sorted_indices = np.argsort(sim_matrix[i])[::-1]
        sorted_indices = sorted_indices[sorted_indices != i]
        top_k = sorted_indices[:k]
        if np.any(labels[top_k] == labels[i]):
            correct += 1
    return correct / n


def cluster_and_evaluate(embeddings, labels, min_cluster_size=3, min_samples=None, cluster_selection_epsilon=0.0):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric='euclidean'
    )
    pred_labels = clusterer.fit_predict(embeddings)
    valid_mask = pred_labels != -1
    if np.sum(valid_mask) == 0:
        return 0.0, 0.0, 0.0
    ari = adjusted_rand_score(labels[valid_mask], pred_labels[valid_mask])
    nmi = normalized_mutual_info_score(labels[valid_mask], pred_labels[valid_mask])
    noise_ratio = 1.0 - np.sum(valid_mask) / len(labels)
    return ari, nmi, noise_ratio


def full_cluster_and_evaluate(embeddings, labels):
    """与老师一致的评估方式：AgglomerativeClustering + 全身参与"""
    from sklearn.metrics import silhouette_score
    best_n, best_score = None, -1
    for n in [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]:
        pred = AgglomerativeClustering(n_clusters=n, metric='euclidean', linkage='ward').fit_predict(embeddings)
        score = silhouette_score(embeddings, pred)
        if score > best_score:
            best_score = score
            best_n = n
    pred = AgglomerativeClustering(n_clusters=best_n, metric='euclidean', linkage='ward').fit_predict(embeddings)
    ari = adjusted_rand_score(labels, pred)
    nmi = normalized_mutual_info_score(labels, pred)
    print(f"  全身评估(n={best_n}): ARI={ari:.4f} | NMI={nmi:.4f}")
    return ari, nmi


if __name__ == '__main__':
    set_seed(SEED)

    # 4. 加载数据集
    print("Loading datasets...")
    train_dataset = AnimalReIDDataset(
        root_dir=DATA_ROOT_TRAIN,
        mapping_csv=os.path.join(DATA_ROOT_TRAIN, "mapping.csv"),
        transform=train_transform
    )
    val_dataset = AnimalReIDDataset(
        root_dir=DATA_ROOT_VAL,
        mapping_csv=os.path.join(DATA_ROOT_VAL, "mapping.csv"),
        transform=val_test_transform
    )
    print(f"Train: {len(train_dataset)} images, {train_dataset.num_classes} individuals")
    print(f"Val: {len(val_dataset)} images, {val_dataset.num_classes} individuals")

    # 训练集标签（用于采样器）
    train_labels = train_dataset.class_ids

    # 加载骨干网络
    print("Loading backbone...")
    backbone = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
    model = MetricLearningModel(backbone, embedding_dim=EMBEDDING_DIM,
                                num_classes=train_dataset.num_classes)
    model = model.to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # 联合损失：度量学习 + 分类
    distance = distances.CosineSimilarity()
    loss_metric = losses.CircleLoss(m=0.25, gamma=512, distance=distance)
    loss_cls = nn.CrossEntropyLoss(label_smoothing=0.1)

    sampler = samplers.MPerClassSampler(
        labels=train_labels,
        m=8,
        length_before_new_iter=len(train_dataset)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # 冻结骨干，只训练投影头
    model.freeze_backbone()

    # 优化器
    param_groups = model.get_trainable_param_groups(LEARNING_RATE)
    optimizer = optim.AdamW(param_groups, lr=LEARNING_RATE)

    # CosineAnnealing 余弦衰减调度
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    # 8. 训练循环
    print("Starting training...")
    best_val_ari = 0.0
    best_val_nmi = 0.0
    nmi_save_path = model_save_path.replace('.pth', '_best_nmi.pth')
    history = {'loss': [], 'ari': [], 'nmi': [], 'noise': [], 'lr': [], 'r1': [], 'r5': []}

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_metric, loss_cls, DEVICE,
                                 epoch=epoch, total_epochs=EPOCHS)
        scheduler.step()
        history['loss'].append(train_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # 每 VAL_INTERVAL 轮验证（第一轮也验证）
        if epoch % VAL_INTERVAL == 0 or epoch == 1:
            val_embeddings, val_labels = evaluate(model, val_loader, DEVICE)
            val_ari, val_nmi, noise_ratio = cluster_and_evaluate(val_embeddings, val_labels)
            val_full_ari, val_full_nmi = full_cluster_and_evaluate(val_embeddings, val_labels)
            val_r1 = compute_recall_at_k(val_embeddings, val_labels, k=1)
            val_r5 = compute_recall_at_k(val_embeddings, val_labels, k=5)
            history['ari'].append(val_ari)
            history['nmi'].append(val_nmi)
            history['noise'].append(noise_ratio)
            history['r1'].append(val_r1)
            history['r5'].append(val_r5)
            history['lr'].append(current_lr)
            print(f"  Val ARI: {val_ari:.4f} | NMI: {val_nmi:.4f} | R@1: {val_r1:.4f} | R@5: {val_r5:.4f} | 噪声: {noise_ratio:.2%} | LR: {current_lr:.2e}")

            if val_ari > best_val_ari:
                best_val_ari = val_ari
                torch.save(model.state_dict(), model_save_path)
                print(f"新最佳 ARI 模型！ARI = {val_ari:.4f}")

            if val_nmi > best_val_nmi:
                best_val_nmi = val_nmi
                torch.save(model.state_dict(), nmi_save_path)
                print(f"新最佳 NMI 模型！NMI = {val_nmi:.4f}")

    print("\nTraining completed!")
    print(f"Best validation ARI: {best_val_ari:.4f}")
    print(f"Best validation NMI: {best_val_nmi:.4f}")
    print(f"ARI model: {model_save_path}")
    print(f"NMI model: {nmi_save_path}")

    # 9. 最终验证集评估（最佳 ARI 模型）
    model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
    val_embeddings, val_labels = evaluate(model, val_loader, DEVICE)
    final_ari, final_nmi, noise_ratio = cluster_and_evaluate(val_embeddings, val_labels)
    final_full_ari, final_full_nmi = full_cluster_and_evaluate(val_embeddings, val_labels)
    final_r1 = compute_recall_at_k(val_embeddings, val_labels, k=1)
    final_r5 = compute_recall_at_k(val_embeddings, val_labels, k=5)
    print(f"\n{'='*50}")
    print(f"最终评估结果（最佳 ARI 模型）")
    print(f"{'='*50}")
    print(f"HDBSCAN: ARI={final_ari:.4f} | NMI={final_nmi:.4f} | 噪声={noise_ratio:.2%}")
    print(f"全身评估: ARI={final_full_ari:.4f} | NMI={final_full_nmi:.4f}")
    print(f"R@1: {final_r1:.4f} | R@5: {final_r5:.4f}")
    print(f"{'='*50}")

    # 评估最佳 NMI 模型
    model.load_state_dict(torch.load(nmi_save_path, map_location=DEVICE))
    val_embeddings, val_labels = evaluate(model, val_loader, DEVICE)
    nmi_final_ari, nmi_final_nmi, nmi_noise_ratio = cluster_and_evaluate(val_embeddings, val_labels)
    nmi_full_ari, nmi_full_nmi = full_cluster_and_evaluate(val_embeddings, val_labels)
    nmi_final_r1 = compute_recall_at_k(val_embeddings, val_labels, k=1)
    nmi_final_r5 = compute_recall_at_k(val_embeddings, val_labels, k=5)
    print(f"\n{'='*50}")
    print(f"最终评估结果（最佳 NMI 模型）")
    print(f"{'='*50}")
    print(f"HDBSCAN: ARI={nmi_final_ari:.4f} | NMI={nmi_final_nmi:.4f} | 噪声={nmi_noise_ratio:.2%}")
    print(f"全身评估: ARI={nmi_full_ari:.4f} | NMI={nmi_full_nmi:.4f}")
    print(f"R@1: {nmi_final_r1:.4f} | R@5: {nmi_final_r5:.4f}")
    print(f"{'='*50}")

    # 10. 绘制训练曲线
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs_range = list(range(1, EPOCHS + 1))
    val_epochs = [i for i in range(1, EPOCHS + 1) if i % VAL_INTERVAL == 0 or i == 1]

    axes[0].plot(epochs_range, history['loss'], label='Train Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(val_epochs, history['ari'], 'b-o', label='ARI')
    axes[1].plot(val_epochs, history['nmi'], 'r-s', label='NMI')
    axes[1].plot(val_epochs, history['r1'], 'c-^', label='R@1')
    axes[1].plot(val_epochs, history['r5'], 'm-v', label='R@5')
    axes[1].axhline(y=best_val_ari, color='g', linestyle='--', alpha=0.5, label=f'Best ARI={best_val_ari:.4f}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Validation Metrics')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(val_epochs, history['noise'], 'g-^', label='Noise Ratio')
    axes[2].plot(val_epochs, history['lr'], 'm-d', label='LR')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Value')
    axes[2].set_title('Noise Ratio & Learning Rate')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    curve_path = os.path.join(SAVE_DIR, f"training_curve_{timestamp}.png")
    plt.savefig(curve_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练曲线已保存: {curve_path}")

    # 保存验证集特征
    np.save(os.path.join(SAVE_DIR, f"val_features_{timestamp}.npy"), val_embeddings)
    np.save(os.path.join(SAVE_DIR, f"val_labels_{timestamp}.npy"), val_labels)
    print(f"Features saved to {SAVE_DIR}")
