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

# 参数
DATA_ROOT_VAL = r"data/cat_train_val/cat_val"
BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = "hf_hub:BVRA/MegaDescriptor-B-224"

# 数据集 (复用 final_exam.py 的 AnimalReIDDataset)
class AnimalReIDDataset(Dataset):
    def __init__(self, root_dir, mapping_csv, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        df = pd.read_csv(mapping_csv, dtype={'new_filename': str})
        self.class_ids, self.unique_labels = pd.factorize(df['class_id'])
        self.num_classes = len(self.unique_labels)
        self.image_paths = [
            os.path.join(root_dir, f"{fname}.jpg")
            for fname in df['new_filename']
        ]

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

# 只用骨干，不用投影头
class BackboneModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
    
    def forward(self, x):
        features = self.backbone(x)
        return nn.functional.normalize(features, p=2, dim=1)

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_embeddings, all_labels = [], []
    for imgs, labels in tqdm(loader, desc="提取特征"):
        imgs = imgs.to(device)
        embeddings = model(imgs)
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

def cluster_and_evaluate(embeddings, labels, min_cluster_size=3):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
    pred_labels = clusterer.fit_predict(embeddings)
    valid_mask = pred_labels != -1
    if np.sum(valid_mask) == 0:
        return 0.0, 0.0, 0.0
    ari = adjusted_rand_score(labels[valid_mask], pred_labels[valid_mask])
    nmi = normalized_mutual_info_score(labels[valid_mask], pred_labels[valid_mask])
    noise_ratio = 1.0 - np.sum(valid_mask) / len(labels)
    return ari, nmi, noise_ratio

if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    
    # 加载验证集
    val_dataset = AnimalReIDDataset(
        root_dir=DATA_ROOT_VAL,
        mapping_csv=os.path.join(DATA_ROOT_VAL, "mapping.csv"),
        transform=val_transform
    )
    print(f"Val: {len(val_dataset)} images, {val_dataset.num_classes} individuals")
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    
    # 加载骨干网络（无投影头）
    print("Loading backbone...")
    backbone = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
    model = BackboneModel(backbone).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 直接评估，不训练
    print("Evaluating raw backbone...")
    val_embeddings, val_labels = evaluate(model, val_loader, DEVICE)
    
    ari, nmi, noise_ratio = cluster_and_evaluate(val_embeddings, val_labels)
    r1 = compute_recall_at_k(val_embeddings, val_labels, k=1)
    r5 = compute_recall_at_k(val_embeddings, val_labels, k=5)
    
    print(f"\n{'='*50}")
    print(f"骨干网络原始评估（无投影头、无训练）")
    print(f"{'='*50}")
    print(f"ARI:       {ari:.4f}")
    print(f"NMI:       {nmi:.4f}")
    print(f"R@1:       {r1:.4f}")
    print(f"R@5:       {r5:.4f}")
    print(f"噪声比例:  {noise_ratio:.2%}")
    print(f"{'='*50}")
