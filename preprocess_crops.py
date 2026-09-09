import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import argparse
import torch
import torchvision
import torchvision.transforms as T
import numpy as np
from PIL import Image
from tqdm import tqdm
import glob

CAT_CLASS_IDS = [18, 17, 19, 20, 21, 22, 23]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_detector():
    print(f"加载 Faster R-CNN...")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    model = model.to(DEVICE)
    model.eval()
    print(f"  模型已加载到 {DEVICE}")
    return model


def detect_animals(model, image_pil, conf_thresh=0.5, target_size=800):
    transform = T.Compose([T.ToTensor()])

    w, h = image_pil.size
    if max(w, h) < target_size:
        scale = target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image_pil = image_pil.resize((new_w, new_h), Image.BILINEAR)

    img_tensor = transform(image_pil).to(DEVICE)

    with torch.no_grad():
        predictions = model([img_tensor])[0]

    boxes = predictions['boxes'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()

    if max(w, h) < target_size:
        scale = max(w, h) / target_size
        boxes = boxes * scale

    animal_mask = np.isin(labels, CAT_CLASS_IDS) & (scores >= conf_thresh)
    animal_boxes = boxes[animal_mask]
    animal_scores = scores[animal_mask]

    if len(animal_scores) > 0:
        order = np.argsort(animal_scores)[::-1]
        animal_boxes = animal_boxes[order]
        animal_scores = animal_scores[order]

    return list(zip(animal_boxes, animal_scores))


def crop_to_animal(image_pil, detections, margin=0.2, min_side=100):
    if not detections:
        return image_pil

    w, h = image_pil.size

    best_box, _ = detections[0]
    x1, y1, x2, y2 = best_box

    box_w = x2 - x1
    box_h = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    new_w = box_w * (1 + margin)
    new_h = box_h * (1 + margin)
    new_w = max(new_w, min_side)
    new_h = max(new_h, min_side)

    x1 = max(0, cx - new_w / 2)
    y1 = max(0, cy - new_h / 2)
    x2 = min(w, cx + new_w / 2)
    y2 = min(h, cy + new_h / 2)

    cropped = image_pil.crop((int(x1), int(y1), int(x2), int(y2)))
    return cropped


def process_directory(model, input_dir, output_dir, conf_thresh=0.5, margin=0.2, target_size=288):
    os.makedirs(output_dir, exist_ok=True)

    image_exts = ('*.jpg', '*.jpeg', '*.png', '*.webp')
    image_paths = []
    for ext in image_exts:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))

    if not image_paths:
        print(f"  未找到图片: {input_dir}")
        return

    stats = {'total': 0, 'detected': 0, 'multi_cat': 0, 'no_detection': 0}

    for img_path in tqdm(image_paths, desc=f"处理 {os.path.basename(input_dir)}"):
        fname = os.path.basename(img_path)
        stats['total'] += 1

        try:
            image_pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  无法读取 {fname}: {e}")
            continue

        detections = detect_animals(model, image_pil, conf_thresh)

        if len(detections) == 0:
            stats['no_detection'] += 1
            cropped = image_pil
        else:
            stats['detected'] += 1
            if len(detections) > 1:
                stats['multi_cat'] += 1
            cropped = crop_to_animal(image_pil, detections, margin=margin)

        cropped = cropped.resize((target_size, target_size), Image.BILINEAR)
        cropped.save(os.path.join(output_dir, fname), quality=95)

    return stats


def main():
    parser = argparse.ArgumentParser(description='猫图片预处理：检测+裁剪')
    parser.add_argument('--input_dir', type=str, default='data/cat_train_val',
                        help='输入目录（包含 cat_train/ cat_val/）')
    parser.add_argument('--output_dir', type=str, default='data/cat_train_val_cropped',
                        help='输出目录')
    parser.add_argument('--conf_thresh', type=float, default=0.5,
                        help='检测置信度阈值')
    parser.add_argument('--margin', type=float, default=0.25,
                        help='裁剪边距比例')
    parser.add_argument('--target_size', type=int, default=288,
                        help='输出图片尺寸')
    args = parser.parse_args()

    print("=" * 50)
    print("猫图片预处理：检测 + 裁剪")
    print(f"设备: {DEVICE}")
    print(f"置信度阈值: {args.conf_thresh}")
    print(f"裁剪边距: {args.margin}")
    print("=" * 50)

    detector = load_detector()

    all_stats = {}
    for split in ['cat_train', 'cat_val']:
        input_path = os.path.join(args.input_dir, split)
        output_path = os.path.join(args.output_dir, split)
        if os.path.isdir(input_path):
            print(f"\n处理 {split}...")
            stats = process_directory(
                detector, input_path, output_path,
                args.conf_thresh, args.margin, args.target_size
            )
            if stats:
                all_stats[split] = stats
                print(f"  {split}: 共{stats['total']}张, "
                      f"检测到猫{stats['detected']}张, "
                      f"多猫{stats['multi_cat']}张, "
                      f"未检测到{stats['no_detection']}张")

    for split in ['cat_train', 'cat_val']:
        src_csv = os.path.join(args.input_dir, split, 'mapping.csv')
        dst_csv = os.path.join(args.output_dir, split, 'mapping.csv')
        if os.path.exists(src_csv):
            import shutil
            os.makedirs(os.path.dirname(dst_csv), exist_ok=True)
            shutil.copy2(src_csv, dst_csv)
            print(f"  已复制 mapping.csv: {dst_csv}")

    print("\n" + "=" * 50)
    print("预处理完成！")
    print(f"输出目录: {os.path.abspath(args.output_dir)}")
    print("=" * 50)


if __name__ == '__main__':
    main()
