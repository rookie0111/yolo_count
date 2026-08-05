#!/usr/bin/env python3
"""SAM2.1 LoRA 物理坐标可视化推理脚本

在基础可视化基础上增加像素→物理坐标(mm)转换。
适用于有相机内参标定的工业检测场景。

相机内参公式:
    物理X = (pixel_x - cx) / fx   (mm)
    物理Y = (pixel_y - cy) / fy   (mm)
    物理宽度 = pixel_width / fx   (mm)
    物理高度 = pixel_height / fy   (mm)
    物理面积 = pixel_area / (fx * fy)  (mm²)

Usage:
    python -m sam2_lora.infer_physical \\
        --checkpoint /path/to/best/checkpoint.pt \\
        --data-root /path/to/dataset \\
        --output /path/to/output \\
        --fx 208.333 --fy 208.333 --cx 2736 --cy 1824

    # 用数据集自带的 class_map.json
    python -m sam2_lora.infer_physical \\
        --checkpoint /path/to/checkpoint.pt \\
        --data-root /mnt/c/Users/finn.zhang/project/Datasets/sam_dataset \\
        --output /path/to/output \\
        --split train
"""

import sys
import os
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SAM2_ROOT = "/home/finn/ultralytics-main-count/sam2"
PROJECT_ROOT = "/home/finn/ultralytics-main-count"
sys.path.insert(0, SAM2_ROOT)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ultralytics", "sam"))

from hydra.core.global_hydra import GlobalHydra
GlobalHydra.instance().clear()

import sam2_lora.config as cfg
from sam2_lora.model import apply_lora, build_model, load_checkpoint


# ============================================================
# ClassHead (must match train.py definition)
# ============================================================
class ClassHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ============================================================
# Visualization colors
# ============================================================
CLASS_COLORS = {
    "class0": (0, 255, 0),
    "class1": (0, 200, 255),
    "class2": (255, 100, 0),
    "class3": (200, 0, 255),
    "class4": (0, 255, 200),
}
DEFAULT_COLORS = [(0, 255, 0), (0, 200, 255), (255, 100, 0), (200, 0, 255), (0, 255, 200)]


# ============================================================
# Physical coordinate conversion
# ============================================================
class PhysicalConverter:
    """像素→物理坐标转换器 (mm)"""

    def __init__(self, fx: float, fy: float, cx: float, cy: float):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def pixel_to_physical(self, px: float, py: float):
        """像素坐标 → 物理坐标(mm)"""
        phys_x = (px - self.cx) / self.fx
        phys_y = (py - self.cy) / self.fy
        return phys_x, phys_y

    def pixel_size_to_physical(self, pw: float, ph: float):
        """像素尺寸 → 物理尺寸(mm)"""
        return pw / self.fx, ph / self.fy

    def pixel_area_to_physical(self, pixel_area: float):
        """像素面积 → 物理面积(mm²)"""
        return pixel_area / (self.fx * self.fy)


# ============================================================
# Model loading
# ============================================================
def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load SAM2 + LoRA + ClassHead from training checkpoint."""
    model = build_model()
    load_checkpoint(model, str(cfg.CHECKPOINT_FILE))
    model = apply_lora(model)

    state = torch.load(checkpoint_path, map_location="cpu")
    model.image_encoder.trunk.load_state_dict(state["lora_weights"])

    num_classes = state.get("num_classes", cfg.NUM_CLASSES)
    class_head = ClassHead(in_channels=256, num_classes=num_classes).to(device)
    if "class_head" in state:
        class_head.load_state_dict(state["class_head"])
    class_head.eval()

    model = model.to(device)
    model.eval()
    return model, class_head, num_classes


# ============================================================
# Prediction
# ============================================================
def predict(model, class_head, image_rgb, device="cuda", threshold=0.5):
    """Run SAM2 + ClassHead on a single image."""
    h_orig, w_orig = image_rgb.shape[:2]
    img_resized = cv2.resize(image_rgb, (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))
    img_tensor = (
        torch.from_numpy(img_resized)
        .float()
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
        / 255.0
    )

    point_coords = torch.tensor(
        [[[cfg.IMAGE_SIZE / 2, cfg.IMAGE_SIZE / 2]]],
        dtype=torch.float, device=device,
    )
    point_labels = torch.ones(1, 1, dtype=torch.long, device=device)

    with torch.no_grad():
        backbone_out = model.forward_image(img_tensor)
        image_embeddings = backbone_out["vision_features"]
        high_res = [backbone_out["backbone_fpn"][0], backbone_out["backbone_fpn"][1]]

        sparse_emb, dense_emb = model.sam_prompt_encoder(
            points=(point_coords, point_labels), boxes=None, masks=None,
        )
        image_pe = model.sam_prompt_encoder.get_dense_pe()

        low_res, ious, _, _ = model.sam_mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res,
        )

        mask = (
            F.interpolate(low_res, size=(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
                           mode="bilinear", align_corners=False)
            .sigmoid().squeeze(0)
        )

        cls_logits = class_head(image_embeddings.float())
        cls_probs = torch.sigmoid(cls_logits).squeeze(0).cpu()

    mask_np = mask.squeeze().float().cpu().numpy()
    mask_binary = (mask_np > threshold).astype(np.uint8)
    if (h_orig, w_orig) != (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE):
        mask_binary = cv2.resize(mask_binary, (w_orig, h_orig),
                                 interpolation=cv2.INTER_NEAREST)

    pred_class = int(cls_probs.argmax().item())
    return mask_binary, pred_class, cls_probs.numpy()


# ============================================================
# Visualization with physical coordinates
# ============================================================
def plot_results_physical(
    image: np.ndarray,
    mask_binary: np.ndarray,
    converter: PhysicalConverter,
    class_name: str = "unknown",
    class_conf: float = 0.0,
    merge_distance: int = 0,
    min_area: int = 20,
):
    """Overlay mask, contours, bounding boxes with physical coordinate labels."""
    annotated = image.copy()
    base_color = CLASS_COLORS.get(class_name, DEFAULT_COLORS[0])

    if int(np.sum(mask_binary)) == 0:
        return annotated

    if merge_distance > 0:
        k = merge_distance * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        work_mask = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
    else:
        work_mask = mask_binary

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        work_mask, connectivity=8,
    )

    defects = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x, y, w, h = (
            stats[i, cv2.CC_STAT_LEFT],
            stats[i, cv2.CC_STAT_TOP],
            stats[i, cv2.CC_STAT_WIDTH],
            stats[i, cv2.CC_STAT_HEIGHT],
        )
        defects.append((i, x, y, w, h))

    if not defects:
        return annotated

    # Draw masks, contours, boxes
    for i, x, y, w, h in defects:
        defect_mask = cv2.bitwise_and(mask_binary, (labels == i).astype(np.uint8))
        alpha = 0.6
        colored = np.zeros_like(image)
        colored[defect_mask == 1] = base_color
        region = defect_mask == 1
        annotated[region] = (
            annotated[region] * (1 - alpha) + colored[region] * alpha
        ).astype(np.uint8)

        contours, _ = cv2.findContours(
            defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(annotated, contours, -1, base_color, 3)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), base_color, 3)

    # Labels with physical coordinates
    # Two-line label: class:conf | area mm² box WxH mm
    label1 = f"{class_name}:{class_conf:.2f}"
    (tw1, th1), _ = cv2.getTextSize(label1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    label_h = th1 * 2 + 14
    label_w = 280  # Fixed width for physical coords

    img_h, img_w = image.shape[:2]
    placed = []

    for idx, (i, x, y, w, h) in enumerate(defects):
        defect_area_px = int(np.sum((labels == i).astype(np.uint8) & mask_binary))

        # Physical coordinates
        phys_w, phys_h = converter.pixel_size_to_physical(w, h)
        phys_area = converter.pixel_area_to_physical(defect_area_px)
        phys_box_area = phys_w * phys_h

        label2 = f"area={phys_area:.2f}mm2 box:{phys_w:.1f}x{phys_h:.1f}mm"

        candidates = [
            (x + w + 5, y + h // 2 - label_h // 2),
            (x - label_w - 5, y + h // 2 - label_h // 2),
            (x + w // 2 - label_w // 2, y - label_h - 5),
            (x + w // 2 - label_w // 2, y + h + 5),
        ]

        best = None
        for lx, ly in candidates:
            if lx < 0 or ly < 0 or lx + label_w > img_w or ly + label_h > img_h:
                continue
            overlap = False
            for px, py in placed:
                if (lx < px + label_w + 2 and lx + label_w + 2 > px and
                    ly < py + label_h + 2 and ly + label_h + 2 > py):
                    overlap = True
                    break
            if not overlap:
                best = (lx, ly)
                break
        if best is None:
            best = (x + w + 5, y + idx * (label_h + 2))

        lx, ly = best
        placed.append((lx, ly))

        cv2.rectangle(annotated, (lx, ly), (lx + label_w, ly + label_h), base_color, -1)
        cv2.putText(annotated, label1, (lx + 3, ly + th1 + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, label2, (lx + 3, ly + th1 * 2 + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Leader line
        cx, cy = x + w // 2, y + h // 2
        cv2.line(annotated, (cx, cy), (lx + label_w // 2, ly + label_h // 2),
                 base_color, 1, cv2.LINE_AA)

    return annotated


# ============================================================
# Data discovery
# ============================================================
def discover_images(data_root: Path, split: str = "train"):
    split_file = data_root / f"{split}.txt"
    img_dir = data_root / "images"

    if split_file.is_file():
        names = [l.strip() for l in split_file.read_text().splitlines() if l.strip()]
        paths = []
        for name in names:
            for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                p = img_dir / f"{name}{ext}"
                if p.exists():
                    paths.append(p)
                    break
        return sorted(paths)

    if img_dir.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        return sorted([f for f in img_dir.iterdir() if f.suffix.lower() in exts])
    return []


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="SAM2.1 LoRA physical coordinate visualization inference",
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument("--merge-distance", type=int, default=0)
    parser.add_argument("--fx", type=float, required=True, help="焦距 fx (pixel)")
    parser.add_argument("--fy", type=float, required=True, help="焦距 fy (pixel)")
    parser.add_argument("--cx", type=float, required=True, help="光心 cx (pixel)")
    parser.add_argument("--cy", type=float, required=True, help="光心 cy (pixel)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    converter = PhysicalConverter(args.fx, args.fy, args.cx, args.cy)

    print("=" * 60)
    print("SAM2.1 LoRA Physical Coordinate Visualization")
    print("=" * 60)
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Data root  : {args.data_root}")
    print(f"Output     : {args.output}")
    print(f"Split      : {args.split}")
    print(f"Device     : {device}")
    print(f"Threshold  : {args.threshold}")
    print(f"Camera     : fx={args.fx} fy={args.fy} cx={args.cx} cy={args.cy}")
    print()

    model, class_head, num_classes = load_model(args.checkpoint, str(device))

    idx_to_class = {}
    class_map_path = data_root / "class_map.json"
    if class_map_path.exists():
        import json
        with open(class_map_path) as f:
            cmap = json.load(f)
        for k, v in cmap.items():
            idx_to_class[v] = f"class{k}"
    else:
        for i in range(num_classes):
            idx_to_class[i] = f"class{i}"

    images = discover_images(data_root, args.split)
    if not images:
        print("[ERROR] No images found.")
        return
    print(f"Found {len(images)} images\n")

    for idx, img_path in enumerate(images):
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        pred_mask, pred_cls, cls_probs = predict(
            model, class_head, image_rgb, str(device), args.threshold,
        )
        class_name = idx_to_class.get(pred_cls, f"class{pred_cls}")
        class_conf = float(cls_probs[pred_cls])

        annotated = plot_results_physical(
            image_rgb, pred_mask, converter,
            class_name=class_name, class_conf=class_conf,
            merge_distance=args.merge_distance, min_area=args.min_area,
        )

        save_path = output_dir / f"{img_path.stem}.jpg"
        cv2.imwrite(str(save_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

        pixel_area = int(np.sum(pred_mask))
        phys_area = converter.pixel_area_to_physical(pixel_area)
        print(
            f"  [{idx + 1}/{len(images)}] {img_path.name}  "
            f"class={class_name}({class_conf:.2f})  "
            f"area={pixel_area}px({phys_area:.2f}mm²)"
        )

    print()
    print(f"Done! Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
