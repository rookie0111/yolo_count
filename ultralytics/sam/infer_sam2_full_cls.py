#!/usr/bin/env python3
"""Batch inference with the FULL SAM2 pipeline + classification head.

Uses a training checkpoint that contains LoRA adapter weights and a ClassHead
for defect type classification.  For each image, foreground/background point
prompts are sampled from the GT mask and fed through the prompt encoder → mask
decoder path.  Image embeddings are also passed through the ClassHead for
classification.

Usage:
    python infer_sam2_full_cls.py \
        --checkpoint /path/to/output_xutong_cls/best/checkpoint.pt \
        --data-root /mnt/c/Users/finn.zhang/project/Datasets/xutong_military_sam2_cls \
        --output /path/to/output \
        --split train
"""

import sys
import os
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup – SAM2 and project roots must be on sys.path before importing sam2
# ---------------------------------------------------------------------------
SAM2_ROOT = "/home/finn/ultralytics-main-count/sam2"
PROJECT_ROOT = "/home/finn/ultralytics-main-count"
os.chdir(SAM2_ROOT)
sys.path.insert(0, SAM2_ROOT)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ultralytics", "sam"))

from hydra.core.global_hydra import GlobalHydra
GlobalHydra.instance().clear()

import sam2_lora.config as cfg
from sam2_lora.model import apply_lora, build_model, load_checkpoint


# ============================================================
# ClassHead (must match train.py definition exactly)
# ============================================================
class ClassHead(nn.Module):
    """Classification head for defect type prediction."""
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
# Class map and label helpers
# ============================================================
# Default reverse map: internal index → class name
# Populated from class_map.json at startup
IDX_TO_CLASS: dict = {}


def load_class_map(data_root: Path) -> dict:
    """Load class_map.json and build reverse mapping.

    Args:
        data_root: Dataset root directory.

    Returns:
        Dict mapping internal index (int) → original class ID string.
    """
    class_map_path = data_root / "class_map.json"
    if not class_map_path.is_file():
        print(f"  [WARN] class_map.json not found at {class_map_path}, using defaults")
        return {}

    with open(class_map_path, "r") as f:
        raw = json.load(f)  # {"0": 0, "1": 1, "3": 2, ...}

    # Reverse: internal_idx → original_class_id
    reverse = {}
    for orig_id_str, internal_idx in raw.items():
        reverse[int(internal_idx)] = f"class{orig_id_str}"
    return reverse


# ============================================================
# Point prompt sampling
# ============================================================
def sample_point_prompts(
    mask: np.ndarray,
    max_fg: int = 6,
    max_bg: int = 4,
    image_size: int = 1024,
):
    """Sample foreground and background point prompts from a binary mask.

    The mask is assumed to be at *original* resolution; coordinates are scaled
    to ``image_size × image_size`` to match the model's input resolution.

    Args:
        mask: Binary mask (H, W), values in {0, 1}.
        max_fg: Max foreground points to sample (actual: random 5–8).
        max_bg: Max background points to sample (actual: random 3–5).
        image_size: Model input size (coordinates are scaled to this).

    Returns:
        Tuple of ``(point_coords, point_labels)`` tensors ready for the
        prompt encoder, both on CPU.
    """
    h, w = mask.shape[:2]

    # Foreground pixels (inside mask)
    fg_ys, fg_xs = np.where(mask > 0)
    # Background pixels (outside mask)
    bg_ys, bg_xs = np.where(mask == 0)

    n_fg = random.randint(max(1, max_fg - 3), max_fg)
    n_bg = random.randint(max(1, max_bg - 2), max_bg)

    coords = []
    labels = []

    if len(fg_ys) > 0:
        idx = np.random.choice(len(fg_ys), size=min(n_fg, len(fg_ys)), replace=False)
        for i in idx:
            # Scale to model resolution
            cx = float(fg_xs[i]) / w * image_size
            cy = float(fg_ys[i]) / h * image_size
            coords.append([cx, cy])
            labels.append(1)

    if len(bg_ys) > 0:
        idx = np.random.choice(len(bg_ys), size=min(n_bg, len(bg_ys)), replace=False)
        for i in idx:
            cx = float(bg_xs[i]) / w * image_size
            cy = float(bg_ys[i]) / h * image_size
            coords.append([cx, cy])
            labels.append(0)

    if len(coords) == 0:
        # Fallback: centre point
        coords = [[image_size / 2, image_size / 2]]
        labels = [1]

    point_coords = torch.tensor([coords], dtype=torch.float)   # (1, N, 2)
    point_labels = torch.tensor([labels], dtype=torch.long)     # (1, N)
    return point_coords, point_labels


# ============================================================
# Model loading
# ============================================================
def load_trained_model(
    checkpoint_path: str,
) -> tuple:
    """Build SAM2, load pretrained weights, apply LoRA, restore training checkpoint and ClassHead.

    Returns:
        Tuple of (model, class_head) both on cfg.DEVICE and in eval mode.
    """
    print("Building SAM2 model ...")
    model = build_model()

    print(f"Loading pretrained weights: {cfg.CHECKPOINT_FILE}")
    load_checkpoint(model, str(cfg.CHECKPOINT_FILE))

    print("Applying LoRA adapters ...")
    model = apply_lora(model)

    # Restore training checkpoint (LoRA adapter weights + ClassHead)
    print(f"Loading training checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu")

    model.image_encoder.trunk.load_state_dict(state["lora_weights"])
    print(
        f"  LoRA weights restored from epoch {state.get('epoch', '?')} "
        f"(loss={state.get('loss', '?')})"
    )

    # Build and restore ClassHead
    num_classes = state.get("num_classes", cfg.NUM_CLASSES)
    class_head = ClassHead(in_channels=256, num_classes=num_classes)
    if "class_head" in state:
        class_head.load_state_dict(state["class_head"])
        print(f"  ClassHead restored ({num_classes} classes)")
    else:
        print("  [WARN] No class_head in checkpoint, using random weights")

    model = model.to(cfg.DEVICE)
    model.eval()
    class_head = class_head.to(cfg.DEVICE)
    class_head.eval()

    return model, class_head


# ============================================================
# Full SAM2 inference (prompt encoder + mask decoder)
# ============================================================
@torch.no_grad()
def predict_full(
    model: torch.nn.Module,
    image_rgb: np.ndarray,
    point_coords: torch.Tensor,
    point_labels: torch.Tensor,
    image_size: int = 1024,
):
    """Run the full SAM2 pipeline: image encoder → prompt encoder → mask decoder.

    Args:
        model: SAM2Base model in eval mode.
        image_rgb: Input image as RGB numpy array (H, W, 3).
        point_coords: ``(1, N, 2)`` tensor of point coordinates.
        point_labels: ``(1, N)`` tensor of point labels.
        image_size: Model input resolution.

    Returns:
        Tuple of ``(pred_mask, iou_pred, image_embeddings)`` where *pred_mask*
        is a binary numpy array at the original image resolution, *iou_pred*
        is a float, and *image_embeddings* is the feature tensor for classification.
    """
    h_orig, w_orig = image_rgb.shape[:2]

    # Resize & normalise
    img_resized = cv2.resize(image_rgb, (image_size, image_size))
    img_tensor = (
        torch.from_numpy(img_resized)
        .float()
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(cfg.DEVICE)
        / 255.0
    )

    coords = point_coords.to(cfg.DEVICE)
    lbls = point_labels.to(cfg.DEVICE)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        # Image backbone
        backbone_out = model.forward_image(img_tensor)
        image_emb = backbone_out["vision_features"]
        high_res = [
            backbone_out["backbone_fpn"][0],
            backbone_out["backbone_fpn"][1],
        ]

        # Prompt encoder
        sparse_emb, dense_emb = model.sam_prompt_encoder(
            points=(coords, lbls), boxes=None, masks=None,
        )
        image_pe = model.sam_prompt_encoder.get_dense_pe()

        # Mask decoder
        low_res, ious, _, _ = model.sam_mask_decoder(
            image_embeddings=image_emb,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res,
        )

    # Upsample → sigmoid → threshold
    mask = (
        F.interpolate(
            low_res,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
        .sigmoid()
        .squeeze(0)  # (1, H, W)
    )
    iou_pred = float(ious.squeeze().cpu())

    # Resize back to original resolution
    mask_np = mask.squeeze().float().cpu().numpy()
    mask_binary = (mask_np > 0.5).astype(np.uint8)
    if (h_orig, w_orig) != (image_size, image_size):
        mask_binary = cv2.resize(
            mask_binary, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST,
        )

    return mask_binary, iou_pred, image_emb


# ============================================================
# Classification
# ============================================================
@torch.no_grad()
def classify(
    class_head: nn.Module,
    image_embeddings: torch.Tensor,
    idx_to_class: dict,
) -> tuple:
    """Run classification head on image embeddings.

    Args:
        class_head: Classification head module in eval mode.
        image_embeddings: Feature tensor from the image encoder (1, C, H, W).
        idx_to_class: Mapping from internal index to class name string.

    Returns:
        Tuple of (class_name, class_idx, confidence).
    """
    logits = class_head(image_embeddings.float())  # (1, num_classes)
    probs = torch.softmax(logits, dim=-1)
    conf, idx = probs.max(dim=-1)
    class_idx = int(idx.item())
    class_name = idx_to_class.get(class_idx, f"unknown_{class_idx}")
    return class_name, class_idx, float(conf.item())


# ============================================================
# IoU
# ============================================================
def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


# ============================================================
# Visualization (infer_v2.py style with class labels)
# ============================================================
# Per-class color palette (BGR)
CLASS_COLORS = {
    "class0": (0, 255, 0),      # green
    "class1": (0, 200, 255),    # orange
    "class2": (255, 100, 0),    # blue
    "class3": (200, 0, 255),    # purple
    "class4": (0, 255, 200),    # teal
}
DEFAULT_COLORS = [(0, 255, 0), (0, 200, 255), (255, 100, 0), (200, 0, 255), (0, 255, 200)]


def plot_results(
    image: np.ndarray,
    mask_binary: np.ndarray,
    class_name: str = "unknown",
    class_conf: float = 0.0,
    iou: float = 0.0,
    merge_distance: int = 0,
):
    """Overlay mask, contours, bounding box, and class label on the image.

    Each connected component in the mask is treated as one independent defect.

    Args:
        image: RGB image.
        mask_binary: Binary mask (0/1).
        class_name: Predicted class name for the label.
        class_conf: Classification confidence.
        iou: IoU score for display.
        merge_distance: If >0, merge nearby components (default 0 = no merge).
    """
    annotated = image.copy()
    base_color = CLASS_COLORS.get(class_name, DEFAULT_COLORS[0])

    pixel_area = int(np.sum(mask_binary))
    if pixel_area == 0:
        return annotated

    # Find connected components directly on original mask
    if merge_distance > 0:
        # Optional: merge nearby components
        kernel_size = merge_distance * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        work_mask = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
    else:
        work_mask = mask_binary

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        work_mask, connectivity=8
    )

    # Collect all defects first
    defects = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 20:  # Filter tiny noise
            continue
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        defects.append((i, x, y, w, h))

    if not defects:
        return annotated

    # Step 1: Draw all masks, contours, and bounding boxes
    for i, x, y, w, h in defects:
        color = base_color
        x1, y1, x2, y2 = x, y, x + w, y + h

        defect_mask_merged = (labels == i).astype(np.uint8)
        defect_mask = cv2.bitwise_and(mask_binary, defect_mask_merged)

        # Mask overlay - use higher alpha for better visibility
        alpha = 0.6
        colored = np.zeros_like(image)
        colored[defect_mask == 1] = color
        region = defect_mask == 1
        annotated[region] = (
            annotated[region] * (1 - alpha) + colored[region] * alpha
        ).astype(np.uint8)

        # Contours
        contours, _ = cv2.findContours(
            defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(annotated, contours, -1, color, 3)

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

    # Step 2: Place labels with overlap avoidance
    # Use right-side label placement with collision detection
    # Include class, confidence, area, and dimensions in label
    # Calculate pixel area for each defect
    defect_areas = []
    for i, x, y, w, h in defects:
        defect_mask = (labels == i).astype(np.uint8)
        pixel_area = int(np.sum(defect_mask))
        defect_areas.append(pixel_area)

    # Use a two-line label format for better readability
    # Line 1: class:conf
    # Line 2: area={area}px {w}x{h}px
    label_line1 = f"{class_name}:{class_conf:.2f}"
    (tw1, th1), _ = cv2.getTextSize(label_line1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    label_h = th1 * 2 + 14  # Two lines + padding
    label_w = max(tw1, 100) + 6  # Minimum width for second line

    img_h, img_w = image.shape[:2]
    placed_boxes = []  # Track placed label positions

    for idx, (i, x, y, w, h) in enumerate(defects):
        x1, y1, x2, y2 = x, y, x + w, y + h

        # Try multiple label positions: right, left, above, below
        candidate_positions = [
            # Right side of box
            (x2 + 5, y1 + h // 2 - label_h // 2),
            # Left side of box
            (x1 - label_w - 5, y1 + h // 2 - label_h // 2),
            # Above box
            (x1 + w // 2 - label_w // 2, y1 - label_h - 5),
            # Below box
            (x1 + w // 2 - label_w // 2, y2 + 5),
        ]

        best_pos = None
        for lx, ly in candidate_positions:
            # Check bounds
            if lx < 0 or ly < 0 or lx + label_w > img_w or ly + label_h > img_h:
                continue
            # Check overlap with existing labels
            overlap = False
            for (px, py) in placed_boxes:
                if (lx < px + label_w + 2 and lx + label_w + 2 > px and
                    ly < py + label_h + 2 and ly + label_h + 2 > py):
                    overlap = True
                    break
            if not overlap:
                best_pos = (lx, ly)
                break

        if best_pos is None:
            # If all positions overlap, use right side and offset vertically
            best_pos = (x2 + 5, y1 + idx * (label_h + 2))

        lx, ly = best_pos
        placed_boxes.append((lx, ly))

        # Get defect area and dimensions for this specific defect
        pixel_area = defect_areas[idx]
        label_line1 = f"{class_name}:{class_conf:.2f}"
        label_line2 = f"area={pixel_area}px {w}x{h}px"

        # Draw label with background
        cv2.rectangle(annotated, (lx, ly), (lx + label_w, ly + label_h), color, -1)
        # Line 1: class:conf
        cv2.putText(
            annotated, label_line1, (lx + 3, ly + th1 + 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )
        # Line 2: area and dimensions
        cv2.putText(
            annotated, label_line2, (lx + 3, ly + th1 * 2 + 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

        # Draw leader line from label to box center
        box_cx = x1 + w // 2
        box_cy = y1 + h // 2
        label_cx = lx + label_w // 2
        label_cy = ly + label_h // 2
        cv2.line(annotated, (box_cx, box_cy), (label_cx, label_cy), color, 1, cv2.LINE_AA)

    return annotated


# ============================================================
# Data discovery
# ============================================================
def discover_images_cls(data_root: Path, split: str = "train"):
    """Return list of (image_path, class_id) tuples for the given split.

    Reads the split file which has format "name class_id" per line.

    Args:
        data_root: Dataset root directory.
        split: Split name (e.g. "train", "val").

    Returns:
        List of (image_path, class_id) tuples, sorted by image name.
    """
    split_file = data_root / f"{split}.txt"
    img_dir = data_root / "images"
    masks_dir = data_root / "masks"

    if not split_file.is_file():
        # Fallback: scan directory without class info
        if img_dir.is_dir():
            exts = {".jpg", ".jpeg", ".png", ".bmp"}
            paths = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in exts])
            return [(p, -1) for p in paths]
        return []

    samples = []
    with open(split_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            name = parts[0]
            class_id_str = parts[1].split(",")[0]
            class_id = int(class_id_str)

            # Find image
            img_path = None
            for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                p = img_dir / f"{name}{ext}"
                if p.exists():
                    img_path = p
                    break
            if img_path is None:
                continue

            samples.append((img_path, class_id))

    return sorted(samples, key=lambda x: x[0].name)


def find_mask_cls(img_path: Path, masks_dir: Path, class_id: int) -> Path | None:
    """Find the GT mask corresponding to an image with class suffix.

    Tries {stem}_cls{class_id}.png first, then {stem}.png.

    Args:
        img_path: Path to the image file.
        masks_dir: Path to the masks directory.
        class_id: The class ID for the mask suffix.

    Returns:
        Path to the mask file, or None if not found.
    """
    stem = img_path.stem

    # Try class-specific mask first
    if class_id >= 0:
        for ext in (".png", ".jpg", ".jpeg"):
            p = masks_dir / f"{stem}_cls{class_id}{ext}"
            if p.exists():
                return p

    # Fallback to generic mask
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        p = masks_dir / f"{stem}{ext}"
        if p.exists():
            return p

    return None


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="SAM2 full-pipeline inference with LoRA + ClassHead classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to training checkpoint (e.g. .../best/checkpoint.pt)",
    )
    parser.add_argument(
        "--data-root", type=str, required=True,
        help="Dataset root containing images/, masks/, train.txt, and class_map.json",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for visualisation images",
    )
    parser.add_argument(
        "--split", type=str, default="train",
        help="Dataset split file to use (default: train)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for inference (default: cuda)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg.DEVICE = device  # override config

    print("=" * 60)
    print("SAM2 Full-Pipeline Inference (LoRA + ClassHead)")
    print("=" * 60)
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Data root  : {args.data_root}")
    print(f"Output     : {args.output}")
    print(f"Split      : {args.split}")
    print(f"Device     : {device}")
    print()

    # --- Class map ---
    IDX_TO_CLASS.update(load_class_map(data_root))
    if not IDX_TO_CLASS:
        # Hardcoded fallback matching class_map.json: {"0":0,"1":1,"3":2,"2":3,"4":4}
        IDX_TO_CLASS.update({
            0: "class0", 1: "class1", 2: "class3", 3: "class2", 4: "class4",
        })
    print(f"Class map  : {IDX_TO_CLASS}")
    print()

    # --- Model ---
    model, class_head = load_trained_model(args.checkpoint)
    print()

    # --- Data ---
    masks_dir = data_root / "masks"
    samples = discover_images_cls(data_root, args.split)
    if not samples:
        # Try train split as fallback
        samples = discover_images_cls(data_root, "train")
    if not samples:
        print("[ERROR] No images found.")
        return
    print(f"Found {len(samples)} images (split={args.split})")
    print()

    # --- Inference loop ---
    total_iou = 0.0
    iou_count = 0
    total_masks = 0
    cls_correct = 0
    cls_total = 0

    for idx, (img_path, class_id) in enumerate(samples):
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"  [{idx+1}/{len(samples)}] [skip] Cannot read: {img_path.name}")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # GT mask → point prompts
        mask_path = find_mask_cls(img_path, masks_dir, class_id)
        gt_mask = None
        if mask_path is not None:
            gt_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if gt_img is not None:
                gt_mask = (gt_img > 127).astype(np.uint8)

        if gt_mask is None:
            # No GT: use centre point as fallback
            point_coords = torch.tensor(
                [[[cfg.IMAGE_SIZE / 2, cfg.IMAGE_SIZE / 2]]], dtype=torch.float,
            )
            point_labels = torch.ones(1, 1, dtype=torch.long)
        else:
            point_coords, point_labels = sample_point_prompts(
                gt_mask, max_fg=6, max_bg=4, image_size=cfg.IMAGE_SIZE,
            )

        # Predict (segmentation + embeddings)
        pred_mask, iou_pred, image_embeddings = predict_full(
            model, image_rgb, point_coords, point_labels, cfg.IMAGE_SIZE,
        )

        # Classify
        class_name, class_idx, class_conf = classify(
            class_head, image_embeddings, IDX_TO_CLASS,
        )

        # IoU vs GT
        iou = 0.0
        if gt_mask is not None:
            # Ensure same size
            if gt_mask.shape != pred_mask.shape:
                gt_mask = cv2.resize(
                    gt_mask, (pred_mask.shape[1], pred_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            iou = compute_iou(pred_mask, gt_mask)
            total_iou += iou
            iou_count += 1

        # Classification accuracy (compare predicted class to GT class from filename)
        gt_class_name = f"class{class_id}" if class_id >= 0 else "unknown"
        cls_total += 1
        if class_name == gt_class_name:
            cls_correct += 1

        pixel_area = int(np.sum(pred_mask))
        total_masks += 1 if pixel_area > 0 else 0

        # Visualise
        annotated = plot_results(
            image_rgb, pred_mask,
            class_name=class_name, class_conf=class_conf, iou=iou,
        )
        save_path = output_dir / f"{img_path.stem}.jpg"
        cv2.imwrite(str(save_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

        iou_str = f"iou={iou:.3f}" if gt_mask is not None else "iou=N/A"
        gt_str = f"gt={gt_class_name}" if class_id >= 0 else "gt=N/A"
        print(
            f"  [{idx+1}/{len(samples)}] {img_path.name}  "
            f"pred={class_name}({class_conf:.2f}) {gt_str}  "
            f"area={pixel_area}px {iou_str}"
        )

    # Summary
    print()
    print("-" * 60)
    print("Done!")
    print(f"  Total images   : {len(samples)}")
    print(f"  With mask      : {total_masks}")
    if iou_count > 0:
        print(f"  Mean IoU       : {total_iou / iou_count:.4f} ({iou_count} with GT)")
    if cls_total > 0:
        print(f"  Cls accuracy   : {cls_correct}/{cls_total} = {cls_correct/cls_total:.4f}")
    print(f"  Output dir     : {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
