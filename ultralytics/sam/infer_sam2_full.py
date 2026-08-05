#!/usr/bin/env python3
"""Batch inference with the FULL SAM2 pipeline (prompt encoder + mask decoder).

Uses a training checkpoint that contains LoRA adapter weights.  For each image,
foreground/background point prompts are sampled from the GT mask and fed through
the prompt encoder → mask decoder path.

Usage:
    python infer_sam2_full.py \
        --checkpoint /path/to/output_xutong_base/best/checkpoint.pt \
        --data-root /mnt/c/Users/finn.zhang/project/Datasets/xutong_military_sam2 \
        --output /path/to/output
"""

import sys
import os
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
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
def load_trained_model(checkpoint_path: str) -> torch.nn.Module:
    """Build SAM2, load pretrained weights, apply LoRA, then restore training checkpoint."""
    print("Building SAM2 model ...")
    model = build_model()

    print(f"Loading pretrained weights: {cfg.CHECKPOINT_FILE}")
    load_checkpoint(model, str(cfg.CHECKPOINT_FILE))

    print("Applying LoRA adapters ...")
    model = apply_lora(model)

    # Restore training checkpoint (LoRA adapter weights)
    print(f"Loading training checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu")
    model.image_encoder.trunk.load_state_dict(state["lora_weights"])
    print(
        f"  LoRA weights restored from epoch {state.get('epoch', '?')} "
        f"(loss={state.get('loss', '?')})"
    )

    model = model.to(cfg.DEVICE)
    model.eval()
    return model


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
        Tuple of ``(pred_mask, iou_pred)`` where *pred_mask* is a binary numpy
        array at the original image resolution and *iou_pred* is a float.
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

    return mask_binary, iou_pred


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
# Visualization (same style as infer_v2.py / infer_sam2_lora.py)
# ============================================================
def plot_results(
    image: np.ndarray,
    mask_binary: np.ndarray,
    iou: float = 0.0,
    merge_distance: int = 15,
):
    """Overlay mask, contours, bounding box, and label on the image.

    Merges nearby disconnected components that belong to the same defect
    using morphological closing, then treats each merged region as one defect.

    Args:
        image: RGB image.
        mask_binary: Binary mask (0/1).
        iou: IoU score for display.
        merge_distance: Max pixel gap to merge nearby components (default 15).
    """
    annotated = image.copy()
    colors = [(0, 255, 0), (0, 200, 255), (255, 100, 0), (200, 0, 255), (0, 255, 200)]

    pixel_area = int(np.sum(mask_binary))
    if pixel_area == 0:
        return annotated

    # Merge nearby disconnected components using morphological closing
    # Closing = dilation then erosion, fills small gaps between nearby regions
    kernel_size = merge_distance * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    merged_mask = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)

    # Find connected components on merged mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        merged_mask, connectivity=8
    )

    # Skip label 0 (background)
    defect_count = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 20:  # Filter tiny noise
            continue

        defect_count += 1
        color = colors[(defect_count - 1) % len(colors)]

        # Bounding box for this defect
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        x1, y1, x2, y2 = x, y, x + w, y + h

        # Use original mask pixels within this merged region for overlay
        defect_mask_merged = (labels == i).astype(np.uint8)
        # Intersect with original mask to keep only actual defect pixels
        defect_mask = cv2.bitwise_and(mask_binary, defect_mask_merged)

        # Mask overlay for this defect
        alpha = 0.4
        colored = np.zeros_like(image)
        colored[defect_mask == 1] = color
        region = defect_mask == 1
        annotated[region] = (
            annotated[region] * (1 - alpha) + colored[region] * alpha
        ).astype(np.uint8)

        # Contours for this defect
        contours, _ = cv2.findContours(
            defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(annotated, contours, -1, color, 2)

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label: defect class + pixel dimensions
        label = f"defect:{defect_count} {w}x{h}px"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            annotated, label, (x1 + 3, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )

    return annotated


# ============================================================
# Data discovery
# ============================================================
def discover_images(data_root: Path, split: str = "val"):
    """Return list of image paths for the given split."""
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

    # Fallback: scan directory
    if img_dir.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        return sorted([f for f in img_dir.iterdir() if f.suffix.lower() in exts])

    return []


def find_mask(img_path: Path, masks_dir: Path) -> Path | None:
    """Find the GT mask corresponding to an image."""
    stem = img_path.stem
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
        description="SAM2 full-pipeline inference with LoRA adapters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to training checkpoint (e.g. .../best/checkpoint.pt)",
    )
    parser.add_argument(
        "--data-root", type=str, required=True,
        help="Dataset root containing images/ and masks/ subdirectories",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for visualisation images",
    )
    parser.add_argument(
        "--split", type=str, default="val",
        help="Dataset split file to use (default: val, falls back to train)",
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
    print("SAM2 Full-Pipeline Inference (prompt encoder + mask decoder)")
    print("=" * 60)
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Data root  : {args.data_root}")
    print(f"Output     : {args.output}")
    print(f"Device     : {device}")
    print()

    # --- Model ---
    model = load_trained_model(args.checkpoint)
    print()

    # --- Data ---
    masks_dir = data_root / "masks"
    image_paths = discover_images(data_root, args.split)
    if not image_paths:
        # Try train split as fallback
        image_paths = discover_images(data_root, "train")
    if not image_paths:
        print("[ERROR] No images found.")
        return
    print(f"Found {len(image_paths)} images (split={args.split})")
    print()

    # --- Inference loop ---
    total_iou = 0.0
    iou_count = 0
    total_masks = 0

    for idx, img_path in enumerate(image_paths):
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"  [{idx+1}/{len(image_paths)}] [skip] Cannot read: {img_path.name}")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # GT mask → point prompts
        mask_path = find_mask(img_path, masks_dir)
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

        # Predict
        pred_mask, iou_pred = predict_full(
            model, image_rgb, point_coords, point_labels, cfg.IMAGE_SIZE,
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

        pixel_area = int(np.sum(pred_mask))
        total_masks += 1 if pixel_area > 0 else 0

        # Visualise
        annotated = plot_results(image_rgb, pred_mask, iou)
        save_path = output_dir / f"{img_path.stem}.jpg"
        cv2.imwrite(str(save_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

        iou_str = f"iou={iou:.3f}" if gt_mask is not None else "iou=N/A"
        print(
            f"  [{idx+1}/{len(image_paths)}] {img_path.name}  "
            f"area={pixel_area}px {iou_str}"
        )

    # Summary
    print()
    print("-" * 60)
    print("Done!")
    print(f"  Total images : {len(image_paths)}")
    print(f"  With mask    : {total_masks}")
    if iou_count > 0:
        print(f"  Mean IoU     : {total_iou / iou_count:.4f} ({iou_count} with GT)")
    print(f"  Output dir   : {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
