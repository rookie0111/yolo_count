#!/usr/bin/env python3
"""
SAM2 + LoRA + Classification Head inference script.

Loads a trained checkpoint containing LoRA adapter weights and a ClassHead,
then performs segmentation + classification on test images and visualizes
results with defect class labels and pixel dimensions.

Usage:
    python infer_sam2_cls.py \
        --checkpoint /home/finn/ultralytics-main-count/ultralytics/sam/weights/sam2_lora_cls_best.pt \
        --data-root /mnt/c/Users/finn.zhang/project/Datasets/xutong_military_sam2_cls \
        --output /home/finn/ultralytics-main-count/ultralytics/sam/results/inference_cls \
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
os.chdir(SAM2_ROOT)
sys.path.insert(0, SAM2_ROOT)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ultralytics", "sam"))

from hydra.core.global_hydra import GlobalHydra
GlobalHydra.instance().clear()


# ============================================================
# LoRA implementation (must match train_sam2_lora_cls.py)
# ============================================================
class LoRA_qkv(nn.Module):
    def __init__(self, qkv, linear_a_q, linear_b_q, linear_a_v, linear_b_v):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features

    def forward(self, x):
        qkv = self.qkv(x)
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, :self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv


def add_lora_to_sam2(model, r=8, device="cuda"):
    w_As = []
    w_Bs = []
    for param in model.image_encoder.parameters():
        param.requires_grad = False
    for blk in model.image_encoder.trunk.blocks:
        w_qkv_linear = blk.attn.qkv
        dim = w_qkv_linear.in_features
        w_a_q = nn.Linear(dim, r, bias=False).to(device)
        w_b_q = nn.Linear(r, dim, bias=False).to(device)
        w_a_v = nn.Linear(dim, r, bias=False).to(device)
        w_b_v = nn.Linear(r, dim, bias=False).to(device)
        nn.init.kaiming_uniform_(w_a_q.weight, a=5**0.5)
        nn.init.zeros_(w_b_q.weight)
        nn.init.kaiming_uniform_(w_a_v.weight, a=5**0.5)
        nn.init.zeros_(w_b_v.weight)
        w_As.extend([w_a_q, w_a_v])
        w_Bs.extend([w_b_q, w_b_v])
        blk.attn.qkv = LoRA_qkv(w_qkv_linear, w_a_q, w_b_q, w_a_v, w_b_v).to(device)
    return model, w_As, w_Bs


# ============================================================
# Classification head (must match train_sam2_lora_cls.py)
# ============================================================
class ClassHead(nn.Module):
    """Global feature classification head."""
    def __init__(self, in_channels, num_classes):
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
# Load SAM2 Base+ model
# ============================================================
def load_sam2_model(device="cuda"):
    """Build SAM2 Base+ model (same architecture as training)."""
    from sam2.modeling.sam2_base import SAM2Base
    from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
    from sam2.modeling.backbones.hieradet import Hiera
    from sam2.modeling.position_encoding import PositionEmbeddingSine
    from sam2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
    from sam2.modeling.sam.transformer import RoPEAttention
    from sam2.modeling.memory_encoder import MemoryEncoder, MaskDownSampler, Fuser, CXBlock

    print("Building SAM2 Base+ model ...")

    trunk = Hiera(
        embed_dim=112, num_heads=2, stages=[2, 3, 16, 3],
        global_att_blocks=[23, 33, 43],
        window_pos_embed_bkg_spatial_size=[14, 14],
        window_spec=[8, 4, 16, 8],
    )
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=256, normalize=True, scale=None, temperature=10000,
    )
    neck = FpnNeck(
        position_encoding=position_encoding, d_model=256,
        backbone_channel_list=[896, 448, 224, 112],
        fpn_top_down_levels=[2, 3], fpn_interp_model="nearest",
    )
    image_encoder = ImageEncoder(scalp=1, trunk=trunk, neck=neck)

    self_attn = RoPEAttention(
        rope_theta=10000.0, feat_sizes=[64, 64], embedding_dim=256,
        num_heads=1, downsample_rate=1, dropout=0.1,
    )
    cross_attn = RoPEAttention(
        rope_theta=10000.0, feat_sizes=[64, 64], rope_k_repeat=True,
        embedding_dim=256, num_heads=1, downsample_rate=1, dropout=0.1,
        kv_in_dim=64,
    )
    memory_attention_layer = MemoryAttentionLayer(
        activation="relu", dim_feedforward=2048, dropout=0.1,
        pos_enc_at_attn=False, self_attention=self_attn,
        d_model=256, pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False, cross_attention=cross_attn,
    )
    memory_attention = MemoryAttention(
        d_model=256, pos_enc_at_input=True,
        layer=memory_attention_layer, num_layers=4,
    )

    mask_downsampler = MaskDownSampler(kernel_size=3, stride=2, padding=1)
    fuser = Fuser(
        layer=CXBlock(
            dim=256, kernel_size=7, padding=3,
            layer_scale_init_value=1e-6, use_dwconv=True,
        ),
        num_layers=2,
    )
    memory_encoder = MemoryEncoder(
        out_dim=64,
        position_encoding=PositionEmbeddingSine(
            num_pos_feats=64, normalize=True, scale=None, temperature=10000,
        ),
        mask_downsampler=mask_downsampler, fuser=fuser,
    )

    model = SAM2Base(
        image_encoder=image_encoder, memory_attention=memory_attention,
        memory_encoder=memory_encoder, num_maskmem=7, image_size=1024,
        sigmoid_scale_for_mem_enc=20.0, sigmoid_bias_for_mem_enc=-10.0,
        use_mask_input_as_output_without_sam=True, directly_add_no_mem_embed=True,
        no_obj_embed_spatial=True, use_high_res_features_in_sam=True,
        multimask_output_in_sam=True, iou_prediction_use_sigmoid=True,
        use_obj_ptrs_in_encoder=True, add_tpos_enc_to_obj_ptrs=True,
        proj_tpos_enc_in_obj_ptrs=True, use_signed_tpos_enc_to_obj_ptrs=True,
        only_obj_ptrs_in_the_past_for_eval=True, pred_obj_scores=True,
        pred_obj_scores_mlp=True, fixed_no_obj_ptr=True,
        multimask_output_for_tracking=True, use_multimask_token_for_obj_ptr=True,
        multimask_min_pt_num=0, multimask_max_pt_num=1,
        use_mlp_for_obj_ptr_proj=True,
    )

    return model


def load_checkpoint_and_lora(checkpoint_path, model, device="cuda"):
    """Load LoRA weights and ClassHead from training checkpoint."""
    SAM2_CHECKPOINT = os.path.join(SAM2_ROOT, "checkpoints/sam2.1_hiera_base_plus.pt")

    # Load pretrained SAM2 weights first
    print(f"Loading pretrained weights: {SAM2_CHECKPOINT}")
    state_dict = torch.load(SAM2_CHECKPOINT, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model_dict = model.state_dict()
    pretrained_dict = {
        k: v for k, v in state_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    print(f"  Loaded {len(pretrained_dict)}/{len(model_dict)} pretrained layers")

    # Add LoRA adapters
    LORA_RANK = 8
    print(f"Applying LoRA (r={LORA_RANK}) ...")
    model, w_As, w_Bs = add_lora_to_sam2(model, r=LORA_RANK, device=device)

    # Load training checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Restore LoRA weights
    saved_w_As = ckpt["w_As"]
    saved_w_Bs = ckpt["w_Bs"]
    for i, w in enumerate(w_As):
        w.weight.data.copy_(saved_w_As[i])
    for i, w in enumerate(w_Bs):
        w.weight.data.copy_(saved_w_Bs[i])
    print(f"  Restored LoRA weights (epoch={ckpt.get('epoch', '?')}, loss={ckpt.get('loss', '?')})")

    # Build and load ClassHead
    num_classes = ckpt.get("num_classes", 5)
    class_head = ClassHead(in_channels=256, num_classes=num_classes).to(device)
    class_head.load_state_dict(ckpt["class_head"])
    class_head.eval()
    print(f"  Loaded ClassHead ({num_classes} classes)")

    # Build seg_head (Conv2d matching training behavior)
    seg_head = nn.Conv2d(256, 1, kernel_size=1).to(device)

    model.eval()
    model = model.to(device)
    return model, class_head, seg_head, num_classes


# ============================================================
# Visualization (same style as infer_sam2_full.py / infer_v2.py)
# ============================================================
# Reverse class map: internal index → original class ID
IDX_TO_CLASS_ID = {0: "0", 1: "1", 2: "3", 3: "2", 4: "4"}
CLASS_COLORS = [
    (0, 255, 0),     # green
    (0, 200, 255),   # orange-ish
    (255, 100, 0),   # blue-ish
    (200, 0, 255),   # purple
    (0, 255, 200),   # teal
]


def plot_results(
    image: np.ndarray,
    mask_binary: np.ndarray,
    pred_class_idx: int,
    num_classes: int = 5,
    merge_distance: int = 15,
):
    """Overlay mask, contours, bounding boxes, and class labels on the image.

    Each connected component (after morphological closing to merge nearby
    regions) is treated as one defect and gets its own bounding box with
    class name and pixel dimensions.

    Args:
        image: RGB image.
        mask_binary: Binary mask (0/1).
        pred_class_idx: Predicted class index from ClassHead.
        num_classes: Total number of classes.
        merge_distance: Max pixel gap to merge nearby components (default 15).
    """
    annotated = image.copy()

    pixel_area = int(np.sum(mask_binary))
    if pixel_area == 0:
        return annotated

    # Get class label string
    class_id_str = IDX_TO_CLASS_ID.get(pred_class_idx, str(pred_class_idx))
    class_label = f"class:{class_id_str}"
    color = CLASS_COLORS[pred_class_idx % len(CLASS_COLORS)]

    # Merge nearby disconnected components using morphological closing
    kernel_size = merge_distance * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    merged_mask = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)

    # Find connected components on merged mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        merged_mask, connectivity=8,
    )

    # Skip label 0 (background)
    defect_count = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 20:  # Filter tiny noise
            continue

        defect_count += 1

        # Bounding box for this defect
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        x1, y1, x2, y2 = x, y, x + w, y + h

        # Use original mask pixels within this merged region for overlay
        defect_mask_merged = (labels == i).astype(np.uint8)
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
        label = f"{class_label} {w}x{h}px"
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
def discover_samples(data_root: Path, split: str = "train"):
    """Return list of (name, class_ids) tuples from the split file."""
    split_file = data_root / f"{split}.txt"
    if not split_file.is_file():
        print(f"[WARN] Split file not found: {split_file}")
        return []

    samples = []
    with open(split_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                name = parts[0]
                class_ids = [int(c) for c in parts[1].split(",")]
                samples.append((name, class_ids))
    return samples


def find_image_path(img_dir: Path, name: str) -> Path | None:
    """Find image file by name, trying common extensions."""
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        p = img_dir / f"{name}{ext}"
        if p.exists():
            return p
    return None


def find_mask_path(mask_dir: Path, name: str, class_id: int) -> Path | None:
    """Find mask file for given name and class_id."""
    p = mask_dir / f"{name}_cls{class_id}.png"
    if p.exists():
        return p
    return None


# ============================================================
# Inference
# ============================================================
@torch.no_grad()
def predict(
    model: nn.Module,
    class_head: ClassHead,
    seg_head: nn.Conv2d,
    image_rgb: np.ndarray,
    device: torch.device,
    image_size: int = 1024,
):
    """Run segmentation + classification inference.

    Args:
        model: SAM2 model with LoRA adapters.
        class_head: Classification head.
        seg_head: Segmentation Conv2d head.
        image_rgb: Input image as RGB numpy array (H, W, 3).
        device: Torch device.
        image_size: Model input resolution.

    Returns:
        Tuple of (pred_mask, pred_class_idx, cls_probs) where pred_mask is
        a binary numpy array at original resolution, pred_class_idx is the
        argmax class index, and cls_probs is the raw logits tensor.
    """
    h_orig, w_orig = image_rgb.shape[:2]

    # Resize and normalise
    img_resized = cv2.resize(image_rgb, (image_size, image_size))
    img_tensor = (
        torch.from_numpy(img_resized)
        .float()
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
        / 255.0
    )

    with torch.no_grad():
        # Image encoder forward
        backbone_out = model.image_encoder(img_tensor)

        if isinstance(backbone_out, dict):
            feat = backbone_out["vision_features"]
        else:
            feat = backbone_out

        # Segmentation: upsample features → seg_head → sigmoid → threshold
        feat_up = F.interpolate(
            feat.float(), size=(image_size, image_size),
            mode="bilinear", align_corners=False,
        )
        seg_logits = seg_head(feat_up).squeeze(1)  # (B, H, W)
        seg_probs = torch.sigmoid(seg_logits).squeeze(0)  # (H, W)

        # Classification
        cls_logits = class_head(feat.float())  # (B, num_classes)
        cls_probs = torch.sigmoid(cls_logits).squeeze(0)  # (num_classes,)

    # Threshold segmentation
    mask_np = seg_probs.float().cpu().numpy()
    mask_binary = (mask_np > 0.5).astype(np.uint8)

    # Resize back to original resolution
    if (h_orig, w_orig) != (image_size, image_size):
        mask_binary = cv2.resize(
            mask_binary, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST,
        )

    # Predicted class (argmax)
    pred_class_idx = int(cls_probs.argmax().cpu().item())

    return mask_binary, pred_class_idx, cls_probs.cpu()


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="SAM2 + LoRA + ClassHead inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to training checkpoint (sam2_lora_cls_best.pt)",
    )
    parser.add_argument(
        "--data-root", type=str, required=True,
        help="Dataset root containing images/, masks/, and split .txt files",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for annotated images",
    )
    parser.add_argument(
        "--split", type=str, default="train",
        help="Dataset split file to use (default: train)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for inference (default: cuda)",
    )
    parser.add_argument(
        "--merge-distance", type=int, default=15,
        help="Morphological closing kernel size for merging nearby defects (default: 15)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("SAM2 + LoRA + ClassHead Inference")
    print("=" * 60)
    print(f"Checkpoint    : {args.checkpoint}")
    print(f"Data root     : {args.data_root}")
    print(f"Output        : {args.output}")
    print(f"Split         : {args.split}")
    print(f"Device        : {device}")
    print(f"Merge distance: {args.merge_distance}")
    print()

    # --- Load model ---
    model = load_sam2_model(device)
    model, class_head, seg_head, num_classes = load_checkpoint_and_lora(
        args.checkpoint, model, device,
    )
    print()

    # --- Discover data ---
    samples = discover_samples(data_root, args.split)
    if not samples:
        print("[ERROR] No samples found.")
        return
    print(f"Found {len(samples)} samples (split={args.split})")
    print()

    img_dir = data_root / "images"
    mask_dir = data_root / "masks"

    # --- Inference loop ---
    total_masks = 0
    class_counts = {}

    for idx, (name, class_ids) in enumerate(samples):
        # Load image
        img_path = find_image_path(img_dir, name)
        if img_path is None:
            print(f"  [{idx+1}/{len(samples)}] [skip] Image not found: {name}")
            continue

        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"  [{idx+1}/{len(samples)}] [skip] Cannot read: {img_path.name}")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Run inference
        pred_mask, pred_class_idx, cls_probs = predict(
            model, class_head, seg_head, image_rgb, device,
        )

        pixel_area = int(np.sum(pred_mask))
        total_masks += 1 if pixel_area > 0 else 0

        # Track class predictions
        pred_class_id = IDX_TO_CLASS_ID.get(pred_class_idx, str(pred_class_idx))
        class_counts[pred_class_id] = class_counts.get(pred_class_id, 0) + 1

        # Build probability string
        prob_str = " ".join(
            f"c{IDX_TO_CLASS_ID.get(i, i)}={cls_probs[i]:.2f}"
            for i in range(num_classes)
        )

        # Visualise
        annotated = plot_results(
            image_rgb, pred_mask, pred_class_idx,
            num_classes=num_classes, merge_distance=args.merge_distance,
        )
        save_path = output_dir / f"{name}.jpg"
        cv2.imwrite(str(save_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

        gt_cls_str = ",".join(str(c) for c in class_ids)
        print(
            f"  [{idx+1}/{len(samples)}] {img_path.name}  "
            f"gt_cls=[{gt_cls_str}] pred_cls={pred_class_id}  "
            f"area={pixel_area}px  [{prob_str}]"
        )

    # Summary
    print()
    print("-" * 60)
    print("Done!")
    print(f"  Total samples : {len(samples)}")
    print(f"  With mask     : {total_masks}")
    print(f"  Class distribution (predicted):")
    for cid in sorted(class_counts.keys()):
        print(f"    class:{cid}  →  {class_counts[cid]} images")
    print(f"  Output dir    : {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
