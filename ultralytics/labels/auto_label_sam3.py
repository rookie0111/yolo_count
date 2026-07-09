#!/usr/bin/env python3
"""
Step 2: SAM3 auto-labeling using ONNX Runtime only (no PyTorch needed).

Saves YOLO OBB format annotations: class_id x1 y1 x2 y2 x3 y3 x4 y4
All coordinates normalized [0, 1], 4 corners top-left→top-right→bottom-right→bottom-left.

Usage:  python auto_label_sam3.py
"""
import os
import glob
import time
import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────

BACKBONE_ONNX  = "/home/finn/ultralytics-main-count/sam3_backbone.onnx"
DETECTOR_ONNX  = "/home/finn/ultralytics-main-count/sam3_detector.onnx"
TEXT_EMB_PATH  = "/home/finn/ultralytics-main-count/ultralytics/labels/sam3_text_emb.npz"

IMAGE_DIR = "/mnt/c/Users/finn.zhang/project/Datasets/yolo_zhonghua_counter/images/val"
LABEL_DIR = "/mnt/c/Users/finn.zhang/project/Datasets/yolo_zhonghua_counter/labels/val"

CONF_THRES = 0.25
IOU_THRES  = 0.50
IMGSZ      = 1008


def load_sessions():
    """Load ONNX Runtime sessions with CUDA provider."""
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    print("Loading backbone ONNX...")
    bb = ort.InferenceSession(BACKBONE_ONNX, providers=providers)
    print("Loading detector ONNX...")
    det = ort.InferenceSession(DETECTOR_ONNX, providers=providers)
    return bb, det


def load_text_embeddings():
    """Load pre-computed text embeddings."""
    data = np.load(TEXT_EMB_PATH)
    feats = data["language_features"].astype(np.float32)  # (32, 1, 256)
    mask  = data["language_mask"].astype(bool)             # (1, 32)
    print(f"Text embeddings: feats={feats.shape} mask={mask.shape}")
    return feats, mask


def preprocess(image_path):
    """Resize to 1008x1008, normalize to [-1,1], HWC→CHW→NCHW."""
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    h0, w0 = img.shape[:2]
    resized = cv2.resize(img, (IMGSZ, IMGSZ))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    normalized = (rgb - 127.5) / 127.5           # [-1, 1]
    blob = normalized.transpose(2, 0, 1)[None]   # (1, 3, IMGSZ, IMGSZ)
    return blob, (h0, w0)


def infer(bb_sess, det_sess, blob, lang_feats, lang_mask):
    """Two-stage SAM3 inference."""
    # ── Stage 1: backbone ──
    bb_outs = bb_sess.run(None, {"image": blob.astype(np.float32)})
    names = [o.name for o in bb_sess.get_outputs()]
    bb = {n: bb_outs[i] for i, n in enumerate(names)}

    # ── Stage 2: detector ──
    det_in = {
        "vision_pos_enc": bb["vision_pos_enc_2"].astype(np.float32),
        "backbone_fpn_0": bb["backbone_fpn_0"].astype(np.float32),
        "backbone_fpn_1": bb["backbone_fpn_1"].astype(np.float32),
        "backbone_fpn_2": bb["backbone_fpn_2"].astype(np.float32),
        "language_features": lang_feats,
        "language_mask": lang_mask,
        "text_ids": np.array([0], dtype=np.int64),
    }
    det_outs = det_sess.run(None, det_in)
    return det_outs[0], det_outs[1], det_outs[3]  # boxes, logits, presence


def nms_xyxy(boxes, scores, iou_thres):
    """IoU-based NMS on xyxy boxes."""
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_o = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        iou = inter / (area_i + area_o - inter + 1e-6)
        order = order[np.where(iou <= iou_thres)[0] + 1]
    return np.array(keep, dtype=np.int64)


def process_image(image_path, bb_sess, det_sess, lang_feats, lang_mask):
    """Run SAM3 on a single image, return YOLO-OBB label lines."""
    blob, (h0, w0) = preprocess(image_path)
    if blob is None:
        return []

    boxes, logits, presence = infer(bb_sess, det_sess, blob, lang_feats, lang_mask)

    # Sigmoid
    scores = 1.0 / (1.0 + np.exp(-logits[:, :, 0]))
    presence_s = 1.0 / (1.0 + np.exp(-presence[:, 0]))
    final = scores * presence_s[:, None]
    keep = final[0] > CONF_THRES
    if not keep.any():
        return []

    b = boxes[0, keep]   # (K, 4) cxcywh normalized
    s = final[0, keep]

    # Convert to xyxy for NMS
    xyxy = np.column_stack([
        b[:, 0] - b[:, 2] / 2,
        b[:, 1] - b[:, 3] / 2,
        b[:, 0] + b[:, 2] / 2,
        b[:, 1] + b[:, 3] / 2,
    ])
    idx = nms_xyxy(xyxy, s, IOU_THRES)
    xyxy = xyxy[idx]

    labels = []
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i]
        # Clamp
        x1, y1 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1))
        x2, y2 = max(0.0, min(1.0, x2)), max(0.0, min(1.0, y2))
        if x2 - x1 < 0.001 or y2 - y1 < 0.001:
            continue
        # YOLO OBB: class x1 y1 x2 y2 x3 y3 x4 y4  (4 corners clockwise)
        # Axis-aligned box → 4 corners
        obb = [0, x1, y1, x2, y1, x2, y2, x1, y2]
        labels.append(obb)
    return labels


def main():
    os.makedirs(LABEL_DIR, exist_ok=True)

    lang_feats, lang_mask = load_text_embeddings()
    bb_sess, det_sess = load_sessions()

    images = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")) +
                    glob.glob(os.path.join(IMAGE_DIR, "*.png")))
    print(f"\nFound {len(images)} images")
    print(f"Saving labels to: {LABEL_DIR}\n")

    total_boxes = 0
    t0 = time.time()

    for i, img_path in enumerate(images):
        basename = Path(img_path).stem
        label_path = os.path.join(LABEL_DIR, basename + ".txt")

        # Skip if already labeled
        if os.path.exists(label_path):
            continue

        try:
            labels = process_image(img_path, bb_sess, det_sess, lang_feats, lang_mask)

            with open(label_path, 'w') as f:
                for obb in labels:
                    f.write(" ".join(f"{v:.6f}" for v in obb) + "\n")

            total_boxes += len(labels)

            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                fps = (i + 1) / elapsed
                print(f"  [{i+1}/{len(images)}] {basename}: {len(labels)} boxes  ({fps:.1f} img/s, {elapsed:.0f}s)")

        except Exception as e:
            print(f"  [!] Error on {img_path}: {e}")

    elapsed = time.time() - t0
    print(f"\nDone! {len(images)} images in {elapsed:.0f}s ({len(images)/elapsed:.1f} img/s)")
    print(f"Total boxes: {total_boxes}")


if __name__ == "__main__":
    main()
