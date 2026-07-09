#!/usr/bin/env python3
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     SAM3 → YOLO 全流程自动化管线                              ║
║              Auto-label → Dataset → Train → ONNX Export                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

────────────────────────────────────────────────────────────────────────────────
 概述
────────────────────────────────────────────────────────────────────────────────

本脚本实现从零标注到ONNX导出的完整YOLO训练管线，核心流程：

    [未标注图像] ──SAM3 ONNX──► [YOLO标注] ──Ultralytics──► [.pt模型] ──导出──► [.onnx]

每条子命令可独立运行，也支持 `all` 一键完成全流程。

────────────────────────────────────────────────────────────────────────────────
 支持的任务类型
────────────────────────────────────────────────────────────────────────────────

  detect  水平边界框 (Horizontal Bounding Box)
          标签格式: class_id cx cy w h  (YOLO归一化中心宽高格式)
          预训练模型: yolo11n.pt, yolo11s.pt, yolov8n.pt, yolo26n.pt ...

  obb     旋转边界框 (Oriented Bounding Box)
          标签格式: class_id x1 y1 x2 y2 x3 y3 x4 y4  (4角点归一化)
          预训练模型: yolo11n-obb.pt, yolo11s-obb.pt, yolo26n-obb.pt ...

  segment 实例分割 (Instance Segmentation)
          标签格式: class_id x1 y1 x2 y2 ... (多边形顶点归一化)
          预训练模型: yolo11n-seg.pt, yolo11s-seg.pt, yolo26n-seg.pt ...

  track   目标追踪 (基于detect模型, 推理时使用.track()方法)
          训练与detect相同, 推理时切换为 tracking 模式即可

────────────────────────────────────────────────────────────────────────────────
 数据集目录结构
────────────────────────────────────────────────────────────────────────────────

训练前需按以下结构组织数据（label 命令会自动生成 labels 目录）：

    dataset_root/
    ├── data.yaml                # 数据集配置（由 data 命令生成）
    ├── images/
    │   ├── train/               # 训练集图像
    │   │   ├── img_001.jpg
    │   │   └── ...
    │   └── val/                 # 验证集图像
    │       ├── img_101.jpg
    │       └── ...
    └── labels/
        ├── train/               # 训练集标注（人工或SAM3生成）
        │   ├── img_001.txt
        │   └── ...
        └── val/                 # 验证集标注（SAM3自动生成）
            ├── img_101.txt
            └── ...

data.yaml 格式（自动生成）:
    path: /absolute/path/to/dataset_root
    train: images/train
    val: images/val
    nc: 1
    names:
      0: bag

────────────────────────────────────────────────────────────────────────────────
 SAM3 ONNX 推理管线
────────────────────────────────────────────────────────────────────────────────

本脚本的 label 命令使用两阶段 ONNX 推理（无需加载 PyTorch SAM3 模型）:

  [图像] → [sam3_backbone.onnx] → vision features ─┐
                                                      ├→ [sam3_detector.onnx] → boxes
  [文本] → [precompute] → text_embeddings.npz ───────┘

前置条件:
  - sam3_backbone.onnx (1.8GB) — 需先在 tools/export_sam3_onnx.py 导出
  - sam3_detector.onnx (124MB) — 需先在 tools/export_sam3_detector_onnx.py 导出
  - sam3_text_emb.npz            — 由 precompute 命令生成

────────────────────────────────────────────────────────────────────────────────
 依赖
────────────────────────────────────────────────────────────────────────────────

  核心:  ultralytics, torch, onnxruntime-gpu, opencv-python, numpy, pyyaml
  可选:  onnx (模型验证)

  安装:  pip install ultralytics onnxruntime-gpu opencv-python pyyaml

────────────────────────────────────────────────────────────────────────────────
 使用示例
────────────────────────────────────────────────────────────────────────────────

  # ── 0. 预计算文本向量（首次运行一次，对应每种文本提示） ──
  python pipeline.py precompute --prompt "bag on the conveyor belt"

  # ── 1. 自动标注（OBB格式，用于旋转框训练） ──
  python pipeline.py label \
      --images /data/images/val \
      --labels /data/labels/val \
      --prompt "bag on the conveyor belt" \
      --label-task obb \
      --conf 0.25 --iou 0.50

  # ── 2. 生成数据集YAML ──
  python pipeline.py data \
      --dataset-root /data \
      --names "fertilizer bag"

  # ── 3a. 训练 OBB 模型 ──
  python pipeline.py train \
      --data /data/data.yaml \
      --task obb \
      --model yolo26n-obb.pt \
      --epochs 100 --batch 16 --imgsz 640

  # ── 3b. 训练检测(HBB)模型 ──
  python pipeline.py train \
      --data /data/data.yaml \
      --task detect \
      --model yolo11n.pt \
      --epochs 100 --batch 16

  # ── 4. 导出 ONNX（best.onnx 放置在 best.pt 同级目录） ──
  python pipeline.py export \
      --weights runs/obb/train/weights/best.pt \
      --task obb --half --dynamic

  # ── 一键全流程 ──
  python pipeline.py all \
      --images /data/images/val \
      --labels /data/labels/val \
      --dataset-root /data \
      --task obb \
      --epochs 100 \
      --prompt "bag on the conveyor belt" \
      --names "fertilizer bag"

  # ── 训练后推理 ──
  yolo predict task=obb model=runs/obb/train/weights/best.pt source=test.jpg
  yolo predict task=obb model=runs/obb/train/weights/best.onnx source=test.jpg half

  # ── 追踪（使用 detection 模型 + .track()）──
  yolo track model=runs/detect/train/weights/best.pt source=video.mp4 show=True

────────────────────────────────────────────────────────────────────────────────
 注意事项
────────────────────────────────────────────────────────────────────────────────

  1. SAM3 推理速度较慢（~10秒/图，Conv层CUDA fallback），适合离线批量标注
  2. 网络受限环境请使用本地预训练模型（--model /path/to/local.pt）
  3. OBB标签与HBB标签格式不同，label命令的 --label-task 参数决定生成哪种
  4. segment 任务需要多边形标注，SAM3的mask输出需额外处理
  5. ONNX 导出默认放置在 .pt 同级目录
  6. tracker 训练与 detection 完全相同，推理时使用 model.track() 即可
"""

import os
import sys
import glob
import time
import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple

import yaml
import numpy as np
import cv2

# ── Paths ────────────────────────────────────────────────────────────────────

BACKBONE_ONNX  = "/home/finn/ultralytics-main-count/sam3_backbone.onnx"
DETECTOR_ONNX  = "/home/finn/ultralytics-main-count/sam3_detector.onnx"
SAM3_PT_PATH   = "/home/finn/ultralytics-main-count/sam3.pt"
TEXT_EMB_DIR   = "/home/finn/ultralytics-main-count/ultralytics/labels"

# ── Task configs ─────────────────────────────────────────────────────────────

TASK_MODELS = {
    "detect":  "yolo11n.pt",
    "obb":     "yolo11n-obb.pt",
    "segment": "yolo11n-seg.pt",
}

TASK_SUFFIX = {
    "detect":  "detect",
    "obb":     "obb",
    "segment": "segment",
}

HBB_MODELS = ["yolo11n.pt", "yolov8n.pt", "yolo26n.pt",
              "yolo11s.pt", "yolov8s.pt", "yolo26s.pt",
              "yolo11m.pt", "yolov8m.pt", "yolo26m.pt",
              "yolo11l.pt", "yolov8l.pt", "yolo26l.pt",
              "yolo11x.pt", "yolov8x.pt", "yolo26x.pt",]
OBB_MODELS = [m.replace(".pt", "-obb.pt") for m in HBB_MODELS]
SEG_MODELS = [m.replace(".pt", "-seg.pt") for m in HBB_MODELS]


# ══════════════════════════════════════════════════════════════════════════════
#  Pre-compute text embeddings
# ══════════════════════════════════════════════════════════════════════════════

def cmd_precompute(args):
    """Pre-compute SAM3 text embeddings for a text prompt.

    This step runs ONCE per text prompt. The PyTorch SAM3 model is loaded
    to encode the text, then embeddings are saved as .npz for reuse by
    the ONNX-only label command.
    """
    import torch
    from ultralytics.models.sam.build_sam3 import build_sam3_image_model
    # ... rest unchanged ...

    prompt = args.prompt
    safe_name = prompt.replace(" ", "_").replace("/", "_")[:40]
    out_path = os.path.join(TEXT_EMB_DIR, f"sam3_emb_{safe_name}.npz")

    print(f"Text prompt: '{prompt}'")
    print("Loading SAM3 model...")
    model = build_sam3_image_model(SAM3_PT_PATH)
    model.eval().cuda()

    with torch.no_grad():
        model.set_classes([prompt])

    feats = model.text_embeddings["language_features"].detach().cpu().numpy()
    mask  = model.text_embeddings["language_mask"].detach().cpu().numpy()

    np.savez_compressed(out_path, language_features=feats, language_mask=mask)
    print(f"Saved: {out_path}")
    print(f"  features: {feats.shape} dtype={feats.dtype}")
    print(f"  mask:     {mask.shape} dtype={mask.dtype}")

    del model
    torch.cuda.empty_cache()
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
#  Auto-label (SAM3 ONNX pipeline)
# ══════════════════════════════════════════════════════════════════════════════

def cmd_label(args):
    """Auto-label images using SAM3 two-stage ONNX pipeline.

    Pipeline: image → backbone.onnx → vision features → detector.onnx → boxes

    The text embeddings (.npz) are loaded once at startup. Each image is
    processed independently and labels are written to disk immediately.
    Existing labels are skipped unless --force is set.

    Label format is controlled by --label-task:
      - hbb: class_id cx cy w h  (YOLO detect format, 5 values)
      - obb: class_id x1 y1 x2 y2 x3 y3 x4 y4  (YOLO OBB format, 9 values)
    """
    import onnxruntime as ort

    image_dir = args.images
    label_dir = args.labels
    prompt = args.prompt
    conf_thres = args.conf
    iou_thres = args.iou
    task = args.label_task  # obb or hbb
    imgsz = args.imgsz

    # Resolve text embedding file
    safe_name = prompt.replace(" ", "_").replace("/", "_")[:40]
    if args.emb:
        emb_path = args.emb
    else:
        emb_path = os.path.join(TEXT_EMB_DIR, f"sam3_emb_{safe_name}.npz")

    if not os.path.exists(emb_path):
        print(f"Text embeddings not found: {emb_path}")
        print("Run: python pipeline.py precompute --prompt \"{prompt}\"")
        return

    os.makedirs(label_dir, exist_ok=True)

    # Load text embeddings
    data = np.load(emb_path)
    lang_feats = data["language_features"].astype(np.float32)
    lang_mask  = data["language_mask"].astype(bool)

    # Load ONNX sessions
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    print("Loading ONNX sessions...")
    bb_sess = ort.InferenceSession(BACKBONE_ONNX, providers=providers)
    det_sess = ort.InferenceSession(DETECTOR_ONNX, providers=providers)

    # Get image list
    images = sorted(
        glob.glob(os.path.join(image_dir, "*.jpg")) +
        glob.glob(os.path.join(image_dir, "*.png")) +
        glob.glob(os.path.join(image_dir, "*.jpeg"))
    )
    print(f"Found {len(images)} images")
    print(f"Task: {task}  conf={conf_thres}  iou={iou_thres}")
    print(f"Text prompt: '{prompt}'\n")

    total_boxes = 0
    t0 = time.time()
    skipped = 0

    for i, img_path in enumerate(images):
        basename = Path(img_path).stem
        label_path = os.path.join(label_dir, basename + ".txt")

        if os.path.exists(label_path) and not args.force:
            skipped += 1
            continue

        try:
            labels = _sam3_detect(img_path, bb_sess, det_sess, lang_feats, lang_mask,
                                  conf_thres, iou_thres, imgsz, task)

            with open(label_path, 'w') as f:
                for line in labels:
                    f.write(line + "\n")

            total_boxes += len(labels)

            if (i + 1) % 10 == 0 or i < 3:
                elapsed = time.time() - t0
                rate = (i + 1 - skipped) / max(elapsed, 0.1)
                print(f"  [{i+1}/{len(images)}] {basename}: {len(labels)} boxes  "
                      f"({rate:.1f} img/s, {elapsed:.0f}s)")

        except Exception as e:
            print(f"  [!] {img_path}: {e}")

    elapsed = time.time() - t0
    processed = len(images) - skipped
    print(f"\nDone: {processed} images in {elapsed:.0f}s "
          f"({processed/max(elapsed,0.1):.1f} img/s)")
    print(f"Total boxes: {total_boxes}  (skipped: {skipped})")


# ── SAM3 inference helpers ──────────────────────────────────────────────────

def _sam3_detect(image_path, bb_sess, det_sess, lang_feats, lang_mask,
                 conf_thres, iou_thres, imgsz, task):
    """Run SAM3 on single image, return label lines in specified format."""
    img = cv2.imread(image_path)
    if img is None:
        return []

    h0, w0 = img.shape[:2]
    resized = cv2.resize(img, (imgsz, imgsz))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    blob = ((rgb - 127.5) / 127.5).transpose(2, 0, 1)[None]

    # Stage 1: backbone
    bb_outs = bb_sess.run(None, {"image": blob.astype(np.float32)})
    names = [o.name for o in bb_sess.get_outputs()]
    bb = {n: bb_outs[i] for i, n in enumerate(names)}

    # Stage 2: detector
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
    boxes, logits, presence = det_outs[0], det_outs[1], det_outs[3]

    # Sigmoid
    scores = 1.0 / (1.0 + np.exp(-logits[:, :, 0]))
    presence_s = 1.0 / (1.0 + np.exp(-presence[:, 0]))
    final = scores * presence_s[:, None]
    keep = final[0] > conf_thres
    if not keep.any():
        return []

    b = boxes[0, keep]
    s = final[0, keep]

    xyxy = _cxcywh_to_xyxy(b)
    idx = _nms(xyxy, s, iou_thres)
    xyxy = xyxy[idx]

    labels = []
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i]
        x1, y1 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1))
        x2, y2 = max(0.0, min(1.0, x2)), max(0.0, min(1.0, y2))
        if x2 - x1 < 0.001 or y2 - y1 < 0.001:
            continue

        if task == "obb":
            # YOLO OBB: class x1 y1 x2 y2 x3 y3 x4 y4 (4 corners)
            line = f"0 {x1:.6f} {y1:.6f} {x2:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x1:.6f} {y2:.6f}"
        else:
            # YOLO HBB: class cx cy w h (center format)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = x2 - x1, y2 - y1
            line = f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        labels.append(line)
    return labels


def _cxcywh_to_xyxy(b):
    return np.column_stack([
        b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
        b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2,
    ])


def _nms(boxes, scores, iou_thres):
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


# ══════════════════════════════════════════════════════════════════════════════
#  Generate dataset YAML
# ══════════════════════════════════════════════════════════════════════════════

def cmd_data(args):
    """Generate dataset YAML config file."""
    dataset_root = args.dataset_root
    class_names = args.names

    config = {
        "path": dataset_root,
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_names),
        "names": {i: name for i, name in enumerate(class_names)},
    }
    out_path = args.output or os.path.join(dataset_root, "data.yaml")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Saved: {out_path}")
    print(f"  train: {dataset_root}/{config['train']}")
    print(f"  val:   {dataset_root}/{config['val']}")
    print(f"  nc:    {config['nc']}")
    print(f"  names: {config['names']}")


# ══════════════════════════════════════════════════════════════════════════════
#  Train
# ══════════════════════════════════════════════════════════════════════════════

def cmd_train(args):
    """Train YOLO model using Ultralytics framework.

    The model is saved to runs/{task}/{name}/weights/best.pt (and last.pt).
    Early stopping is controlled by --patience (default 100 = disabled).

    Training metrics are logged in the run directory. Use TensorBoard for
    visualization: tensorboard --logdir runs/
    """
    from ultralytics import YOLO

    data_yaml = args.data
    task = args.task
    model_name = args.model

    # Auto-detect model if not specified
    if not model_name:
        model_name = TASK_MODELS.get(task, "yolo11n.pt")

    print(f"Training: task={task} model={model_name} data={data_yaml}")
    print(f"  epochs={args.epochs} batch={args.batch} imgsz={args.imgsz}")

    model = YOLO(model_name)
    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        lr0=args.lr0,
        name=args.name,
        patience=args.patience,
        pretrained=True,
    )

    # Find best weights path
    run_dir = f"runs/{TASK_SUFFIX.get(task, task)}/{args.name}"
    best_pt = os.path.join(run_dir, "weights", "best.pt")
    if os.path.exists(best_pt):
        print(f"\nBest weights: {best_pt}")
    else:
        print(f"\nTraining completed. Check: {run_dir}/weights/")


# ══════════════════════════════════════════════════════════════════════════════
#  Export ONNX
# ══════════════════════════════════════════════════════════════════════════════

def cmd_export(args):
    """Export trained model to ONNX format.

    ONNX file is saved alongside the .pt weights (same directory).
    Supports FP16, dynamic batch, and ONNX simplification.
    For OBB models, --nms flag embeds NMS into the ONNX graph.

    Output: best.onnx in the same directory as best.pt
    """
    from ultralytics import YOLO

    weights_path = args.weights
    if not os.path.exists(weights_path):
        print(f"Error: weights not found: {weights_path}")
        return

    print(f"Loading: {weights_path}")
    model = YOLO(weights_path)

    # Determine optimal export params
    # Try to get trained imgsz from model
    try:
        imgsz = args.imgsz if args.imgsz else 640
    except Exception:
        imgsz = 640

    export_kwargs = {
        "format": "onnx",
        "imgsz": imgsz,
        "half": args.half,
        "dynamic": args.dynamic,
        "simplify": not args.no_simplify,
        "opset": args.opset,
    }

    # OBB with NMS needs special handling
    if args.task == "obb" and args.nms:
        export_kwargs["nms"] = True

    print(f"Exporting ONNX (imgsz={imgsz}, half={args.half}, "
          f"dynamic={args.dynamic}, nms={args.nms})...")

    export_path = model.export(**export_kwargs)
    print(f"\nONNX exported: {export_path}")

    # Verify ONNX file is next to .pt
    onnx_path = weights_path.replace(".pt", ".onnx")
    if os.path.exists(onnx_path):
        pt_size = os.path.getsize(weights_path) / 1e6
        onnx_size = os.path.getsize(onnx_path) / 1e6
        print(f"  .pt:   {pt_size:.1f} MB")
        print(f"  .onnx: {onnx_size:.1f} MB")
        print(f"  Both in: {os.path.dirname(onnx_path)}/")
    else:
        print(f"  ... check: {os.path.dirname(weights_path)}/")


# ══════════════════════════════════════════════════════════════════════════════
#  Full pipeline
# ══════════════════════════════════════════════════════════════════════════════

def cmd_all(args):
    """Run full pipeline: label → data → train → export."""
    # Step 1: Label
    print("=" * 60)
    print("Step 1: Auto-labeling with SAM3")
    print("=" * 60)
    cmd_label(args)

    # Step 2: Generate data.yaml
    print("\n" + "=" * 60)
    print("Step 2: Generate dataset YAML")
    print("=" * 60)

    # Auto-detect dataset root from images path
    dataset_root = args.dataset_root
    if not dataset_root:
        # Assume images/labels are siblings: /data/images/val → /data is root
        img_dir = Path(args.images).resolve()
        dataset_root = str(img_dir.parents[1]) if "images" in str(img_dir) else str(img_dir.parent)

    class_names = args.names if args.names else ["object"]
    data_yaml = os.path.join(dataset_root, "data.yaml")
    cmd_data(argparse.Namespace(
        dataset_root=dataset_root, names=class_names, output=data_yaml
    ))

    # Step 3: Train
    print("\n" + "=" * 60)
    print("Step 3: Train YOLO model")
    print("=" * 60)
    cmd_train(argparse.Namespace(
        data=data_yaml, task=args.task, model=args.model,
        epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
        device=args.device, workers=args.workers, lr0=args.lr0,
        name=args.name, patience=args.patience,
    ))

    # Step 4: Export ONNX
    print("\n" + "=" * 60)
    print("Step 4: Export ONNX")
    print("=" * 60)
    task = args.task
    run_dir = f"runs/{TASK_SUFFIX.get(task, task)}/{args.name}"
    best_pt = os.path.join(run_dir, "weights", "best.pt")
    if not os.path.exists(best_pt):
        # Try last.pt
        best_pt = os.path.join(run_dir, "weights", "last.pt")

    cmd_export(argparse.Namespace(
        weights=best_pt, task=args.task, imgsz=args.imgsz,
        half=True, dynamic=args.dynamic, nms=args.nms,
        no_simplify=False, opset=12,
    ))

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="YOLO pipeline: SAM3 auto-label → train → ONNX export",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pre-compute text embeddings
  python pipeline.py precompute --prompt "bag on the conveyor belt"

  # Auto-label images (HBB format for detect training)
  python pipeline.py label --images /data/images/val --labels /data/labels/val \\
      --prompt "bag on the conveyor belt" --label-task hbb

  # Generate dataset YAML
  python pipeline.py data --dataset-root /data --names bag

  # Train YOLO detect
  python pipeline.py train --data /data/data.yaml --task detect --epochs 100

  # Train YOLO OBB
  python pipeline.py train --data /data/data.yaml --task obb --epochs 100

  # Export ONNX (saved alongside best.pt)
  python pipeline.py export --weights runs/detect/train/weights/best.pt

  # Full pipeline
  python pipeline.py all --images /data/images/val --labels /data/labels/val \\
      --dataset-root /data --task detect --epochs 50 \\
      --prompt "bag on the conveyor belt" --names bag
"""
    )

    sub = parser.add_subparsers(dest="command", help="Command")

    # ── precompute ──
    p_pre = sub.add_parser("precompute", help="Pre-compute SAM3 text embeddings")
    p_pre.add_argument("--prompt", required=True, help="Text prompt for detection")

    # ── label ──
    p_label = sub.add_parser("label", help="Auto-label images with SAM3")
    p_label.add_argument("--images", required=True, help="Directory of images")
    p_label.add_argument("--labels", required=True, help="Output directory for labels")
    p_label.add_argument("--prompt", required=True, help="Text prompt for SAM3 detection")
    p_label.add_argument("--emb", default=None, help="Path to pre-computed .npz embeddings")
    p_label.add_argument("--label-task", default="hbb", choices=["hbb", "obb"],
                          help="Label format: hbb=xywh center, obb=4 corners (default: hbb)")
    p_label.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p_label.add_argument("--iou", type=float, default=0.50, help="IoU threshold for NMS")
    p_label.add_argument("--imgsz", type=int, default=1008, help="SAM3 input size")
    p_label.add_argument("--force", action="store_true", help="Overwrite existing labels")

    # ── data ──
    p_data = sub.add_parser("data", help="Generate dataset YAML config")
    p_data.add_argument("--dataset-root", required=True, help="Dataset root directory")
    p_data.add_argument("--names", nargs="+", default=["object"], help="Class names")
    p_data.add_argument("--output", default=None, help="Output YAML path (default: data.yaml in dataset-root)")

    # ── train ──
    p_train = sub.add_parser("train", help="Train YOLO model")
    p_train.add_argument("--data", required=True, help="Dataset YAML path")
    p_train.add_argument("--task", default="detect", choices=["detect", "obb", "segment"],
                          help="Training task (default: detect)")
    p_train.add_argument("--model", default=None, help="Pretrained model (auto-detected from task)")
    p_train.add_argument("--epochs", type=int, default=100, help="Training epochs")
    p_train.add_argument("--batch", type=int, default=16, help="Batch size")
    p_train.add_argument("--imgsz", type=int, default=640, help="Input image size")
    p_train.add_argument("--device", default="cuda", help="Training device")
    p_train.add_argument("--workers", type=int, default=8, help="DataLoader workers")
    p_train.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    p_train.add_argument("--name", default="train", help="Run name")
    p_train.add_argument("--patience", type=int, default=100, help="Early stopping patience")

    # ── export ──
    p_export = sub.add_parser("export", help="Export model to ONNX")
    p_export.add_argument("--weights", required=True, help="Path to best.pt")
    p_export.add_argument("--task", default="detect", choices=["detect", "obb", "segment"],
                           help="Model task type")
    p_export.add_argument("--imgsz", type=int, default=None, help="Export image size")
    p_export.add_argument("--half", action="store_true", help="FP16 export")
    p_export.add_argument("--dynamic", action="store_true", default=True, help="Dynamic batch")
    p_export.add_argument("--no-simplify", action="store_true", help="Skip ONNX simplification")
    p_export.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    p_export.add_argument("--nms", action="store_true", help="Embed NMS (for OBB deployment)")

    # ── all ──
    p_all = sub.add_parser("all", help="Run full pipeline: label → train → export")
    p_all.add_argument("--images", required=True, help="Directory of images to label")
    p_all.add_argument("--labels", required=True, help="Output directory for labels")
    p_all.add_argument("--dataset-root", default="", help="Dataset root (auto-detected)")
    p_all.add_argument("--prompt", required=True, help="Text prompt for SAM3 detection")
    p_all.add_argument("--names", nargs="+", default=["object"], help="Class names")
    p_all.add_argument("--task", default="detect", choices=["detect", "obb", "segment"])
    p_all.add_argument("--model", default=None, help="Pretrained model")
    p_all.add_argument("--epochs", type=int, default=100)
    p_all.add_argument("--batch", type=int, default=16)
    p_all.add_argument("--imgsz", type=int, default=640)
    p_all.add_argument("--device", default="cuda")
    p_all.add_argument("--workers", type=int, default=8)
    p_all.add_argument("--lr0", type=float, default=0.01)
    p_all.add_argument("--name", default="train")
    p_all.add_argument("--patience", type=int, default=100)
    p_all.add_argument("--emb", default=None, help="Pre-computed embeddings .npz")
    p_all.add_argument("--label-task", default="hbb", choices=["hbb", "obb"])
    p_all.add_argument("--conf", type=float, default=0.25)
    p_all.add_argument("--iou", type=float, default=0.50)
    p_all.add_argument("--dynamic", action="store_true", default=True)
    p_all.add_argument("--nms", action="store_true")

    args = parser.parse_args()

    if args.command == "precompute":
        cmd_precompute(args)
    elif args.command == "label":
        cmd_label(args)
    elif args.command == "data":
        cmd_data(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "all":
        cmd_all(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
