#!/usr/bin/env python3
"""
SAM2.1 Base+ LoRA 推理可视化脚本
加载训练好的 LoRA checkpoint，在测试图片上推理并可视化结果。

用法:
    python infer_sam2_lora.py --checkpoint weights/sam2_lora_best.pt \
        --images /path/to/images --output results/infer

    # 带 GT mask 计算 IoU:
    python infer_sam2_lora.py --checkpoint weights/sam2_lora_best.pt \
        --images /path/to/images --masks /path/to/masks --output results/infer

    # 使用 train.txt 文件列表:
    python infer_sam2_lora.py --checkpoint weights/sam2_lora_best.pt \
        --images /path/to/train.txt --output results/infer
"""

import sys
import os
import argparse
import cv2
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# SAM2 路径
SAM2_ROOT = "/home/finn/ultralytics-main-count/sam2"
PROJECT_ROOT = "/home/finn/ultralytics-main-count"
os.chdir(SAM2_ROOT)
sys.path.insert(0, SAM2_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from hydra.core.global_hydra import GlobalHydra
GlobalHydra.instance().clear()

# 从 train_sam2_lora 导入模型构建函数
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ultralytics", "sam"))
from train_sam2_lora import LoRA_qkv, add_lora_to_sam2, load_sam2_direct


# ============================================================
# 模型加载
# ============================================================
def load_model(checkpoint_path, device="cuda"):
    """
    加载 SAM2 Base+ 模型，应用 LoRA，恢复 checkpoint 中的权重。
    与 train_sam2_lora.py 中的模型构建完全一致。
    """
    SAM2_CKPT = os.path.join(SAM2_ROOT, "checkpoints/sam2.1_hiera_base_plus.pt")

    # 1. 构建基础模型
    print("构建 SAM2 Base+ 模型...")
    model = load_sam2_direct(SAM2_CKPT, device)

    # 2. 添加 LoRA (rank=8)
    print("添加 LoRA (r=8)...")
    model, w_As, w_Bs = add_lora_to_sam2(model, r=8, device=device)

    # 3. 创建 seg_head
    seg_head = nn.Conv2d(256, 1, kernel_size=1).to(device)

    # 4. 加载 checkpoint
    print(f"加载 checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # 恢复 LoRA 权重
    for i, (wa_sd, wb_sd) in enumerate(zip(ckpt['w_As'], ckpt['w_Bs'])):
        w_As[i].load_state_dict(wa_sd)
        w_Bs[i].load_state_dict(wb_sd)

    # 恢复 seg_head 权重
    if 'seg_head' in ckpt:
        seg_head.load_state_dict(ckpt['seg_head'])
        print(f"  已加载 seg_head 权重")
    else:
        print(f"  [警告] checkpoint 中未找到 seg_head，使用随机初始化")

    epoch = ckpt.get('epoch', '?')
    loss = ckpt.get('loss', '?')
    print(f"  checkpoint epoch={epoch}, loss={loss}")

    model.eval()
    seg_head.eval()
    return model, seg_head


# ============================================================
# 推理
# ============================================================
def predict_mask(model, seg_head, image_rgb, device="cuda", threshold=0.5):
    """
    推理单张图片，返回原始尺寸的二值 mask。
    流程: resize→1024 → image_encoder → seg_head → sigmoid → resize→原尺寸
    """
    h_orig, w_orig = image_rgb.shape[:2]

    # Resize to 1024x1024，归一化
    img_resized = cv2.resize(image_rgb, (1024, 1024))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().to(device) / 255.0
    img_tensor = img_tensor.unsqueeze(0)  # [1, 3, 1024, 1024]

    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            backbone_out = model.image_encoder(img_tensor)
            feat = backbone_out["vision_features"]  # [1, 256, H', W']
            feat_up = nn.functional.interpolate(
                feat.float(), size=(1024, 1024), mode='bilinear', align_corners=False
            )
            seg_logits = seg_head(feat_up)  # [1, 1, 1024, 1024]
            prob = torch.sigmoid(seg_logits.squeeze(1))  # [1, 1024, 1024]

    # 二值化
    mask_1024 = (prob.squeeze(0) > threshold).cpu().numpy().astype(np.uint8)

    # Resize 回原尺寸
    mask_orig = cv2.resize(mask_1024, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    return mask_orig


# ============================================================
# 可视化（与 infer_v2.py 风格一致）
# ============================================================
def compute_iou(pred_mask, gt_mask):
    """计算两个二值 mask 的 IoU"""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def plot_results(image, mask_binary, iou=0.0):
    """
    可视化分割结果，风格与 infer_v2.py 一致：
    - 彩色 mask overlay (alpha=0.4)
    - 轮廓线 (thickness=2)
    - 边界框 (thickness=2)
    - 标签文字 + 背景框
    """
    annotated = image.copy()
    color = (0, 255, 0)  # 绿色（SAM 单类别）

    # 计算像素面积
    pixel_area = int(np.sum(mask_binary))
    if pixel_area == 0:
        return annotated

    # 从 mask 轮廓点计算边界框
    contours_pts = np.argwhere(mask_binary > 0)
    y_min, x_min = contours_pts.min(axis=0)
    y_max, x_max = contours_pts.max(axis=0)
    x1, y1, x2, y2 = int(x_min), int(y_min), int(x_max), int(y_max)
    box_w = x2 - x1
    box_h = y2 - y1
    box_area = box_w * box_h

    # 彩色 mask overlay (alpha=0.4)
    colored_mask = np.zeros_like(image)
    colored_mask[mask_binary == 1] = color
    alpha = 0.4
    mask_region = mask_binary == 1
    annotated[mask_region] = (
        annotated[mask_region] * (1 - alpha) + colored_mask[mask_region] * alpha
    ).astype(np.uint8)

    # 轮廓线
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(annotated, contours, -1, color, 2)

    # 边界框
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

    # 标签文字 + 背景框
    label = f"area:{pixel_area}px box:{box_w}x{box_h}({box_area}px) iou:{iou:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(annotated, (x1, y1 - th - 12), (x1 + tw + 8, y1), color, -1)
    cv2.putText(annotated, label, (x1 + 4, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return annotated


# ============================================================
# 图片列表加载
# ============================================================
def load_image_paths(images_arg, data_root=None):
    """
    从 --images 参数加载图片路径列表。
    支持两种模式:
      1. 目录路径 → 扫描图片文件
      2. .txt 文件 → 读取图片名称，在 data_root/images/ 下查找
    """
    images_path = Path(images_arg)

    if images_path.is_file() and images_path.suffix == '.txt':
        # 模式 2: 文本文件列表
        if data_root is None:
            data_root = images_path.parent
        img_dir = Path(data_root) / "images"
        with open(images_path, 'r') as f:
            names = [line.strip() for line in f if line.strip()]

        paths = []
        for name in names:
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                p = img_dir / f"{name}{ext}"
                if p.exists():
                    paths.append(p)
                    break
            else:
                print(f"  [跳过] 未找到图片: {name}")
        return sorted(paths)

    elif images_path.is_dir():
        # 模式 1: 目录扫描
        exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        return sorted([f for f in images_path.iterdir()
                       if f.suffix.lower() in exts])

    else:
        print(f"[错误] --images 路径无效: {images_arg}")
        return []


def find_gt_mask(img_path, masks_dir):
    """查找与图片对应的 GT mask 文件"""
    if masks_dir is None:
        return None
    masks_dir = Path(masks_dir)
    stem = img_path.stem
    for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
        mask_path = masks_dir / f"{stem}{ext}"
        if mask_path.exists():
            return mask_path
    return None


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="SAM2.1 Base+ LoRA 推理可视化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="LoRA checkpoint 路径 (sam2_lora_best.pt)")
    parser.add_argument("--images", type=str, required=True,
                        help="测试图片目录，或包含图片名列表的 .txt 文件")
    parser.add_argument("--output", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--masks", type=str, default=None,
                        help="GT mask 目录 (可选，用于计算 IoU)")
    parser.add_argument("--data-root", type=str, default=None,
                        help="数据集根目录 (当 --images 为 .txt 时用于定位 images/ 目录)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="推理设备 (默认: cuda)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="mask 二值化阈值 (默认: 0.5)")
    args = parser.parse_args()

    print("=" * 60)
    print("SAM2.1 Base+ LoRA 推理可视化")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Images:     {args.images}")
    print(f"Output:     {args.output}")
    if args.masks:
        print(f"Masks:      {args.masks}")
    print()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载模型
    model, seg_head = load_model(args.checkpoint, device)
    print()

    # 加载图片列表
    image_paths = load_image_paths(args.images, args.data_root)
    if not image_paths:
        print("[错误] 未找到任何图片")
        return
    print(f"找到 {len(image_paths)} 张图片")

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 推理循环
    print()
    total_masks = 0
    total_iou = 0.0
    iou_count = 0

    for idx, img_path in enumerate(image_paths):
        # 读取图片
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"  [{idx+1}/{len(image_paths)}] [跳过] 无法读取: {img_path.name}")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = image_rgb.shape[:2]

        # 推理
        pred_mask = predict_mask(model, seg_head, image_rgb, device, args.threshold)

        # 计算 IoU (如果有 GT mask)
        iou = 0.0
        gt_mask_path = find_gt_mask(img_path, args.masks)
        if gt_mask_path is not None:
            gt_mask_img = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
            if gt_mask_img is not None:
                gt_mask_binary = (gt_mask_img > 127).astype(np.uint8)
                # 确保尺寸一致
                if gt_mask_binary.shape != pred_mask.shape:
                    gt_mask_binary = cv2.resize(
                        gt_mask_binary, (pred_mask.shape[1], pred_mask.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )
                iou = compute_iou(pred_mask, gt_mask_binary)
                total_iou += iou
                iou_count += 1

        # 可视化
        annotated = plot_results(image_rgb, pred_mask, iou)

        # 保存
        save_path = output_dir / f"{img_path.stem}.jpg"
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(save_path), annotated_bgr)

        pixel_area = int(np.sum(pred_mask))
        total_masks += 1 if pixel_area > 0 else 0

        iou_str = f"iou={iou:.3f}" if gt_mask_path else "iou=N/A"
        print(f"  [{idx+1}/{len(image_paths)}] {img_path.name}  "
              f"area={pixel_area}px {iou_str}")

    # 汇总
    print()
    print("-" * 60)
    print(f"处理完成!")
    print(f"  总图片: {len(image_paths)}")
    print(f"  有 mask: {total_masks}")
    if iou_count > 0:
        print(f"  平均 IoU: {total_iou / iou_count:.4f} ({iou_count} 张有 GT)")
    print(f"  输出目录: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
