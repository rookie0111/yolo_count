#!/usr/bin/env python3
"""
yolo11l-seg 物理坐标版本推理可视化
像素坐标 → 物理坐标转换（基于相机内参）

相机内参:
  fx = 208.333333
  fy = 208.333333
  cx = 2736
  cy = 1824

转换公式:
  物理X = (pixel_x - cx) / fx
  物理Y = (pixel_y - cy) / fy
  物理宽度 = pixel_width / fx
  物理高度 = pixel_height / fy
  物理面积 = pixel_area / (fx * fy)
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# 相机内参
FX = 416.666667
FY = 416.666667
CX = 2736
CY = 1824
# fx = 416.666667
# fy = 416.666667
# cx = 2736
# cy = 1824

def pixel_to_physical(px, py):
    """像素坐标转物理坐标(mm)"""
    phys_x = (px - CX) / FX
    phys_y = (py - CY) / FY
    return phys_x, phys_y


def plot_results_physical(image, result, class_names):
    annotated = image.copy()
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255)]

    if result.masks is None:
        return annotated

    masks = result.masks.data.cpu().numpy()

    for mask, box in zip(masks, result.boxes):
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        color = colors[cls_id % len(colors)]

        mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
        mask_binary = (mask_resized > 0.5).astype(np.uint8)

        # 像素面积
        pixel_area = int(np.sum(mask_binary))

        # 从 mask 轮廓点计算边界框
        contours_pts = np.argwhere(mask_binary > 0)
        if len(contours_pts) == 0:
            continue
        y_min, x_min = contours_pts.min(axis=0)
        y_max, x_max = contours_pts.max(axis=0)
        x1, y1, x2, y2 = int(x_min), int(y_min), int(x_max), int(y_max)
        box_w_px = x2 - x1
        box_h_px = y2 - y1

        # 物理坐标转换
        phys_x1, phys_y1 = pixel_to_physical(x1, y1)
        phys_x2, phys_y2 = pixel_to_physical(x2, y2)
        phys_w = abs(phys_x2 - phys_x1)   # mm (边界框物理宽度)
        phys_h = abs(phys_y2 - phys_y1)    # mm (边界框物理高度)
        phys_box_area = phys_w * phys_h    # mm² (边界框物理面积 = 宽 × 高)
        phys_area = box_w_px * box_h_px / (FX * FY)  # mm² (掩码物理面积，用像素宽高转换)

        # 绘制掩码
        colored_mask = np.zeros_like(image)
        colored_mask[mask_binary == 1] = color
        alpha = 0.4
        mask_region = mask_binary == 1
        annotated[mask_region] = (
            annotated[mask_region] * (1 - alpha) + colored_mask[mask_region] * alpha
        ).astype(np.uint8)

        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, contours, -1, color, 2)

        # 绘制检测框
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # 标签：物理坐标
        label1 = f"cls:{cls_id} area:{phys_area:.4f}mm2"
        label2 = f"box:{phys_w:.4f}x{phys_h:.4f}mm({phys_box_area:.4f}mm2)"

        (tw1, th1), _ = cv2.getTextSize(label1, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        (tw2, th2), _ = cv2.getTextSize(label2, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        max_tw = max(tw1, tw2)
        total_h = th1 + th2 + 18

        # 绘制标签背景
        cv2.rectangle(annotated, (x1, y1 - total_h - 4), (x1 + max_tw + 8, y1), color, -1)

        # 绘制标签文字
        cv2.putText(annotated, label1, (x1 + 4, y1 - th2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, label2, (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return annotated


def run_inference(model, img_dir, output_dir, prefix=""):
    class_names = model.names
    img_dir = Path(img_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([f for f in img_dir.glob("*")
                     if f.suffix.lower() in ['.bmp', '.jpg', '.jpeg', '.png']])
    print(f"  找到 {len(images)} 张图片")

    total_det = 0
    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        results = model.predict(
            source=image,
            imgsz=640,
            conf=0.75,
            iou=0.5,
            agnostic_nms=True,
            device=0,
            verbose=False,
        )

        result = results[0]
        annotated = plot_results_physical(image, result, class_names)
        n_det = len(result.boxes) if result.boxes is not None else 0
        total_det += n_det

        save_path = output_dir / f"{prefix}{img_path.stem}.jpg"
        cv2.imwrite(str(save_path), annotated)
        print(f"    [{n_det}] {img_path.name}")

    print(f"  总检测数: {total_det}")
    return total_det


def main():
    PROJECT_DIR = "/home/finn/ultralytics-main-count/ultralytics/yolo-seg"
    model_path = Path(PROJECT_DIR) / "weights" / "yolo11l_seg_v3_final" / "weights" / "best.pt"

    if not model_path.exists():
        print(f"[错误] 未找到模型: {model_path}")
        return

    output_base = Path(PROJECT_DIR) / "results" / "physical_inference"

    print("=" * 60)
    print("yolo11l-seg 物理坐标推理可视化")
    print("=" * 60)
    print(f"模型: {model_path}")
    print(f"相机内参: fx={FX} fy={FY} cx={CX} cy={CY}")
    print()

    model = YOLO(str(model_path))
    print(f"类别: {model.names}")
    print()

    # 推理 img_org
    print("[1/3] 推理 img_org 目录...")
    run_inference(
        model,
        "/mnt/c/Users/finn.zhang/project/Datasets/xutong_military/img_org",
        output_base / "img_org",
        prefix="org_"
    )

    print()

    # 推理训练集
    print("[2/3] 推理训练集...")
    train_dir = "/mnt/c/Users/finn.zhang/project/Datasets/xutong_military_yolo_seg_v3/images/train"
    run_inference(
        model,
        train_dir,
        output_base / "train",
        prefix="train_"
    )

    print()

    # 推理 v3 验证集
    print("[3/3] 推理 v3 验证集...")
    val_dir = "/mnt/c/Users/finn.zhang/project/Datasets/xutong_military_yolo_seg_v3/images/val"
    run_inference(
        model,
        val_dir,
        output_base / "val_v3",
        prefix="val_"
    )

    print()
    print("=" * 60)
    print(f"推理完成! 结果保存在: {output_base}")
    print("=" * 60)


if __name__ == "__main__":
    main()
