#!/usr/bin/env python3
"""
yolo11l-seg v2 推理可视化
对 img_org 和 img 目录中的图片进行推理
"""
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def plot_results(image, result, class_names):
    annotated = image.copy()
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255)]

    if result.masks is None:
        return annotated

    masks = result.masks.data.cpu().numpy()
    boxes = result.boxes

    for mask, box in zip(masks, boxes):
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = class_names.get(cls_id, str(cls_id))
        color = colors[cls_id % len(colors)]

        mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
        mask_binary = (mask_resized > 0.5).astype(np.uint8)

        # 计算像素面积
        pixel_area = int(np.sum(mask_binary))

        # 从 mask 轮廓点计算边界框（最小xy → 最大xy）
        contours_pts = np.argwhere(mask_binary > 0)
        if len(contours_pts) == 0:
            continue
        y_min, x_min = contours_pts.min(axis=0)
        y_max, x_max = contours_pts.max(axis=0)
        x1, y1, x2, y2 = int(x_min), int(y_min), int(x_max), int(y_max)
        box_w = x2 - x1
        box_h = y2 - y1
        box_area = box_w * box_h

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

        # 标签：类别 + 缺陷面积 + 框宽高 + 框面积
        label = f"cls:{cls_id} area:{pixel_area}px box:{box_w:.4f}x{box_h:.4f}({box_area:.4f}px)"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 12), (x1 + tw + 8, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 4, y1 - 6),
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
        annotated = plot_results(image, result, class_names)
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

    output_base = Path(PROJECT_DIR) / "results" / "v2_inference"

    print("=" * 60)
    print("yolo11l-seg v2 推理可视化")
    print("=" * 60)
    print(f"模型: {model_path}")
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
