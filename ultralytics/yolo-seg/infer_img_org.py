#!/usr/bin/env python3
"""
对 img_org 目录中的图片进行推理可视化
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

        colored_mask = np.zeros_like(image)
        colored_mask[mask_binary == 1] = color
        alpha = 0.4
        mask_region = mask_binary == 1
        annotated[mask_region] = (
            annotated[mask_region] * (1 - alpha) + colored_mask[mask_region] * alpha
        ).astype(np.uint8)

        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, contours, -1, color, 2)

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 5, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return annotated


def main():
    PROJECT_DIR = "/home/finn/ultralytics-main-count/ultralytics/yolo-seg"
    IMG_DIR = "/mnt/c/Users/finn.zhang/project/Datasets/xutong_military/img_org"

    weights_dir = Path(PROJECT_DIR) / "weights" / "yolo11m_seg_military"
    model_path = weights_dir / "weights" / "best.pt"
    if not model_path.exists():
        model_path = weights_dir / "weights" / "last.pt"

    output_dir = Path(PROJECT_DIR) / "results" / "img_org_inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("img_org 推理可视化")
    print("=" * 60)
    print(f"模型: {model_path}")
    print(f"输入: {IMG_DIR}")
    print(f"输出: {output_dir}")
    print()

    model = YOLO(str(model_path))
    class_names = model.names
    print(f"类别: {class_names}")

    img_dir = Path(IMG_DIR)
    images = sorted([f for f in img_dir.glob("*") if f.suffix.lower() in ['.bmp', '.jpg', '.jpeg', '.png']])
    print(f"找到 {len(images)} 张图片")
    print()

    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[跳过] 无法读取: {img_path.name}")
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

        # 统计检测数
        n_det = len(result.boxes) if result.boxes is not None else 0

        save_path = output_dir / f"{img_path.stem}.jpg"
        cv2.imwrite(str(save_path), annotated)
        print(f"[OK] {img_path.name} -> {save_path.name}  ({n_det} 个检测)")

    print()
    print("=" * 60)
    print(f"推理完成! 结果保存在: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
