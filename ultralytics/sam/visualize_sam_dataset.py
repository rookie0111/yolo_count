#!/usr/bin/env python3
"""
SAM 数据集可视化
展示：原图 + mask叠加 + bbox + 前景点/背景点
"""

import cv2
import numpy as np
from pathlib import Path


def visualize_sample(img_path, mask_path, label_path, point_path, output_path):
    """可视化单个样本"""
    # 读取原图
    image = cv2.imread(str(img_path))
    if image is None:
        return False
    h, w = image.shape[:2]

    # 读取 mask
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False

    # 创建可视化图
    vis = image.copy()

    # 1. 叠加半透明 mask（绿色）
    mask_colored = np.zeros_like(vis)
    mask_colored[mask == 255] = [0, 255, 0]  # 绿色
    alpha = 0.35
    mask_region = mask == 255
    vis[mask_region] = (
        vis[mask_region] * (1 - alpha) + mask_colored[mask_region] * alpha
    ).astype(np.uint8)

    # 2. 绘制 mask 轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)

    # 3. 绘制 bbox（红色）
    if label_path and Path(label_path).exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # 标签
                label = f"cls:{cls_id}"
                cv2.putText(vis, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 4. 绘制点提示
    if point_path and Path(point_path).exists():
        with open(point_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                label = int(parts[0])
                px = int(float(parts[1]) * w)
                py = int(float(parts[2]) * h)
                # 前景点=绿色圆，背景点=红色叉
                if label == 1:
                    cv2.circle(vis, (px, py), 6, (0, 255, 0), -1)
                    cv2.circle(vis, (px, py), 6, (255, 255, 255), 1)
                else:
                    cv2.drawMarker(vis, (px, py), (0, 0, 255),
                                   cv2.MARKER_TILTED_CROSS, 12, 2)

    # 5. 图例
    y0 = 30
    cv2.putText(vis, "Green overlay = mask", (10, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(vis, "Red box = bbox", (10, y0 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(vis, "Green dot = fg point", (10, y0 + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(vis, "Red cross = bg point", (10, y0 + 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imwrite(str(output_path), vis)
    return True


def main():
    dataset_dir = Path("/mnt/c/Users/finn.zhang/project/Datasets/sam_dataset")
    output_dir = Path("/mnt/c/Users/finn.zhang/project/Datasets/sam_dataset/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    img_dir = dataset_dir / "images"
    mask_dir = dataset_dir / "masks"
    lbl_dir = dataset_dir / "labels"
    pts_dir = dataset_dir / "points"

    images = sorted(img_dir.glob("*.png"))
    print(f"=" * 60)
    print(f"SAM 数据集可视化")
    print(f"=" * 60)
    print(f"图片数: {len(images)}")
    print(f"输出: {output_dir}")
    print()

    success = 0
    for img_path in images:
        stem = img_path.stem
        mask_path = mask_dir / f"{stem}.png"
        label_path = lbl_dir / f"{stem}.txt"
        point_path = pts_dir / f"{stem}.txt"
        out_path = output_dir / f"{stem}.jpg"

        if not mask_path.exists():
            print(f"  [跳过] {stem} - 无 mask")
            continue

        ok = visualize_sample(img_path, mask_path, label_path, point_path, out_path)
        if ok:
            success += 1
            print(f"  [OK] {stem}")
        else:
            print(f"  [错误] {stem}")

    print(f"\n完成: {success}/{len(images)} 张可视化结果")
    print(f"保存在: {output_dir}")


if __name__ == "__main__":
    main()
