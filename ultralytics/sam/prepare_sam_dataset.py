#!/usr/bin/env python3
"""
从 YOLO-seg 标注生成 SAM 训练数据集

输入: xutong_military_yolo_seg_v3/{images,labels}/{train,val}
输出: sam_dataset/{images,masks,labels,points}

YOLO-seg 标注格式: class_id x1 y1 x2 y2 ... xn yn (归一化多边形坐标)
"""

import json
import os
import shutil
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path


def parse_yolo_seg_label(label_path, img_width, img_height):
    """
    解析 YOLO-seg 标注文件
    返回: [(class_id, polygon_points_px), ...]
    """
    annotations = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # 至少 class_id + 3个点(6个坐标)
                continue
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            # 归一化坐标转像素坐标
            points = []
            for i in range(0, len(coords), 2):
                px = coords[i] * img_width
                py = coords[i + 1] * img_height
                points.append((px, py))
            annotations.append((class_id, points))
    return annotations


def create_mask_from_polygons(annotations, img_width, img_height):
    """
    从多边形标注生成二值 mask
    所有缺陷合并为一张 mask (255=缺陷, 0=背景)
    """
    mask = Image.new('L', (img_width, img_height), 0)
    draw = ImageDraw.Draw(mask)
    for class_id, points in annotations:
        polygon = [(int(p[0]), int(p[1])) for p in points]
        draw.polygon(polygon, fill=255)
    return mask


def compute_bbox_from_polygon(points, img_width, img_height):
    """
    从多边形点集计算最小外接矩形 (axis-aligned bbox)
    返回归一化格式: cx cy w h
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # 转为归一化中心点+宽高
    cx = (x_min + x_max) / 2.0 / img_width
    cy = (y_min + y_max) / 2.0 / img_height
    w = (x_max - x_min) / img_width
    h = (y_max - y_min) / img_height

    # 确保在 [0, 1] 范围内
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    return cx, cy, w, h


def sample_points(mask_array, img_width, img_height, num_fg=6, num_bg=4):
    """
    从 mask 区域采样前景点和背景点
    前景点: mask=255 区域内随机采样
    背景点: mask=0 区域内随机采样

    返回: [(label, x_normalized, y_normalized), ...]
    """
    fg_coords = np.argwhere(mask_array == 255)
    bg_coords = np.argwhere(mask_array == 0)

    points = []

    # 前景点 (label=1)
    if len(fg_coords) > 0:
        num_fg_actual = min(num_fg, len(fg_coords))
        fg_idx = np.random.choice(len(fg_coords), num_fg_actual, replace=False)
        for idx in fg_idx:
            y, x = fg_coords[idx]
            points.append((1, x / img_width, y / img_height))

    # 背景点 (label=0)
    if len(bg_coords) > 0:
        num_bg_actual = min(num_bg, len(bg_coords))
        bg_idx = np.random.choice(len(bg_coords), num_bg_actual, replace=False)
        for idx in bg_idx:
            y, x = bg_coords[idx]
            points.append((0, x / img_width, y / img_height))

    return points


def process_split(source_dir, output_dir, split_name):
    """
    处理一个数据集分割 (train 或 val)
    """
    img_dir = Path(source_dir) / "images" / split_name
    lbl_dir = Path(source_dir) / "labels" / split_name

    out_img_dir = Path(output_dir) / "images"
    out_mask_dir = Path(output_dir) / "masks"
    out_lbl_dir = Path(output_dir) / "labels"
    out_pts_dir = Path(output_dir) / "points"

    # 收集所有类别
    class_map = {}
    file_names = []

    # 获取有标注的文件
    label_files = sorted(lbl_dir.glob("*.txt"))
    print(f"\n处理 {split_name} 集:")
    print(f"  标注文件: {len(label_files)} 个")

    success = 0
    skip = 0

    for lbl_path in label_files:
        stem = lbl_path.stem

        # 查找对应图片
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            p = img_dir / f"{stem}{ext}"
            if p.exists():
                img_path = p
                break

        if img_path is None:
            print(f"  [跳过] {stem} - 找不到对应图片")
            skip += 1
            continue

        # 读取图片尺寸
        with Image.open(img_path) as img:
            img_width, img_height = img.size

        # 解析标注
        annotations = parse_yolo_seg_label(str(lbl_path), img_width, img_height)
        if not annotations:
            print(f"  [跳过] {stem} - 标注为空")
            skip += 1
            continue

        # 收集类别
        for class_id, _ in annotations:
            if class_id not in class_map:
                class_map[class_id] = class_id

        # 1. 复制原图
        shutil.copy2(str(img_path), str(out_img_dir / f"{stem}.png"))

        # 2. 生成二值 mask
        mask = create_mask_from_polygons(annotations, img_width, img_height)
        mask.save(str(out_mask_dir / f"{stem}.png"), "PNG")

        # 3. 生成 bbox 标注 (class_id cx cy w h)
        with open(out_lbl_dir / f"{stem}.txt", 'w') as f:
            for class_id, points in annotations:
                cx, cy, w, h = compute_bbox_from_polygon(points, img_width, img_height)
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        # 4. 生成点提示
        mask_array = np.array(mask)
        points = sample_points(mask_array, img_width, img_height, num_fg=6, num_bg=4)
        with open(out_pts_dir / f"{stem}.txt", 'w') as f:
            for label, x, y in points:
                f.write(f"{label} {x:.6f} {y:.6f}\n")

        file_names.append(stem)
        success += 1

    print(f"  成功: {success}, 跳过: {skip}")
    return file_names, class_map


def main():
    # 路径配置
    source_dir = Path("/mnt/c/Users/finn.zhang/project/Datasets/xutong_military_yolo_seg_v3")
    output_dir = Path("/mnt/c/Users/finn.zhang/project/Datasets/sam_dataset")

    # 创建输出目录
    for sub in ["images", "masks", "labels", "points"]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("YOLO-seg → SAM 训练数据集转换")
    print("=" * 60)
    print(f"源目录: {source_dir}")
    print(f"输出目录: {output_dir}")

    # 处理 train 和 val
    all_files = []
    all_class_map = {}

    for split in ["train", "val"]:
        files, class_map = process_split(source_dir, output_dir, split)
        all_files.extend(files)
        all_class_map.update(class_map)

    # 生成 train.txt
    train_txt = output_dir / "train.txt"
    with open(train_txt, 'w') as f:
        for name in sorted(set(all_files)):
            f.write(f"{name}\n")

    # 生成 class_map.json
    class_map_json = output_dir / "class_map.json"
    with open(class_map_json, 'w') as f:
        json.dump(all_class_map, f, indent=2, ensure_ascii=False)

    # 统计
    img_count = len(list((output_dir / "images").glob("*.png")))
    mask_count = len(list((output_dir / "masks").glob("*.png")))
    lbl_count = len(list((output_dir / "labels").glob("*.txt")))
    pts_count = len(list((output_dir / "points").glob("*.txt")))

    print(f"\n{'=' * 60}")
    print(f"转换完成!")
    print(f"{'=' * 60}")
    print(f"  图片: {img_count} 张")
    print(f"  掩码: {mask_count} 张")
    print(f"  bbox 标注: {lbl_count} 个")
    print(f"  点提示: {pts_count} 个")
    print(f"  train.txt: {len(set(all_files))} 行")
    print(f"  类别映射: {all_class_map}")
    print(f"\n输出目录: {output_dir}")


if __name__ == "__main__":
    main()
