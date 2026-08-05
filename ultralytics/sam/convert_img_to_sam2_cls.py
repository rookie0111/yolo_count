#!/usr/bin/env python3
"""
SAM 2.1 训练数据准备脚本（带类别信息）
将 img 文件夹中的 LabelMe 标注转换为 SAM 2.1 训练格式
每个掩码文件名包含类别ID，同时生成类别映射文件
"""

import json
import os
import shutil
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path


def convert_labelme_to_sam2_with_classes(source_dir, output_dir, target_size=1024):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    img_folder = output_dir / "images"
    mask_folder = output_dir / "masks"
    img_folder.mkdir(parents=True, exist_ok=True)
    mask_folder.mkdir(parents=True, exist_ok=True)

    json_files = sorted([f for f in source_dir.glob("*.json")
                         if not f.name.startswith('convert_') and not f.name.startswith('json_')])
    print(f"源目录: {source_dir}")
    print(f"输出目录: {output_dir}")
    print(f"找到 {len(json_files)} 个标注文件")

    # 收集所有类别
    class_map = {}  # label_name -> class_id
    for json_path in json_files:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for shape in data.get('shapes', []):
            label = shape.get('label', '0')
            if label not in class_map:
                class_map[label] = len(class_map)

    # 保存类别映射
    class_map_path = output_dir / "class_map.json"
    with open(class_map_path, 'w', encoding='utf-8') as f:
        json.dump(class_map, f, indent=2, ensure_ascii=False)
    print(f"类别映射: {class_map}")
    print(f"类别映射保存: {class_map_path}")

    success_count = 0
    file_names = []
    class_labels = {}  # filename -> list of class_ids

    for json_path in json_files:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            img_name = data.get('imagePath', json_path.stem + '.png')
            img_path = source_dir / img_name
            if not img_path.exists():
                continue

            img_width = data.get('imageWidth')
            img_height = data.get('imageHeight')

            with Image.open(img_path) as img:
                orig_width, orig_height = img.size
                if img_height is None or img_width is None:
                    img_width, img_height = orig_width, orig_height

                img_resized = img.resize((target_size, target_size), Image.LANCZOS)

                # 为每个类别创建独立的掩码
                shapes = data.get('shapes', [])
                file_classes = []

                for shape in shapes:
                    label = shape.get('label', '0')
                    points = shape.get('points', [])
                    shape_type = shape.get('shape_type', 'polygon')
                    if not points:
                        continue

                    class_id = class_map.get(label, 0)
                    file_classes.append(class_id)

                    mask = Image.new('L', (orig_width, orig_height), 0)
                    draw = ImageDraw.Draw(mask)

                    if shape_type == 'polygon':
                        polygon = [(int(p[0]), int(p[1])) for p in points]
                        draw.polygon(polygon, fill=255)
                    elif shape_type == 'rectangle':
                        p1 = (int(points[0][0]), int(points[0][1]))
                        p2 = (int(points[1][0]), int(points[1][1]))
                        draw.rectangle([p1, p2], fill=255)
                    elif shape_type == 'circle':
                        cx, cy = int(points[0][0]), int(points[0][1])
                        if len(points) > 1:
                            r = int(((points[1][0] - points[0][0])**2 +
                                    (points[1][1] - points[0][1])**2) ** 0.5)
                            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=255)
                    else:
                        polygon = [(int(p[0]), int(p[1])) for p in points]
                        draw.polygon(polygon, fill=255)

                    mask_resized = mask.resize((target_size, target_size), Image.NEAREST)

                    # 保存掩码：文件名包含类别ID
                    stem = json_path.stem
                    mask_filename = f"{stem}_cls{class_id}.png"
                    mask_resized.save(str(mask_folder / mask_filename), 'PNG')

                # 保存图片（只保存一次）
                stem = json_path.stem
                img_resized_rgb = img_resized.convert('RGB')
                img_resized_rgb.save(str(img_folder / f"{stem}.jpg"), 'JPEG', quality=95)

                file_names.append(stem)
                class_labels[stem] = list(set(file_classes))
                success_count += 1
                print(f"  [OK] {json_path.name} -> 类别: {list(set(file_classes))}")

        except Exception as e:
            print(f"  [错误] {json_path.name}: {e}")

    # 保存训练文件列表（每行格式：文件名 类别ID1,类别ID2,...）
    file_list_path = output_dir / "train.txt"
    with open(file_list_path, 'w', encoding='utf-8') as f:
        for name in file_names:
            classes = class_labels.get(name, [0])
            class_str = ','.join(map(str, classes))
            f.write(f"{name} {class_str}\n")

    # 保存类别标签文件
    labels_path = output_dir / "labels.txt"
    with open(labels_path, 'w', encoding='utf-8') as f:
        for label, cid in sorted(class_map.items(), key=lambda x: x[1]):
            f.write(f"{cid}: {label}\n")

    print(f"\n转换完成: {success_count} 个样本")
    print(f"类别数: {len(class_map)}")
    print(f"类别映射: {class_map}")
    return output_dir, class_map


def create_sam2_config(output_dir, class_map):
    output_dir = Path(output_dir)
    config_path = output_dir / "defect_finetune_cls.yaml"

    num_classes = len(class_map)

    config_content = f"""# SAM 2.1 带类别信息的训练配置
dataset:
  img_folder: {output_dir / "images"}
  gt_folder: {output_dir / "masks"}
  file_list_txt: {output_dir / "train.txt"}
  class_map: {output_dir / "class_map.json"}
  multiplier: 2

model:
  num_classes: {num_classes}
  use_class_head: true

scratch:
  resolution: 1024
  train_batch_size: 1
  num_train_workers: 4
  num_frames: 1
  max_num_objects: 10
  base_lr: 5.0e-6
  vision_lr: 3.0e-06
  phases_per_epoch: 1
  num_epochs: 40
"""

    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)

    print(f"配置文件: {config_path}")
    return config_path


def main():
    source_dir = Path("/mnt/c/Users/finn.zhang/project/Datasets/xutong_military/img")
    output_dir = Path("/mnt/c/Users/finn.zhang/project/Datasets/xutong_military_sam2_cls")

    print("=" * 60)
    print("SAM 2.1 训练数据准备（带类别信息）")
    print("=" * 60)

    result_dir, class_map = convert_labelme_to_sam2_with_classes(source_dir, output_dir, target_size=1024)
    config_path = create_sam2_config(result_dir, class_map)

    print(f"\n输出目录: {result_dir}")
    print(f"  ├── images/         (图片)")
    print(f"  ├── masks/          (掩码，文件名含类别ID)")
    print(f"  ├── train.txt       (文件列表+类别)")
    print(f"  ├── class_map.json  (类别映射)")
    print(f"  ├── labels.txt      (类别标签)")
    print(f"  └── defect_finetune_cls.yaml (配置)")


if __name__ == "__main__":
    main()
