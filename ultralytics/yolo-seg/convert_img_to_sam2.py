#!/usr/bin/env python3
"""
SAM 2.1 训练数据准备脚本（WSL 版本）
将 img 文件夹中的 LabelMe 标注转换为 SAM 2.1 训练格式
"""

import json
import os
import shutil
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path


def convert_labelme_to_sam2_format(source_dir, output_dir, target_size=1024):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    img_folder = output_dir / "images"
    gt_folder = output_dir / "masks"
    img_folder.mkdir(parents=True, exist_ok=True)
    gt_folder.mkdir(parents=True, exist_ok=True)

    json_files = sorted([f for f in source_dir.glob("*.json")
                         if not f.name.startswith('convert_') and not f.name.startswith('json_')])
    print(f"源目录: {source_dir}")
    print(f"输出目录: {output_dir}")
    print(f"找到 {len(json_files)} 个标注文件")

    success_count = 0
    file_names = []

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

                mask = Image.new('L', (orig_width, orig_height), 0)
                draw = ImageDraw.Draw(mask)

                for shape in data.get('shapes', []):
                    points = shape.get('points', [])
                    shape_type = shape.get('shape_type', 'polygon')
                    if not points:
                        continue

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

                stem = json_path.stem
                img_resized_rgb = img_resized.convert('RGB')
                img_resized_rgb.save(str(img_folder / f"{stem}.jpg"), 'JPEG', quality=95)
                mask_resized.save(str(gt_folder / f"{stem}.png"), 'PNG')

                file_names.append(stem)
                success_count += 1
                print(f"  [OK] {json_path.name}")

        except Exception as e:
            print(f"  [错误] {json_path.name}: {e}")

    file_list_path = output_dir / "train.txt"
    with open(file_list_path, 'w', encoding='utf-8') as f:
        for name in file_names:
            f.write(name + '\n')

    print(f"\n转换完成: {success_count} 个样本")
    return output_dir


def create_sam2_config(output_dir):
    output_dir = Path(output_dir)
    config_path = output_dir / "defect_finetune.yaml"

    config_content = f"""# SAM 2.1 自定义数据集微调配置
dataset:
  img_folder: {output_dir / "images"}
  gt_folder: {output_dir / "masks"}
  file_list_txt: {output_dir / "train.txt"}
  multiplier: 2

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
    output_dir = Path("/mnt/c/Users/finn.zhang/project/Datasets/xutong_military_sam2")

    print("=" * 60)
    print("SAM 2.1 训练数据准备")
    print("=" * 60)

    result_dir = convert_labelme_to_sam2_format(source_dir, output_dir, target_size=1024)
    config_path = create_sam2_config(result_dir)

    print(f"\n输出目录: {result_dir}")
    print(f"  ├── images/    (图片)")
    print(f"  ├── masks/     (掩码)")
    print(f"  ├── train.txt  (文件列表)")
    print(f"  └── defect_finetune.yaml (配置)")


if __name__ == "__main__":
    main()
