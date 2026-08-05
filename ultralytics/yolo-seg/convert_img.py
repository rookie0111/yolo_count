#!/usr/bin/env python3
"""
LabelMe → YOLO-seg 转换（img 文件夹）
"""
import json, os, random, shutil
from pathlib import Path
from PIL import Image

source_dir = Path('/mnt/c/Users/finn.zhang/project/Datasets/xutong_military/img')
output_dir = Path('/mnt/c/Users/finn.zhang/project/Datasets/xutong_military_yolo_seg_v3')
val_ratio = 0.2
random.seed(42)

# 创建目录
for split in ['train', 'val']:
    (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

# 获取 JSON 文件（排除脚本文件）
json_files = sorted([f for f in source_dir.glob('*.json')
                     if not f.name.startswith('convert_') and not f.name.startswith('json_')])
print(f'找到 {len(json_files)} 个标注文件')

# 收集类别
label_map = {}
for json_path in json_files:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for shape in data.get('shapes', []):
        label = shape.get('label', '0')
        if label not in label_map:
            label_map[label] = len(label_map)

print(f'类别映射: {label_map}')

# 划分
random.shuffle(json_files)
val_count = max(1, int(len(json_files) * val_ratio))
val_files = set(json_files[:val_count])

stats = {'train': 0, 'val': 0}

for json_path in json_files:
    split = 'val' if json_path in val_files else 'train'

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    img_name = data.get('imagePath', json_path.stem + '.png')
    img_path = source_dir / img_name
    if not img_path.exists():
        continue

    img_width = data.get('imageWidth')
    img_height = data.get('imageHeight')

    lines = []
    for shape in data.get('shapes', []):
        label = shape.get('label', '0')
        points = shape.get('points', [])
        shape_type = shape.get('shape_type', 'polygon')
        if not points:
            continue

        class_id = label_map.get(label, 0)

        if shape_type == 'polygon':
            normalized = []
            for p in points:
                x = max(0.0, min(1.0, p[0] / img_width))
                y = max(0.0, min(1.0, p[1] / img_height))
                normalized.extend([f'{x:.6f}', f'{y:.6f}'])
            lines.append(f'{class_id} ' + ' '.join(normalized))
        elif shape_type == 'rectangle':
            x1, y1 = points[0]
            x2, y2 = points[1]
            rect_points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            normalized = []
            for p in rect_points:
                x = max(0.0, min(1.0, p[0] / img_width))
                y = max(0.0, min(1.0, p[1] / img_height))
                normalized.extend([f'{x:.6f}', f'{y:.6f}'])
            lines.append(f'{class_id} ' + ' '.join(normalized))

    if not lines:
        continue

    stem = json_path.stem
    img_ext = Path(img_name).suffix or '.png'

    shutil.copy2(str(img_path), str(output_dir / 'images' / split / f'{stem}{img_ext}'))
    with open(output_dir / 'labels' / split / f'{stem}.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')

    stats[split] += 1
    print(f'[{split}] {json_path.name} -> {len(lines)} targets')

# data.yaml
names_list = sorted(label_map.items(), key=lambda x: x[1])
names_str = ', '.join(f"'{k}'" for k, v in names_list)
yaml_content = f"""path: {output_dir}
train: images/train
val: images/val
nc: {len(label_map)}
names: [{names_str}]
"""
with open(output_dir / 'data.yaml', 'w') as f:
    f.write(yaml_content)

print(f'\n训练集: {stats["train"]} | 验证集: {stats["val"]}')
print(f'类别: {label_map}')
print(f'输出: {output_dir}')
