#!/usr/bin/env python3
"""Batch fix class ID 0.000000 -> 0 in all label files."""
import os

val_dir = '/mnt/c/Users/finn.zhang/project/Datasets/yolo_zhonghua_counter/labels/val'
fixed = 0
for fn in os.listdir(val_dir):
    if fn.endswith('.txt'):
        path = os.path.join(val_dir, fn)
        with open(path) as f:
            lines = f.readlines()
        with open(path, 'w') as f:
            for line in lines:
                line = line.strip()
                if line:
                    parts = line.split()
                    parts[0] = '0'
                    f.write(' '.join(parts) + '\n')
        fixed += 1
print(f'Fixed {fixed} files')
