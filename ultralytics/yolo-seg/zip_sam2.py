#!/usr/bin/env python3
"""压缩 SAM 2.1 数据集为 ZIP"""
import zipfile
import os
from pathlib import Path

source_dir = Path('/mnt/c/Users/finn.zhang/project/Datasets/xutong_military_sam2')
zip_path = Path('/mnt/c/Users/finn.zhang/project/Datasets/xutong_military_sam2.zip')

print(f"压缩 {source_dir} -> {zip_path}")

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            file_path = Path(root) / file
            arcname = file_path.relative_to(source_dir.parent)
            zipf.write(file_path, arcname)
            print(f"  {arcname}")

size_mb = zip_path.stat().st_size / 1024 / 1024
print(f"\n完成! ZIP 大小: {size_mb:.1f} MB")
