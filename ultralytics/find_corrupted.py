import os
import shutil

from PIL import Image

src = r"/mnt/c/Users/finn.zhang/project/Datasets/20260525_29_blue/OK/all"
dst = r"/mnt/c/Users/finn.zhang/project/Datasets/20260525_29_blue/OK/all_corrupted"

os.makedirs(dst, exist_ok=True)
exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

corrupted = []
for f in sorted(os.listdir(src)):
    if os.path.splitext(f.lower())[1] in exts:
        path = os.path.join(src, f)
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception:
            corrupted.append(f)
            print(f"Bad: {f}")

if corrupted:
    print(f"\nFound {len(corrupted)} corrupted images. Moving...")
    for f in corrupted:
        shutil.move(os.path.join(src, f), os.path.join(dst, f))
    print(f"Moved {len(corrupted)} files to {dst}")
else:
    print("No corrupted images found.")
