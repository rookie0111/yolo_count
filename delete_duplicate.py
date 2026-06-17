import os

folder1 = r"/mnt/c/Users/finn.zhang/project/Datasets/hainanjiahua_20260416/NG_1"
folder2 = r"/mnt/c/Users/finn.zhang/project/Datasets/hainanjiahua_20260416/ng"

extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

files1 = set()
for f in os.listdir(folder1):
    if os.path.splitext(f.lower())[1] in extensions:
        files1.add(os.path.splitext(f)[0])  # 只保存文件名（不含扩展名）

files2 = set()
for f in os.listdir(folder2):
    if os.path.splitext(f.lower())[1] in extensions:
        files2.add(os.path.splitext(f)[0])

# 找出重名的文件
common = files1 & files2
print(f"Folder1 文件数: {len(files1)}")
print(f"Folder2 文件数: {len(files2)}")
print(f"重名文件数: {len(common)}")

if len(common) == 0:
    print("没有重名文件")
else:
    deleted = 0
    for name in common:
        for ext in extensions:
            path = os.path.join(folder1, name + ext)
            if os.path.exists(path):
                os.remove(path)
                deleted += 1
                print(f"Deleted: {name}{ext}")

    print(f"/n共删除 {deleted} 个文件")
