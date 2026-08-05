# SAM2.1 LoRA 推理可视化使用文档

## 脚本对比

| 脚本 | 功能 | 物理坐标 |
|------|------|---------|
| `infer_vis.py` | 基础可视化（mask+框+分类标签+面积+尺寸） | ❌ |
| `infer_physical.py` | 物理坐标可视化（mm² 面积 + mm 尺寸） | ✅ |
| `infer_sam2_full_cls.py` | 完整版（含 IoU 计算） | ❌ |

## 环境要求

```bash
conda activate yolo
cd /home/finn/ultralytics-main-count
```

## 1. 基础可视化 (`infer_vis.py`)

### 功能

- 完整 SAM2 pipeline (prompt encoder + mask decoder)
- 逐缺陷独立检测框（连通域分析）
- 分类标签 + 置信度 + 像素面积 + 宽高
- 防重叠标签放置 + 引导线

### 命令

```bash
python -m sam2_lora.infer_vis \
    --checkpoint /home/finn/ultralytics-main-count/ultralytics/sam/output_xutong_cls/best/checkpoint.pt \
    --data-root /mnt/c/Users/finn.zhang/project/Datasets/sam_dataset \
    --output /home/finn/ultralytics-main-count/ultralytics/sam/results/infer_vis \
    --split train
```

### 参数

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--checkpoint` | ✅ | — | 模型 checkpoint 路径 |
| `--data-root` | ✅ | — | 数据集根目录 |
| `--output` | ✅ | — | 输出目录 |
| `--split` | ❌ | train | 使用 train.txt 或 scan images/ |
| `--device` | ❌ | cuda | 推理设备 |
| `--threshold` | ❌ | 0.5 | mask 二值化阈值 |
| `--min-area` | ❌ | 20 | 最小缺陷面积过滤 |
| `--merge-distance` | ❌ | 0 | 形态学闭合距离（合并相邻缺陷） |

### 输出示例

每张图片包含：
- 绿色半透明 mask 叠加 (alpha=0.6)
- 缺陷轮廓线 (thickness=3)
- 检测框 (thickness=3)
- 标签: `class0:0.95` + `area=1234px 45x32px`
- 引导线: 标签 → 检测框中心

---

## 2. 物理坐标可视化 (`infer_physical.py`)

### 功能

- 所有基础可视化功能
- 像素坐标 → 物理坐标 (mm) 转换
- 物理面积 (mm²) + 物理尺寸 (mm)
- 需要相机内参参数

### 命令

```bash
python -m sam2_lora.infer_physical \
    --checkpoint /home/finn/ultralytics-main-count/ultralytics/sam/output_xutong_cls/best/checkpoint.pt \
    --data-root /mnt/c/Users/finn.zhang/project/Datasets/sam_dataset \
    --output /home/finn/ultralytics-main-count/ultralytics/sam/results/infer_physical \
    --split train \
    --fx 208.333 --fy 208.333 --cx 2736 --cy 1824
```

### 参数

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--checkpoint` | ✅ | — | 模型 checkpoint 路径 |
| `--data-root` | ✅ | — | 数据集根目录 |
| `--output` | ✅ | — | 输出目录 |
| `--fx` | ✅ | — | 相机焦距 fx (pixel) |
| `--fy` | ✅ | — | 相机焦距 fy (pixel) |
| `--cx` | ✅ | — | 光心 cx (pixel) |
| `--cy` | ✅ | — | 光心 cy (pixel) |
| `--split` | ❌ | train | 数据集 split |
| `--threshold` | ❌ | 0.5 | mask 二值化阈值 |
| `--min-area` | ❌ | 20 | 最小缺陷面积过滤 |
| `--merge-distance` | ❌ | 0 | 形态学闭合距离 |

### 坐标转换公式

```
物理X = (pixel_x - cx) / fx   (mm)
物理Y = (pixel_y - cy) / fy   (mm)
物理宽度 = pixel_width / fx   (mm)
物理高度 = pixel_height / fy   (mm)
物理面积 = pixel_area / (fx × fy)  (mm²)
```

### 输出示例

标签格式:
- Line 1: `class0:0.95`（类别 + 置信度）
- Line 2: `area=12.34mm2 box:2.1x1.5mm`（物理面积 + 物理尺寸）

---

## 3. 完整版 (`infer_sam2_full_cls.py`)

```bash
python ultralytics/sam/infer_sam2_full_cls.py \
    --checkpoint /path/to/checkpoint.pt \
    --data-root /path/to/dataset \
    --output /path/to/output \
    --split train
```

功能：含 GT mask 的 IoU 计算、逐缺陷框、分类标签、面积+尺寸。

---

## 数据集格式要求

```
dataset/
├── images/          # 图片 (*.jpg/*.png)
├── masks/           # 二值 mask (推理时可选)
├── train.txt        # 文件列表 (name 或 name class_id)
└── class_map.json   # 类别映射 (可选)
```

## 常见问题

**Q: 推理很慢？**
A: 首次推理需要加载 SAM2 模型（~10s），后续每张约0.5-1s。

**Q: 没有检测到缺陷？**
A: 降低 `--threshold` 值（如 0.3），或检查数据集格式。

**Q: 如何合并相邻缺陷？**
A: 设置 `--merge-distance 15`（值越大合并范围越广）。
