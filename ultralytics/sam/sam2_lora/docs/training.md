# SAM2.1 LoRA 训练使用文档

## 概述

使用 SAM2.1 Hiera-Base+ 预训练权重，通过 LoRA 微调训练缺陷分割+分类模型。

**训练脚本**: `sam2_lora/train.py`（项目中唯一的训练入口）

## 环境要求

```bash
conda activate yolo
cd /home/finn/ultralytics-main-count
```

## 数据集结构

```
dataset/
├── images/              # 图片 (*.jpg/*.png)
├── masks/               # 二值 mask (0=背景, 255=缺陷)
├── labels/              # YOLO bbox 标注 (class_id cx cy w h)
├── points/              # 点提示 (label x y)
├── train.txt            # 文件列表: "name class_id"
└── class_map.json       # 类别映射: {"0": 0, "1": 1, ...}
```

### train.txt 格式

```
1783648000202864 0
1783648147949539 0
1783654699311719 1
```

### class_map.json 格式

```json
{
  "0": 0,
  "1": 1,
  "2": 2,
  "3": 3,
  "4": 4
}
```

### points/ 格式

```
1 0.512 0.345   # 前景点 (label=1)
1 0.498 0.367
0 0.123 0.456   # 背景点 (label=0)
0 0.789 0.234
```

## 配置文件 (`config.py`)

```python
# 路径
DATA_DIR = Path('/mnt/c/Users/finn.zhang/project/Datasets/sam_dataset')
OUTPUT_DIR = Path('/home/finn/ultralytics-main-count/ultralytics/sam/output_xutong_cls')
CHECKPOINT_FILE = WEIGHTS_DIR / 'sam2.1_hiera_base_plus.pt'

# 模型架构 (SAM 2.1 Hiera-Base+)
IMAGE_SIZE = 1024
HIERA_CONFIG = {
    'embed_dim': 112,
    'num_heads': 2,
    'stages': [2, 3, 16, 3],          # 24 blocks
    'window_pos_embed_bkg_spatial_size': [14, 14],
}
FPN_CONFIG = {
    'backbone_channel_list': [896, 448, 224, 112],
    'd_model': 256,
}

# LoRA
LORA_RANK = 8
LORA_ALPHA = 16

# 训练
BATCH_SIZE = 1
NUM_EPOCHS = 100
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4

# 分类
NUM_CLASSES = 5
CLASS_LOSS_WEIGHT = 0.3
```

## 训练步骤

### 1. 修改配置

编辑 `ultralytics/sam/sam2_lora/config.py`:

```python
DATA_DIR = Path('/mnt/c/Users/finn.zhang/project/Datasets/sam_dataset')
NUM_EPOCHS = 100
```

### 2. 删除旧 checkpoint（可选）

```bash
rm -rf /home/finn/ultralytics-main-count/ultralytics/sam/output_xutong_cls/best
```

### 3. 启动训练

```bash
# 方式1: 直接运行
python -m sam2_lora.train

# 方式2: 后台运行（推荐）
nohup python -m sam2_lora.train > ultralytics/sam/train.log 2>&1 &
```

### 4. 监控训练

```bash
tail -f ultralytics/sam/train.log
```

输出示例:
```
Epoch [1/100] Step 0: loss=1.2771  (seg=1.0773, cls=0.6660)
Epoch [1/100] Step 50: loss=1.1539  (seg=0.9635, cls=0.6346)
Epoch [1/100]  train_loss=1.1013
  test_loss=0.7571
  Checkpoint saved: .../output_xutong_cls/best  (epoch=1, loss=0.7571)
```

### 5. 查看 GPU 使用

```bash
nvidia-smi
```

## 训练脚本对比

| 脚本 | 流程 | 分割 | 分类 | 效果 |
|------|------|------|------|------|
| `train.py` | 完整 SAM2 pipeline | ✅ prompt encoder + mask decoder | ✅ ClassHead | **最佳** |
| `train_sam2_lora.py` | 简化 image_encoder | ❌ Conv2d seg head | ❌ | 差 |
| `train_sam2_lora_cls.py` | 简化 image_encoder | ❌ Conv2d seg head | ✅ ClassHead | 差 |

## Checkpoint 管理

### 保存位置

```
output_xutong_cls/
└── best/
    └── checkpoint.pt  (包含 LoRA 权重 + ClassHead 权重)
```

### Checkpoint 内容

```python
{
    'epoch': 100,
    'loss': 0.1451,
    'best_loss': 0.1451,
    'lora_weights': ...,           # image_encoder.trunk state_dict
    'class_head': ...,              # ClassHead state_dict
    'num_classes': 5,
    'optimizer_state': ...,         # 优化器状态（可选恢复训练）
}
```

### 恢复训练

训练脚本支持自动恢复：如果 `output_xutong_cls/best/checkpoint.pt` 存在，会自动加载并从上次 epoch 继续。

## 超参数参考

| 参数 | 默认值 | 建议范围 | 说明 |
|------|--------|---------|------|
| `LORA_RANK` | 8 | 4-16 | 越大容量越大，显存越多 |
| `LORA_ALPHA` | 16 | rank×2 | LoRA 缩放因子 |
| `LEARNING_RATE` | 5e-5 | 1e-5 ~ 1e-4 | 学习率 |
| `BATCH_SIZE` | 1 | 1-2 | RTX 4080 12GB 最多2 |
| `NUM_EPOCHS` | 100 | 40-200 | 86张图片建议100+ |
| `CLASS_LOSS_WEIGHT` | 0.3 | 0.1-0.5 | 分类损失权重 |

## 显存估算

| 配置 | 显存 |
|------|------|
| Base+ rank=8 batch=1 imgsz=1024 | ~7.3 GB |
| Base+ rank=16 batch=1 imgsz=1024 | ~8.5 GB |
| Base+ rank=8 batch=2 imgsz=1024 | ~10 GB |

RTX 4080 (12GB): 推荐 rank=8, batch=1。

## 常见问题

**Q: 训练 loss 不下降？**
A: 检查 train.txt 格式是否正确（需要 `name class_id`），检查 masks 是否为二值图。

**Q: 分类准确率低？**
A: 数据集类别不平衡时，调整 `CLASS_LOSS_WEIGHT` 或扩充数据集。

**Q: 如何用新数据集训练？**
A: 只需修改 `config.py` 中的 `DATA_DIR` 路径，确保数据集格式正确。

**Q: 训练完成后如何推理？**
A: 参考 `docs/inference.md`。
