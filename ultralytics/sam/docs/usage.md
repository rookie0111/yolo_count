# SAM 2.1 LoRA 使用文档

## 项目概述

本项目基于 **SAM 2.1 Hiera-Large** 模型，使用 **LoRA（Low-Rank Adaptation）** 对表面缺陷数据集 CGFSDS-9 进行微调，实现缺陷分割。

## 目录结构

```
E:\sam\
├── sam2_lora/                  # 核心代码包
│   ├── __init__.py
│   ├── config.py               # 配置参数
│   ├── model.py                # 模型构建与 LoRA 适配
│   ├── dataset.py              # 数据集加载与批处理
│   ├── train.py                # 训练入口
│   └── infer.py                # 推理入口
├── data/
│   └── CGFSDS-9-split/         # 拆分后的数据集
│       ├── train/              # 训练集（1042 张）
│       │   ├── AL/Images/, AL/GT/
│       │   ├── MT/Images/, MT/GT/
│       │   ├── Steel/Images/, Steel/GT/
│       │   └── ...
│       └── test/               # 测试集（每类 3 张，共 27 张）
│           ├── AL/Images/, AL/GT/
│           └── ...
├── weights/
│   └── sam2-1-hiera-large/     # SAM 2.1 预训练权重
│       ├── model.safetensors
│       └── config.json
├── output/                     # 训练输出目录
│   ├── best/checkpoint.pt      # 最佳模型 checkpoint
│   └── epoch-10/checkpoint.pt  # 周期性 checkpoint
├── docs/
│   ├── usage.md                # 使用文档（本文件）
│   └── training.md             # 训练文档
└── split_dataset.py            # 数据集拆分工具（已运行）
```

## 环境要求

- **操作系统**: Windows 10/11
- **Python**: 3.10（推荐使用 Conda 环境）
- **PyTorch**: 2.6.0+cu124
- **GPU**: NVIDIA RTX 4090（24GB+ 显存）或同等级别
- **依赖包**:
  - `sam2` (v1.1.0) — Meta SAM2 官方包
  - `peft` — HuggingFace PEFT 库（LoRA 适配）
  - `safetensors` — 安全张量格式
  - `transformers` — HuggingFace Transformers
  - `torchvision` — 计算机视觉工具

## 安装步骤

```bash
# 1. 创建 Conda 环境
conda create -n sam2-lora python=3.10 -y
conda activate sam2-lora

# 2. 安装 PyTorch（CUDA 12.4）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. 安装 sam2（使用清华镜像加速）
pip install sam2 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 安装其他依赖
pip install peft safetensors transformers pillow numpy
```

## 使用方法

### 推理（单张图片）

```bash
# 激活环境
conda activate sam2-lora

# 运行推理
python -m sam2_lora.infer <checkpoint_dir> <image_path> [--output mask.png]

# 示例
python -m sam2_lora.infer output/best data/CGFSDS-9-split/test/AL/Images/AL_Con_6.jpg -o result.png
```

**参数说明**:
- `checkpoint_dir`: 包含 `checkpoint.pt` 的目录（由训练脚本生成）
- `image_path`: 输入图片路径
- `--output, -o`: 输出 mask 路径（可选，默认仅打印 IoU 预测）

### Python API 推理

```python
from sam2_lora.infer import load_trained_model, predict

# 加载模型
model = load_trained_model('output/best')

# 预测（单点提示）
mask, iou = predict(model, 'path/to/image.jpg')
print(f'IoU: {iou.item():.4f}')

# 预测（多点提示）
import torch
coords = torch.tensor([[[500, 500], [600, 600]]]).cuda().float()
labels = torch.ones(1, 2, dtype=torch.long).cuda()
mask, iou = predict(model, 'path/to/image.jpg', coords, labels)
```

### 模型输出

- `mask`: `(1, H, W)` 浮点张量，值域 `[0, 1]`，表示缺陷区域的概率
- `iou_prediction`: 标量张量，预测的 IoU 置信度

## 数据集类别

CGFSDS-9 包含 9 类表面缺陷：

| 类别 | 描述 | 训练集 | 测试集 |
|------|------|--------|--------|
| AL | 铝材缺陷 | 60 | 3 |
| MT | 金属缺陷 | 140 | 3 |
| Steel | 钢材缺陷 | 597 | 3 |
| Tube_Ak | 管材 Ak | 37 | 3 |
| Tube_Gs | 管材 Gs | 45 | 3 |
| Tube_Lw | 管材 Lw | 25 | 3 |
| Tube_Tq | 管材 Tq | 47 | 3 |
| Tube_Xs | 管材 Xs | 41 | 3 |
| Tube_Yh | 管材 Yh | 50 | 3 |
| **总计** | | **1042** | **27** |

## 配置参数

所有超参数在 `sam2_lora/config.py` 中集中管理：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `LORA_RANK` | 16 | LoRA 秩（越大越有表达力，显存需求越高） |
| `LORA_ALPHA` | 32 | LoRA 缩放因子 |
| `LORA_DROPOUT` | 0.1 | LoRA Dropout 率 |
| `BATCH_SIZE` | 1 | 批大小 |
| `NUM_EPOCHS` | 100 | 训练轮数 |
| `LEARNING_RATE` | 1e-4 | 学习率 |
| `WEIGHT_DECAY` | 1e-4 | 权重衰减 |
| `GRAD_CLIP_NORM` | 1.0 | 梯度裁剪范数 |
| `MAX_POINTS` | 16 | 每张图片的最大点提示数 |
| `IMAGE_SIZE` | 1024 | 输入图片尺寸 |
| `SAVE_EVERY_N_EPOCHS` | 10 | Checkpoint 保存间隔 |

## Checkpoint 格式

训练生成的 `checkpoint.pt` 包含：

```python
{
    'epoch': int,           # 当前轮数
    'loss': float,          # 验证损失
    'lora_weights': dict,   # LoRA 适配器权重
}
```

## 注意事项

1. **CUDA 稳定性**: 在某些虚拟化 GPU 环境中可能出现 CUDA 错误。代码内置重试机制，最多重试 3 次。
2. **显存需求**: rank=16 + batch_size=1 约需 12-15GB 显存；rank=128 约需 20-25GB。
3. **数据格式**: 图片支持 JPG/PNG，Ground Truth 为 PNG 二值 mask。
4. **预训练权重**: 从 ModelScope 下载的 `AI-ModelScope/sam2-hiera-large-hf`，使用 safetensors 格式。
