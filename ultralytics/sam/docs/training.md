# SAM 2.1 LoRA 训练文档

## 概述

本文档说明如何使用 SAM 2.1 Hiera-Large 模型对 CGFSDS-9 数据集进行 LoRA 微调训练。

## 快速开始

```bash
# 1. 激活环境
conda activate sam2-lora

# 2. 确保数据集已拆分（每类 3 张测试集）
python split_dataset.py

# 3. 开始训练
python -m sam2_lora.train
```

## 训练流程

### 1. 模型构建

训练脚本自动完成以下步骤：

1. **构建 SAM2Base 模型** — 使用 Hiera-Large 骨干网络（48 个 Transformer 块）
2. **加载预训练权重** — 从 `weights/sam2-1-hiera-large/model.safetensors` 加载
3. **应用 LoRA** — 冻结除 LoRA 适配器外的所有参数
4. **移动到 GPU** — 模型移至 CUDA 设备

### 2. 数据加载

- **训练集**: 1042 张图片（9 类缺陷）
- **测试集**: 27 张图片（每类 3 张）
- **数据增强**: 随机采样前景点作为 SAM 点提示（最多 16 个点）
- **批处理**: 使用 `pad_collate` 函数将不同长度的点坐标填充到统一长度

### 3. 训练循环

每个 epoch 包含：

1. **前向传播** — 图像编码器（含 LoRA）→ 提示编码器 → 掩码解码器
2. **损失计算** — BCE Loss + Dice Loss
3. **反向传播** — 仅更新 LoRA 适配器参数
4. **梯度裁剪** — 最大范数 1.0
5. **参数更新** — AdamW 优化器
6. **验证** — 在测试集上评估损失
7. **保存 Checkpoint** — 每 10 个 epoch + 最佳模型

### 4. 损失函数

```python
loss = BCE_loss + Dice_loss
```

- **BCE Loss**: 二元交叉熵，衡量像素级分类准确性
- **Dice Loss**: 衡量预测 mask 与真实 mask 的重叠程度

## 超参数调优

### LoRA 秩（Rank）

LoRA 秩决定了适配器的表达能力：

| 秩 | 可训练参数 | 显存需求 | 推荐场景 |
|----|-----------|----------|----------|
| 8 | ~3.5M | ~10GB | 快速实验 |
| 16 | ~7M | ~12GB | **默认推荐** |
| 32 | ~14M | ~16GB | 更好效果 |
| 64 | ~28M | ~20GB | 高精度需求 |
| 128 | ~56M | ~25GB | 最大表达力（需大显存） |

修改 `config.py` 中的 `LORA_RANK` 和 `LORA_ALPHA`：

```python
LORA_RANK = 64      # 增大秩
LORA_ALPHA = 128    # alpha 通常 = 2 × rank
```

### 学习率

- 默认 `1e-4` 适合大多数情况
- 如果损失震荡，可降至 `5e-5`
- 如果收敛过慢，可增至 `2e-4`

### 批大小与梯度累积

显存不足时使用梯度累积：

```python
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4  # 有效批大小 = 1 × 4 = 4
```

## 输出文件

训练过程中生成的文件：

```
output/
├── best/
│   └── checkpoint.pt      # 最佳模型（最低验证损失）
├── epoch-10/
│   └── checkpoint.pt      # 第 10 轮 checkpoint
├── epoch-20/
│   └── checkpoint.pt      # 第 20 轮 checkpoint
└── ...
```

### Checkpoint 内容

```python
{
    'epoch': 1,              # 轮数
    'loss': 0.3185,          # 验证损失
    'lora_weights': {        # LoRA 适配器权重
        'base_model.model.0.attn.qkv.lora_A.default.weight': ...,
        'base_model.model.0.attn.qkv.lora_B.default.weight': ...,
        ...
    }
}
```

## 训练监控

训练过程中会打印以下信息：

```
Device: cuda
  GPU: NVIDIA GeForce RTX 4090
  VRAM: 51.5 GB
Building model ...
Loading checkpoint ...
Checkpoint loaded: 898 keys matched, 5 missing, 2 unexpected
Applying LoRA ...
LoRA rank=16, alpha=32
Trainable parameters: 7,042,369 / 231,487,714 (3.04%)
Loading dataset ...
  Train samples: 1042
  Test  samples: 27
  Epoch [1/100] Step 0: loss=0.3371  (bce=0.0422, dice=0.2948)
  Epoch [1/100] Step 50: loss=0.4006  (bce=0.1570, dice=0.2436)
  ...
Epoch [1/100]  train_loss=0.2567
  test_loss=0.3185
  Checkpoint saved: E:\sam\output\best  (epoch=1, loss=0.3185)
```

## 故障排除

### CUDA 内存不足 (OOM)

```
torch.OutOfMemoryError: CUDA out of memory
```

**解决方案**:
1. 减小 `LORA_RANK`（如 128 → 64 → 32）
2. 减小 `BATCH_SIZE`（如 4 → 2 → 1）
3. 启用梯度累积：`GRADIENT_ACCUMULATION_STEPS = 2`
4. 使用混合精度训练（在 `train.py` 中启用 `torch.cuda.amp.autocast`）

### CUDA 内核错误

```
RuntimeError: CUDA error: an illegal instruction was encountered
```

**原因**: GPU 驱动不稳定或虚拟化环境问题

**解决方案**:
1. 代码内置重试机制，自动跳过出错的 step
2. 使用 `CUDA_LAUNCH_BLOCKING=1` 环境变量启用同步模式
3. 降低 LoRA 秩减少计算复杂度
4. 更新 GPU 驱动

### Checkpoint 加载问题

```
RuntimeError: Error(s) in loading state_dict
```

**解决方案**:
- 确保 LoRA 秩与训练时一致
- 确保模型架构配置未修改
- 使用 `strict=False` 加载（已内置）

## 性能参考

在不同配置下的训练时间（100 epochs, 1042 张图片）：

| 配置 | 每 step 时间 | 每 epoch 时间 | 总时间 |
|------|-------------|---------------|--------|
| rank=16, batch=1 | ~0.3s | ~5 min | ~8 hrs |
| rank=32, batch=1 | ~0.4s | ~7 min | ~11 hrs |
| rank=64, batch=1 | ~0.5s | ~9 min | ~15 hrs |
| rank=128, batch=1 | ~0.7s | ~12 min | ~20 hrs |

## 技术细节

### 模型架构

- **骨干网络**: Hiera-Large (48 blocks, embed_dim=144)
- **FPN 颈部**: 特征金字塔网络，输出 256 通道
- **提示编码器**: SAM 标准提示编码器（点/框/掩码）
- **掩码解码器**: TwoWayTransformer + IoU 预测头

### LoRA 应用范围

LoRA 适配器应用于 Hiera 骨干网络的所有 Linear 和 Conv2d 层：
- 注意力 QKV 投影
- 注意力输出投影
- MLP 全连接层
- 各种投影层

其他组件（掩码解码器、提示编码器、记忆注意力、记忆编码器）保持冻结。

### 损失函数细节

```python
# BCE Loss（二元交叉熵）
loss_bce = F.binary_cross_entropy_with_logits(pred, mask_gt)

# Dice Loss
pred_sigmoid = pred.sigmoid()
intersection = (pred_sigmoid * mask_gt).sum()
union = pred_sigmoid.sum() + mask_gt.sum()
loss_dice = 1 - 2 * intersection / (union + 1e-8)

# 总损失
loss = loss_bce + loss_dice
```

## 参考资料

- [SAM 2 论文](https://arxiv.org/abs/2408.00714)
- [SAM 2 GitHub](https://github.com/facebookresearch/sam2)
- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [PEFT 库](https://github.com/huggingface/peft)
