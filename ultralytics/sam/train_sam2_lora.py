#!/usr/bin/env python3
"""
SAM 2.1 Base+ LoRA 微调训练脚本
绕过 hydra，直接加载模型配置
"""

import sys
import os
import yaml
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# SAM2 路径
SAM2_ROOT = "/home/finn/ultralytics-main-count/sam2"
os.chdir(SAM2_ROOT)
sys.path.insert(0, SAM2_ROOT)

from hydra.core.global_hydra import GlobalHydra
GlobalHydra.instance().clear()


# ============================================================
# LoRA 实现
# ============================================================
class LoRA_qkv(nn.Module):
    def __init__(self, qkv, linear_a_q, linear_b_q, linear_a_v, linear_b_v):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features

    def forward(self, x):
        qkv = self.qkv(x)
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, :self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv


def add_lora_to_sam2(model, r=8, device="cuda"):
    w_As = []
    w_Bs = []

    for param in model.image_encoder.parameters():
        param.requires_grad = False

    for blk in model.image_encoder.trunk.blocks:
        w_qkv_linear = blk.attn.qkv
        dim = w_qkv_linear.in_features

        w_a_q = nn.Linear(dim, r, bias=False).to(device)
        w_b_q = nn.Linear(r, dim, bias=False).to(device)
        w_a_v = nn.Linear(dim, r, bias=False).to(device)
        w_b_v = nn.Linear(r, dim, bias=False).to(device)

        nn.init.kaiming_uniform_(w_a_q.weight, a=5**0.5)
        nn.init.zeros_(w_b_q.weight)
        nn.init.kaiming_uniform_(w_a_v.weight, a=5**0.5)
        nn.init.zeros_(w_b_v.weight)

        w_As.extend([w_a_q, w_a_v])
        w_Bs.extend([w_b_q, w_b_v])

        blk.attn.qkv = LoRA_qkv(w_qkv_linear, w_a_q, w_b_q, w_a_v, w_b_v).to(device)

    return model, w_As, w_Bs


# ============================================================
# 直接加载 SAM2 模型（绕过 hydra）
# ============================================================
def load_sam2_direct(checkpoint_path, device="cuda"):
    """直接从 checkpoint 加载 SAM2 模型"""
    from sam2.modeling.sam2_base import SAM2Base
    from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
    from sam2.modeling.backbones.hieradet import Hiera
    from sam2.modeling.position_encoding import PositionEmbeddingSine
    from sam2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
    from sam2.modeling.sam.transformer import RoPEAttention
    from sam2.modeling.memory_encoder import MemoryEncoder, MaskDownSampler, Fuser, CXBlock

    print("构建 SAM2 Base+ 模型...")

    # Hiera Base+ 配置
    trunk = Hiera(
        embed_dim=112,
        num_heads=2,
        stages=[2, 6, 36, 4],
        global_att_blocks=[23, 33, 43],
        window_pos_embed_bkg_spatial_size=[7, 7],
        window_spec=[8, 4, 16, 8],
    )

    position_encoding = PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
    )

    neck = FpnNeck(
        position_encoding=position_encoding,
        d_model=256,
        backbone_channel_list=[896, 448, 224, 112],
        fpn_top_down_levels=[2, 3],
        fpn_interp_model="nearest",
    )

    image_encoder = ImageEncoder(
        scalp=1,
        trunk=trunk,
        neck=neck,
    )

    # Memory attention
    self_attn = RoPEAttention(
        rope_theta=10000.0,
        feat_sizes=[64, 64],
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
    )

    cross_attn = RoPEAttention(
        rope_theta=10000.0,
        feat_sizes=[64, 64],
        rope_k_repeat=True,
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
        kv_in_dim=64,
    )

    memory_attention_layer = MemoryAttentionLayer(
        activation="relu",
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        self_attention=self_attn,
        d_model=256,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        cross_attention=cross_attn,
    )

    memory_attention = MemoryAttention(
        d_model=256,
        pos_enc_at_input=True,
        layer=memory_attention_layer,
        num_layers=4,
    )

    # Memory encoder
    mask_downsampler = MaskDownSampler(kernel_size=3, stride=2, padding=1)
    fuser = Fuser(
        layer=CXBlock(dim=256, kernel_size=7, padding=3, layer_scale_init_value=1e-6, use_dwconv=True),
        num_layers=2,
    )
    memory_encoder = MemoryEncoder(
        out_dim=64,
        position_encoding=PositionEmbeddingSine(num_pos_feats=64, normalize=True, scale=None, temperature=10000),
        mask_downsampler=mask_downsampler,
        fuser=fuser,
    )

    # 构建完整模型
    model = SAM2Base(
        image_encoder=image_encoder,
        memory_attention=memory_attention,
        memory_encoder=memory_encoder,
        num_maskmem=7,
        image_size=1024,
        sigmoid_scale_for_mem_enc=20.0,
        sigmoid_bias_for_mem_enc=-10.0,
        use_mask_input_as_output_without_sam=True,
        directly_add_no_mem_embed=True,
        no_obj_embed_spatial=True,
        use_high_res_features_in_sam=True,
        multimask_output_in_sam=True,
        iou_prediction_use_sigmoid=True,
        use_obj_ptrs_in_encoder=True,
        add_tpos_enc_to_obj_ptrs=True,
        proj_tpos_enc_in_obj_ptrs=True,
        use_signed_tpos_enc_to_obj_ptrs=True,
        only_obj_ptrs_in_the_past_for_eval=True,
        pred_obj_scores=True,
        pred_obj_scores_mlp=True,
        fixed_no_obj_ptr=True,
        multimask_output_for_tracking=True,
        use_multimask_token_for_obj_ptr=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        use_mlp_for_obj_ptr_proj=True,
    )

    # 加载权重
    print(f"加载权重: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]

    # 加载匹配的权重
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)

    print(f"加载了 {len(pretrained_dict)}/{len(model_dict)} 层权重")
    return model.to(device)


# ============================================================
# 数据集
# ============================================================
class SAM2Dataset(Dataset):
    def __init__(self, img_folder, mask_folder, file_list):
        self.img_folder = Path(img_folder)
        self.mask_folder = Path(mask_folder)

        with open(file_list, 'r') as f:
            self.file_names = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        name = self.file_names[idx]

        img_path = self.img_folder / f"{name}.jpg"
        if not img_path.exists():
            img_path = self.img_folder / f"{name}.png"
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        mask_path = self.mask_folder / f"{name}.png"
        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask)
        mask = (mask > 127).astype(np.float32)

        return {"image": image, "mask": mask, "name": name}


# ============================================================
# 训练
# ============================================================
def train():
    CHECKPOINT = os.path.join(SAM2_ROOT, "checkpoints/sam2.1_hiera_base_plus.pt")
    DATA_DIR = "/mnt/c/Users/finn.zhang/project/Datasets/xutong_military_sam2"
    OUTPUT_DIR = "/home/finn/ultralytics-main-count/ultralytics/sam/weights"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    BATCH_SIZE = 1
    NUM_EPOCHS = 40
    LEARNING_RATE = 5e-6
    LORA_RANK = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载模型
    sam2_model = load_sam2_direct(CHECKPOINT, device)

    # 添加 LoRA
    print(f"添加 LoRA (r={LORA_RANK})...")
    sam2_model, w_As, w_Bs = add_lora_to_sam2(sam2_model, r=LORA_RANK, device=device)

    # 分割头（在训练循环外创建，确保参数被优化器跟踪）
    seg_head = nn.Conv2d(256, 1, kernel_size=1).to(device)

    total_params = sum(p.numel() for p in sam2_model.parameters()) + sum(p.numel() for p in seg_head.parameters())
    trainable_params = sum(p.numel() for p in sam2_model.parameters() if p.requires_grad) + sum(p.numel() for p in seg_head.parameters())
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    # 数据集
    dataset = SAM2Dataset(
        os.path.join(DATA_DIR, "images"),
        os.path.join(DATA_DIR, "masks"),
        os.path.join(DATA_DIR, "train.txt")
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # 优化器（包含 seg_head 参数）
    optimizer = torch.optim.Adam(
        [p for p in sam2_model.parameters() if p.requires_grad] + list(seg_head.parameters()),
        lr=LEARNING_RATE, weight_decay=0.01
    )

    criterion = nn.BCEWithLogitsLoss()

    print(f"\n开始训练...")
    print(f"  数据集: {len(dataset)} 样本")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  LoRA Rank: {LORA_RANK}")
    print("-" * 60)

    best_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        sam2_model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(dataloader):
            images = batch["image"]
            masks = batch["mask"]

            # 前向传播 - 使用 image encoder
            img_tensor = images.permute(0, 3, 1, 2).float().to(device) / 255.0

            with torch.autocast("cuda", dtype=torch.bfloat16):
                # 获取 image embedding
                backbone_out = sam2_model.image_encoder(img_tensor)

                # 简化的损失计算：使用 backbone 输出和 mask
                # 将 backbone 输出上采样到 mask 尺寸
                if isinstance(backbone_out, dict):
                    feat = backbone_out["vision_features"]
                else:
                    feat = backbone_out

                # 上采样特征到 mask 尺寸
                feat_up = nn.functional.interpolate(
                    feat.float(), size=masks.shape[1:], mode='bilinear', align_corners=False
                )

                # 简单的分割头（复用外部创建的 seg_head）
                seg_logits = seg_head(feat_up)
                seg_logits = seg_logits.squeeze(1)

                mask_gt = masks.float().to(device)
                loss = criterion(seg_logits, mask_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 5 == 0:
                print(f"  Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] 平均 Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            lora_state = {
                'w_As': [w.state_dict() for w in w_As],
                'w_Bs': [w.state_dict() for w in w_Bs],
                'seg_head': seg_head.state_dict(),
                'epoch': epoch,
                'loss': best_loss
            }
            torch.save(lora_state, os.path.join(OUTPUT_DIR, "sam2_lora_best.pt"))
            print(f"  保存最佳模型 (loss: {best_loss:.4f})")

    print(f"\n训练完成! 最佳 Loss: {best_loss:.4f}")
    print(f"模型保存在: {OUTPUT_DIR}/sam2_lora_best.pt")


if __name__ == "__main__":
    train()
