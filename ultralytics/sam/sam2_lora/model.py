"""SAM 2.1 Hiera-Large model construction and LoRA adaptation.

Provides `build_model()` to create the full SAM2Base with the correct
architecture, `load_checkpoint()` to restore pretrained weights, and
`apply_lora()` to attach LoRA adapters to the image encoder trunk.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file as load_safetensors

from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.backbones.image_encoder import FpnNeck, ImageEncoder
from sam2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
from sam2.modeling.memory_encoder import CXBlock, Fuser, MaskDownSampler, MemoryEncoder
from sam2.modeling.position_encoding import PositionEmbeddingSine
from sam2.modeling.sam.transformer import RoPEAttention
from sam2.modeling.sam2_base import SAM2Base

import sam2_lora.config as cfg


def _build_image_encoder() -> ImageEncoder:
    """Construct the image encoder (Hiera trunk + FPN neck)."""
    trunk = Hiera(
        embed_dim=cfg.HIERA_CONFIG['embed_dim'],
        num_heads=cfg.HIERA_CONFIG['num_heads'],
        stages=cfg.HIERA_CONFIG['stages'],
        global_att_blocks=cfg.HIERA_CONFIG['global_att_blocks'],
        window_pos_embed_bkg_spatial_size=cfg.HIERA_CONFIG['window_pos_embed_bkg_spatial_size'],
        window_spec=cfg.HIERA_CONFIG['window_spec'],
    )
    neck = FpnNeck(
        position_encoding=PositionEmbeddingSine(
            num_pos_feats=cfg.FPN_CONFIG['d_model'] // 2,
            normalize=True,
        ),
        d_model=cfg.FPN_CONFIG['d_model'],
        backbone_channel_list=cfg.FPN_CONFIG['backbone_channel_list'],
        fpn_top_down_levels=cfg.FPN_CONFIG['fpn_top_down_levels'],
        fpn_interp_model=cfg.FPN_CONFIG['fpn_interp_model'],
    )
    return ImageEncoder(scalp=1, trunk=trunk, neck=neck)


def _build_memory_attention() -> MemoryAttention:
    """Construct the memory attention module for video tracking."""
    d_model = cfg.FPN_CONFIG['d_model']
    layer = MemoryAttentionLayer(
        activation='relu',
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        self_attention=RoPEAttention(
            rope_theta=10000.0,
            feat_sizes=[64, 64],
            embedding_dim=d_model,
            num_heads=1,
            downsample_rate=1,
            dropout=0.1,
        ),
        d_model=d_model,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        cross_attention=RoPEAttention(
            rope_theta=10000.0,
            feat_sizes=[64, 64],
            rope_k_repeat=True,
            embedding_dim=d_model,
            num_heads=1,
            downsample_rate=1,
            dropout=0.1,
            kv_in_dim=64,
        ),
    )
    return MemoryAttention(d_model=d_model, pos_enc_at_input=True,
                           layer=layer, num_layers=4)


def _build_memory_encoder() -> MemoryEncoder:
    """Construct the memory encoder for mask propagation."""
    return MemoryEncoder(
        out_dim=64,
        position_encoding=PositionEmbeddingSine(num_pos_feats=64, normalize=True),
        mask_downsampler=MaskDownSampler(kernel_size=3, stride=2, padding=1),
        fuser=Fuser(
            layer=CXBlock(dim=256, kernel_size=7, padding=3,
                          layer_scale_init_value=1e-6, use_dwconv=True),
            num_layers=2,
        ),
    )


def build_model() -> SAM2Base:
    """Build a full SAM2Base model with the SAM 2.1 Hiera-Large architecture.

    Returns:
        SAM2Base: The constructed model with randomly initialized weights.
            Call `load_checkpoint()` to restore pretrained weights.
    """
    model = SAM2Base(
        image_encoder=_build_image_encoder(),
        memory_attention=_build_memory_attention(),
        memory_encoder=_build_memory_encoder(),
        num_maskmem=7,
        image_size=cfg.IMAGE_SIZE,
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
        compile_image_encoder=False,
    )
    return model


def load_checkpoint(
    model: SAM2Base,
    ckpt_path: str,
) -> Tuple[List[str], List[str]]:
    """Load pretrained checkpoint into model.

    Args:
        model: The SAM2Base model instance.
        ckpt_path: Path to checkpoint file (.pt or .safetensors).

    Returns:
        Tuple of (missing_keys, unexpected_keys) from ``load_state_dict``.
    """
    if ckpt_path.endswith('.safetensors'):
        ckpt = load_safetensors(ckpt_path)
    else:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        if 'model' in ckpt:
            ckpt = ckpt['model']
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    matched = len(ckpt) - len(unexpected)
    print(f'Checkpoint loaded: {matched} keys matched, '
          f'{len(missing)} missing, {len(unexpected)} unexpected')
    return missing, unexpected


def _freeze_all_except_trunk(model: SAM2Base) -> None:
    """Freeze every sub-module except ``image_encoder.trunk``.

    The trunk will later receive LoRA adapters; its base weights remain frozen.
    """
    for module_name in [
        'image_encoder', 'sam_mask_decoder', 'sam_prompt_encoder',
        'memory_attention', 'memory_encoder', 'obj_ptr_proj',
    ]:
        module = getattr(model, module_name, None)
        if module is not None:
            for param in module.parameters():
                param.requires_grad = False

    if hasattr(model, 'obj_ptr_tpos_proj'):
        for param in model.obj_ptr_tpos_proj.parameters():
            param.requires_grad = False


def _collect_lora_targets(trunk: nn.Module) -> List[str]:
    """Return names of all Linear / Conv2d submodules in the trunk."""
    targets = []
    for name, module in trunk.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            targets.append(name)
    return targets


def apply_lora(model: SAM2Base) -> SAM2Base:
    """Wrap the image encoder trunk with LoRA adapters.

    All other components are frozen.  The trunk's original parameters remain
    frozen; only the low-rank adapters are trained.

    Args:
        model: The SAM2Base model (should have pretrained weights loaded).

    Returns:
        The same model instance with LoRA applied to ``image_encoder.trunk``.
    """
    _freeze_all_except_trunk(model)

    trunk = model.image_encoder.trunk
    targets = _collect_lora_targets(trunk)

    lora_config = LoraConfig(
        r=cfg.LORA_RANK,
        lora_alpha=cfg.LORA_ALPHA,
        target_modules=targets,
        lora_dropout=cfg.LORA_DROPOUT,
        bias='none',
    )
    trunk = get_peft_model(trunk, lora_config)
    model.image_encoder.trunk = trunk

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f'LoRA rank={cfg.LORA_RANK}, alpha={cfg.LORA_ALPHA}')
    print(f'Trainable parameters: {trainable:,} / {total:,} '
          f'({100.0 * trainable / total:.2f}%)')

    return model


def dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute the Dice loss between predicted and target masks.

    Args:
        pred: Raw logits of shape ``(B, 1, H, W)``.
        target: Binary mask of shape ``(B, 1, H, W)``, values in {0, 1}.

    Returns:
        Scalar Dice loss.
    """
    pred = pred.sigmoid().flatten(1)
    target = target.flatten(1)
    intersection = (pred * target).sum(dim=-1)
    union = pred.sum(dim=-1) + target.sum(dim=-1) + 1e-8
    return (1.0 - 2.0 * intersection / union).mean()
