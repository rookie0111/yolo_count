"""
Export SAM3 detector core to ONNX format.

The SAM3 detection pipeline involves:
  A. Vision backbone (ViT+Neck)        -> sam3_backbone.onnx (already exported)
  B. Text embeddings (CLIP tokenizer)  -> Pre-computed in Python (non-exportable)
  C. Detector core (encoder+decoder+seg) -> This script exports to ONNX

Strategy:
  1. Monkey-patch nn.MultiheadAttention to use explicit matmul+softmax
     (replaces ~37 SDPA calls that ONNX cannot handle)
  2. Freeze boxRPB coordinate cache for fixed 1008x1008 -> 72x72 feature map
  3. Pre-compute text embeddings via Python CLIP tokenizer
  4. Wrap forward_grounding() in a traceable module
  5. Export to ONNX
"""

import argparse
import math
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.models.sam.build_sam3 import build_sam3_image_model
from ultralytics.models.sam.sam3.geometry_encoders import Prompt
from ultralytics.models.sam.sam3.sam3_image import SAM3SemanticModel


# ── SDPA -> explicit matmul+softmax patch ───────────────────────────────────

_ORIGINAL_MHA_FORWARD = nn.MultiheadAttention.forward


def _patched_mha_forward(self, query, key, value, key_padding_mask=None,
                         need_weights=True, attn_mask=None,
                         average_attn_weights=True, is_causal=False):
    """Explicit matmul+softmax attention for ONNX compatibility (batch_first aware)."""
    is_batch_first = query.dim() == 3 and self.batch_first

    if is_batch_first:
        query, key, value = query.transpose(1, 0), key.transpose(1, 0), value.transpose(1, 0)

    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    head_dim = embed_dim // self.num_heads
    num_heads = self.num_heads

    w_q, w_k, w_v = self.in_proj_weight.chunk(3, dim=0)
    b_q, b_k, b_v = self.in_proj_bias.chunk(3, dim=0) if self.in_proj_bias is not None else (None, None, None)

    q = F.linear(query, w_q, b_q)
    k = F.linear(key, w_k, b_k)
    v = F.linear(value, w_v, b_v)

    q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.view(src_len, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.view(src_len, bsz * num_heads, head_dim).transpose(0, 1)

    scale = 1.0 / math.sqrt(head_dim)
    attn_weights = torch.bmm(q, k.transpose(1, 2)) * scale

    if attn_mask is not None:
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
        attn_weights = attn_weights + attn_mask

    if key_padding_mask is not None:
        expanded = key_padding_mask.unsqueeze(1).unsqueeze(2)
        expanded = expanded.repeat(1, num_heads, tgt_len, 1)
        expanded = expanded.reshape(bsz * num_heads, tgt_len, src_len)
        attn_weights = attn_weights.masked_fill(expanded, float("-inf"))

    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_output = torch.bmm(attn_weights, v)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

    if is_batch_first:
        attn_output = attn_output.transpose(1, 0)

    attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)
    return attn_output, None


def patch_mha_for_onnx():
    """Replace all nn.MultiheadAttention.forward with ONNX-compatible version."""
    nn.MultiheadAttention.forward = _patched_mha_forward


def unpatch_mha():
    """Restore original nn.MultiheadAttention.forward."""
    nn.MultiheadAttention.forward = _ORIGINAL_MHA_FORWARD


# ── BoxRPB cache freeze ─────────────────────────────────────────────────────

def freeze_rpb_cache(decoder, feat_h=72, feat_w=72):
    """Pre-compute boxRPB coordinate cache. Just clear dict cache to force
    deterministic compilable_cord_cache path during tracing."""
    decoder.compilable_cord_cache = None  # force fresh creation during first forward
    decoder.compilable_stored_size = None
    decoder.coord_cache.clear()


# ── Detector wrapper ────────────────────────────────────────────────────────

class DetectorWrapper(nn.Module):
    """Wraps SAM3SemanticModel.forward_grounding for ONNX export.

    Inputs (all from vision backbone ONNX or pre-computed text encoding):
      vision_pos_enc:  [1, 256, 72, 72]    - Position encoding (lowest FPN level)
      backbone_fpn_0:  [1, 256, 288, 288]  - FPN features level 0
      backbone_fpn_1:  [1, 256, 144, 144]  - FPN features level 1
      backbone_fpn_2:  [1, 256, 72, 72]    - FPN features level 2
      language_feats:  [nc, 32, 256]       - Pre-computed text embeddings
      language_mask:   [nc, 32]            - Text attention mask (True=padded)
      text_ids:        [nc]                - Class indices to detect (int64)
      vis_feat_size:   [2]                 - (72, 72) as int64 tensor
    """

    def __init__(self, model: SAM3SemanticModel):
        super().__init__()
        self.model = model
        self.num_feature_levels = 1

    def _dummy_geometric_prompt(self, nc, device):
        box_embeddings = torch.zeros(0, nc, 4, device=device)
        box_mask = torch.zeros(nc, 0, device=device, dtype=torch.bool)
        return Prompt(box_embeddings=box_embeddings, box_mask=box_mask)

    def _zero_prompt(self, nc, device, dtype):
        """Create zero-length prompt tensors (text-only mode, no geometric prompt).
        Bypasses geometry_encoder which has ONNX issues with zero-length sequences."""
        geo_feats = torch.zeros(0, nc, 256, device=device, dtype=dtype)
        geo_masks = torch.zeros(nc, 0, device=device, dtype=torch.bool)
        return geo_feats, geo_masks

    def _encode_prompt_wrapper(self, img_feat, img_pos_embed, vis_feat_size, nc):
        # Bypass geometry_encoder for ONNX compatibility (zero-box text-only mode)
        return self._zero_prompt(nc, img_feat.device, img_feat.dtype)

    def forward(self, vision_pos_enc, backbone_fpn_0, backbone_fpn_1, backbone_fpn_2,
                language_feats, language_mask, text_ids, vis_feat_size):
        nc = text_ids.shape[0]

        # ── Prepare backbone features ──
        backbone_fpn = [backbone_fpn_0, backbone_fpn_1, backbone_fpn_2]
        pos_enc = [vision_pos_enc, vision_pos_enc, vision_pos_enc]  # 3 copies for 3 levels

        # Expand to batch = nc
        if nc > 1:
            backbone_fpn = [feat.expand(nc, -1, -1, -1) for feat in backbone_fpn]
            pos_enc = [p.expand(nc, -1, -1, -1) for p in pos_enc]

        # Select last num_feature_levels=1, flatten to seq-first
        img_feat = backbone_fpn[-1]  # (nc, 256, 72, 72)
        img_feat_seq = img_feat.flatten(2).permute(2, 0, 1)  # (5184, nc, 256)
        img_pos_seq = pos_enc[-1].flatten(2).permute(2, 0, 1)  # (5184, nc, 256)

        # ── Encode prompt ──
        h, w = int(vis_feat_size[0].item()), int(vis_feat_size[1].item())
        prompt, prompt_mask = self._encode_prompt_wrapper(
            img_feat_seq, img_pos_seq, (h, w), nc
        )

        # ── Index + concat text features ──
        # language_feats is (seq_len, num_classes, 256) from forward_text (already seq-first)
        prompt = torch.cat([language_feats, prompt], dim=0)  # (32+0, nc, 256)
        prompt_mask = torch.cat([language_mask, prompt_mask], dim=1)  # (nc, 32+0)

        # ── Run encoder ──
        # Encoder expects seq-first format: (seq_len, batch, dim)
        encoder_out = self.model._run_encoder(
            img_feats=[img_feat_seq],
            img_pos_embeds=[img_pos_seq],
            vis_feat_sizes=[(h, w)],
            prompt=prompt,
            prompt_mask=prompt_mask,
        )

        # ── Run decoder ──
        out_dict, hs = self.model._run_decoder(
            memory=encoder_out["encoder_hidden_states"],
            pos_embed=encoder_out["pos_embed"],
            src_mask=encoder_out["padding_mask"],
            out={},
            prompt=prompt,
            prompt_mask=prompt_mask,
            encoder_out=encoder_out,
        )

        # ── Run segmentation head ──
        self.model._run_segmentation_heads(
            out=out_dict,
            backbone_out={"backbone_fpn": backbone_fpn},
            encoder_hidden_states=encoder_out["encoder_hidden_states"],
            prompt=prompt,
            prompt_mask=prompt_mask,
            hs=hs,
        )

        return (
            out_dict["pred_boxes"],        # (nc, num_queries=200, 4) cxcywh
            out_dict["pred_logits"],       # (nc, num_queries, 1)
            out_dict["pred_masks"],        # (nc, num_queries, H, W)
            out_dict["presence_logit_dec"],# (nc, num_queries)
        )


# ── Main export ──────────────────────────────────────────────────────────────

def precompute_text_embeddings(model, class_names):
    """Pre-compute text embeddings for a list of class names.

    Returns:
        language_features: (num_classes, 32, 256)
        language_mask: (num_classes, 32)
    """
    # Load CLIP tokenizer via the model's backbone
    from ultralytics.models.sam.sam3.vl_combiner import SAM3VLBackbone
    texts = model.backbone.forward_text(class_names)
    return texts["language_features"], texts["language_mask"]


def export_detector_core(model_path: str, output_path: str, class_names: list[str]):
    """Export SAM3 detector core to ONNX."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    # ── Build model ──
    model = build_sam3_image_model(model_path)
    model.eval()
    model = model.to(device=device, dtype=dtype)

    # ── Pre-compute text embeddings ──
    print(f"Pre-computing text embeddings for: {class_names}")
    lang_feats, lang_mask = precompute_text_embeddings(model, class_names)
    print(f"  language_features: {lang_feats.shape}")
    print(f"  language_mask: {lang_mask.shape}")

    # ── Patch for ONNX ──
    patch_mha_for_onnx()
    freeze_rpb_cache(model.transformer.decoder, feat_h=72, feat_w=72)
    model.transformer.encoder.compile_mode = None  # disable torch.compile
    model.transformer.decoder.compile_mode = None
    print("Patched: MHA (SDPA → matmul+softmax), boxRPB cache frozen, compile disabled")

    # ── Create wrapper ──
    wrapper = DetectorWrapper(model)
    wrapper.eval()

    # ── Dummy inputs ──
    nc = len(class_names)
    H, W = 72, 72  # feature map size for 1008x1008 input

    dummy_inputs = (
        torch.randn(1, 256, H, W, device=device, dtype=dtype),     # vision_pos_enc
        torch.randn(1, 256, 288, 288, device=device, dtype=dtype), # backbone_fpn_0
        torch.randn(1, 256, 144, 144, device=device, dtype=dtype), # backbone_fpn_1
        torch.randn(1, 256, H, W, device=device, dtype=dtype),     # backbone_fpn_2
        lang_feats.to(device=device, dtype=dtype),                 # language_features
        lang_mask.to(device=device),                                # language_mask (bool)
        torch.arange(nc, device=device, dtype=torch.int64),        # text_ids
        torch.tensor([H, W], device=device, dtype=torch.int64),    # vis_feat_size
    )

    input_names = [
        "vision_pos_enc",
        "backbone_fpn_0", "backbone_fpn_1", "backbone_fpn_2",
        "language_features", "language_mask", "text_ids", "vis_feat_size",
    ]
    output_names = ["pred_boxes", "pred_logits", "pred_masks", "presence_logit_dec"]

    dynamic_axes = {
        "vision_pos_enc": {0: "batch"},
        "backbone_fpn_0": {0: "batch"}, "backbone_fpn_1": {0: "batch"}, "backbone_fpn_2": {0: "batch"},
        "language_features": {0: "num_classes"},
        "language_mask": {0: "num_classes"},
        "text_ids": {0: "num_classes"},
        "pred_boxes": {0: "num_classes"},
        "pred_logits": {0: "num_classes"},
        "pred_masks": {0: "num_classes"},
        "presence_logit_dec": {0: "num_classes"},
    }

    # ── Verify forward first ──
    with torch.no_grad():
        out = wrapper(*dummy_inputs)
        for name, val in zip(output_names, out):
            print(f"  {name}: {val.shape}")

    # ── Export ──
    torch.onnx.export(
        wrapper,
        dummy_inputs,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )

    # ── Verify ──
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"\nONNX exported and verified: {output_path}")

    # ── Restore patches ──
    unpatch_mha()
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export SAM3 detector core to ONNX")
    parser.add_argument("--model", "-m", default="sam3.pt", help="Path to sam3.pt")
    parser.add_argument("--output", "-o", default="sam3_detector.onnx", help="Output ONNX path")
    parser.add_argument("--classes", "-c", nargs="+", default=["bag", "box"],
                        help="Class names to pre-compute text embeddings for")
    args = parser.parse_args()

    export_detector_core(args.model, args.output, args.classes)


if __name__ == "__main__":
    main()
