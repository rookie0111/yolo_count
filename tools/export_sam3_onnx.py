"""
Export SAM3 vision backbone to ONNX format.

SAM3 uses complex-valued RoPE (Rotary Position Embedding) which is not
supported by ONNX. This script patches the RoPE to use real-valued
operations before exporting, making it compatible with ONNX opset 17+.
"""

import argparse
import types

import torch

from ultralytics.models.sam.build_sam3 import build_sam3_image_model
from ultralytics.models.sam.modules.utils import apply_rotary_enc


def apply_rotary_enc_real(xq, xk, freqs_cos, freqs_sin):
    """Real-valued rotary position encoding for ONNX compatibility.
    
    Replaces the complex-number approach with:
      (x1 + i*x2) * (cos + i*sin) = (x1*cos - x2*sin) + i*(x1*sin + x2*cos)
    """
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)

    xq_out = torch.stack([
        xq_[..., 0] * freqs_cos - xq_[..., 1] * freqs_sin,
        xq_[..., 0] * freqs_sin + xq_[..., 1] * freqs_cos,
    ], dim=-1).flatten(-2)

    xk_out = torch.stack([
        xk_[..., 0] * freqs_cos - xk_[..., 1] * freqs_sin,
        xk_[..., 0] * freqs_sin + xk_[..., 1] * freqs_cos,
    ], dim=-1).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def _patched_apply_rope(self, q, k):
    """Patched version of _apply_rope using real-valued cos/sin."""
    return apply_rotary_enc_real(q, k,
        freqs_cos=self.freqs_cis_cos.unsqueeze(0).unsqueeze(0).to(q.device),
        freqs_sin=self.freqs_cis_sin.unsqueeze(0).unsqueeze(0).to(q.device))


def patch_rope_for_onnx(module):
    """Replace complex freqs_cis with separate cos/sin tensors."""
    if hasattr(module, "freqs_cis") and module.freqs_cis is not None:
        freqs_cis = module.freqs_cis
        if freqs_cis.is_complex():
            module.freqs_cis_cos = freqs_cis.real.clone()
            module.freqs_cis_sin = freqs_cis.imag.clone()
            module.freqs_cis = None
            module._apply_rope = types.MethodType(_patched_apply_rope, module)
            return True
    return False


class BackboneWrapper(torch.nn.Module):
    """Wraps SAM3 backbone to return flat tensor tuple for ONNX export."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        out = self.backbone.forward_image(x)
        return (
            out["vision_features"],
            out["vision_pos_enc"][0],
            out["vision_pos_enc"][1],
            out["vision_pos_enc"][2],
            out["backbone_fpn"][0],
            out["backbone_fpn"][1],
            out["backbone_fpn"][2],
            out["sam2_backbone_out"]["backbone_fpn"],
            out["sam2_backbone_out"]["vision_pos_enc"],
        )


OUTPUT_NAMES = [
    "vision_features",
    "vision_pos_enc_0", "vision_pos_enc_1", "vision_pos_enc_2",
    "backbone_fpn_0", "backbone_fpn_1", "backbone_fpn_2",
    "sam2_backbone_fpn", "sam2_vision_pos_enc",
]


def export_vision_backbone(model_path: str, output_path: str, half: bool = False):
    """Export SAM3 vision backbone (forward_image) to ONNX."""
    dtype = torch.float16 if half else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_sam3_image_model(model_path)
    model.eval()
    model = model.to(device=device, dtype=dtype)

    # Patch RoPE to avoid complex numbers
    trunk = model.backbone.vision_backbone.trunk
    patched = sum(patch_rope_for_onnx(blk.attn) for blk in trunk.blocks)
    print(f"Patched {patched} attention blocks for ONNX compatibility")

    # Input must be 1008x1008 to match ViT window config (patch=14, 72 tokens)
    imgsz = 1008
    dummy_input = torch.randn(1, 3, imgsz, imgsz, device=device, dtype=dtype)

    # Verify forward works after patching
    with torch.no_grad():
        features = model.backbone.forward_image(dummy_input)

    print(f"Backbone output keys: {list(features.keys())}")
    for k, v in features.items():
        if isinstance(v, dict):
            print(f"  {k}: dict with keys {list(v.keys())}")
        elif isinstance(v, list):
            print(f"  {k}: list[{len(v)}], first shape={v[0].shape}")
        else:
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

    wrapper = BackboneWrapper(model.backbone)
    wrapper.eval()

    dynamic_axes = {"image": {0: "batch"}}
    for name in OUTPUT_NAMES:
        dynamic_axes[name] = {0: "batch"}

    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        input_names=["image"],
        output_names=OUTPUT_NAMES,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )

    # Verify exported model
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"\nONNX exported and verified: {output_path}")
    print(f"Model size: {sum(p.numel() for p in wrapper.parameters()) / 1e6:.1f}M params")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export SAM3 vision backbone to ONNX")
    parser.add_argument("--model", "-m", default="sam3.pt", help="Path to sam3.pt")
    parser.add_argument("--output", "-o", default="sam3_backbone.onnx", help="Output ONNX path")
    parser.add_argument("--half", action="store_true", help="Export FP16")
    args = parser.parse_args()

    export_vision_backbone(args.model, args.output, args.half)


if __name__ == "__main__":
    main()
