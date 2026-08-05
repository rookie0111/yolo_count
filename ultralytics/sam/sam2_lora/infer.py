"""Inference with a trained SAM 2.1 LoRA model.

Loads a checkpoint, runs inference on single images, and visualises the
predicted masks.
"""

import argparse
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import sam2_lora.config as cfg
from sam2_lora.model import apply_lora, build_model, load_checkpoint


def load_trained_model(
    checkpoint_dir: str,
) -> torch.nn.Module:
    """Load a trained model with LoRA adapters.

    Args:
        checkpoint_dir: Directory containing ``checkpoint.pt`` (saved by
            ``train.py``).

    Returns:
        Model in evaluation mode on the correct device.
    """
    ckpt_path = os.path.join(checkpoint_dir, 'checkpoint.pt')
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    model = build_model()
    load_checkpoint(model, str(cfg.CHECKPOINT_FILE))
    model = apply_lora(model)

    state = torch.load(ckpt_path, map_location='cpu')
    model.image_encoder.trunk.load_state_dict(state['lora_weights'])
    print(f'LoRA weights restored from epoch {state["epoch"]} '
          f'(loss={state["loss"]:.4f})')

    model = model.to(cfg.DEVICE)
    model.eval()
    return model


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    image_path: str,
    point_coords: Optional[torch.Tensor] = None,
    point_labels: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Predict a binary mask for a single image.

    Args:
        model: The SAM2Base model in eval mode.
        image_path: Path to the input image.
        point_coords: Optional ``(1, N, 2)`` tensor of foreground point
            coordinates in ``(x, y)`` pixel space.  If ``None``, a single
            center point is used.
        point_labels: Optional ``(1, N)`` tensor of point labels (``1`` =
            foreground, ``0`` = background, ``-1`` = padding).  If ``None``,
            all points are treated as foreground.

    Returns:
        Tuple of ``(mask, iou_prediction)`` where:
        - ``mask`` is a ``(1, H, W)`` float tensor with values in ``[0, 1]``.
        - ``iou_prediction`` is a scalar tensor estimating IoU.
    """
    image = Image.open(image_path).convert('RGB')
    image = image.resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), Image.BILINEAR)
    img_tensor = (
        torch.from_numpy(
            np.array(image)
        ).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    ).to(cfg.DEVICE)

    # Default: single centre point.
    if point_coords is None:
        point_coords = torch.tensor(
            [[[cfg.IMAGE_SIZE // 2, cfg.IMAGE_SIZE // 2]]],
            dtype=torch.float, device=cfg.DEVICE,
        )
        point_labels = torch.ones(1, 1, dtype=torch.long, device=cfg.DEVICE)

    backbone_out = model.forward_image(img_tensor)
    image_emb = backbone_out['vision_features']
    high_res = [
        backbone_out['backbone_fpn'][0],
        backbone_out['backbone_fpn'][1],
    ]

    sparse_emb, dense_emb = model.sam_prompt_encoder(
        points=(point_coords, point_labels),
        boxes=None,
        masks=None,
    )
    image_pe = model.sam_prompt_encoder.get_dense_pe()

    low_res, ious, _, _ = model.sam_mask_decoder(
        image_embeddings=image_emb,
        image_pe=image_pe,
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=False,
        repeat_image=False,
        high_res_features=high_res,
    )

    mask = F.interpolate(
        low_res,
        size=(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
        mode='bilinear',
        align_corners=False,
    ).sigmoid().squeeze(0)  # (1, H, W)

    return mask, ious.squeeze()


def main() -> None:
    """Command-line inference entry point."""
    parser = argparse.ArgumentParser(
        description='SAM 2.1 LoRA inference on CGFSDS-9',
    )
    parser.add_argument(
        'checkpoint_dir', type=str,
        help='Directory containing checkpoint.pt',
    )
    parser.add_argument(
        'image_path', type=str,
        help='Path to input image',
    )
    parser.add_argument(
        '--output', '-o', type=str, default=None,
        help='Output path for predicted mask (default: print IoU only)',
    )
    args = parser.parse_args()

    model = load_trained_model(args.checkpoint_dir)

    mask, iou = predict(model, args.image_path)
    print(f'Predicted IoU: {iou.item():.4f}')

    if args.output:
        mask_np = (mask.squeeze().cpu().numpy() * 255).astype('uint8')
        Image.fromarray(mask_np).save(args.output)
        print(f'Mask saved: {args.output}')


if __name__ == '__main__':
    main()
