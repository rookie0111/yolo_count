"""Training entry-point for SAM 2.1 LoRA fine-tuning with classification.

Usage:
    python -m sam2_lora.train
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sam2_lora.config as cfg
from sam2_lora.dataset import SegmentationDataset, FlatSegmentationDataset, ClassificationSegmentationDataset, InstanceSegmentationDataset, BboxPointPromptDataset, pad_collate
from sam2_lora.model import (
    apply_lora,
    build_model,
    dice_loss,
    load_checkpoint,
)


class ClassHead(nn.Module):
    """Classification head for defect type prediction."""
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_epoch(
    model: torch.nn.Module,
    class_head: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    num_epochs: int,
) -> float:
    """Run one training epoch over the DataLoader.

    Uses gradient accumulation when ``cfg.GRADIENT_ACCUMULATION_STEPS > 1``
    to simulate a larger effective batch size without exceeding GPU memory.

    Args:
        model: The SAM2Base model with LoRA adapters (in train mode).
        class_head: Classification head module.
        loader: DataLoader providing batches from ``SegmentationDataset``.
        optimizer: AdamW optimizer (only LoRA parameters).
        epoch: Current epoch number (0-indexed, used for logging).
        num_epochs: Total epochs (used for logging).

    Returns:
        Average loss over the epoch.
    """
    model.train()
    class_head.train()
    total_loss = 0.0
    total_seg_loss = 0.0
    total_cls_loss = 0.0
    device = cfg.DEVICE
    accum_steps = cfg.GRADIENT_ACCUMULATION_STEPS
    cls_criterion = nn.BCEWithLogitsLoss()

    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        image = batch['image'].to(device)
        mask_gt = batch['mask'].to(device)
        points = batch['points'].to(device)
        labels = batch['labels'].to(device)
        class_label = batch['class_label'].to(device)

        for retry in range(3):
            try:
                # Forward through image encoder (with LoRA adapters active).
                backbone_out = model.forward_image(image)
                image_embeddings = backbone_out['vision_features']
                high_res_features = [
                    backbone_out['backbone_fpn'][0],
                    backbone_out['backbone_fpn'][1],
                ]

                # Encode point prompts.
                sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
                    points=(points, labels), boxes=None, masks=None,
                )
                image_pe = model.sam_prompt_encoder.get_dense_pe()

                # Decode mask.
                low_res_multimasks, _, _, _ = model.sam_mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=image_pe,
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    repeat_image=False,
                    high_res_features=high_res_features,
                )

                # Upsample to full resolution and compute segmentation loss.
                pred = F.interpolate(
                    low_res_multimasks,
                    size=(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
                    mode='bilinear',
                    align_corners=False,
                )
                loss_bce = F.binary_cross_entropy_with_logits(pred, mask_gt)
                loss_dice = dice_loss(pred, mask_gt)
                seg_loss = (loss_bce + loss_dice) / accum_steps

                # Classification loss.
                cls_logits = class_head(image_embeddings.float())
                cls_loss = cls_criterion(cls_logits, class_label) / accum_steps

                # Combined loss.
                loss = seg_loss + cfg.CLASS_LOSS_WEIGHT * cls_loss

                loss.backward()
                total_loss += loss.item() * accum_steps
                total_seg_loss += seg_loss.item() * accum_steps
                total_cls_loss += cls_loss.item() * accum_steps

                if (step + 1) % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.image_encoder.trunk.parameters(),
                        cfg.GRAD_CLIP_NORM,
                    )
                    optimizer.step()
                    optimizer.zero_grad()

                if step % 50 == 0:
                    print(
                        f'  Epoch [{epoch + 1}/{num_epochs}] Step {step}: '
                        f'loss={loss.item() * accum_steps:.4f}  '
                        f'(seg={seg_loss.item() * accum_steps:.4f}, '
                        f'cls={cls_loss.item() * accum_steps:.4f})'
                    )
                break  # Success, exit retry loop.

            except RuntimeError as exc:
                if 'CUDA' in str(exc) and retry < 2:
                    print(
                        f'  [WARN] CUDA error at step {step}, '
                        f'retry {retry + 1}/3: {exc}'
                    )
                    optimizer.zero_grad()
                    try:
                        torch.cuda.empty_cache()
                    except RuntimeError:
                        pass
                else:
                    raise

    return total_loss / len(loader)


def validate(
    model: torch.nn.Module,
    loader: DataLoader,
) -> float:
    """Evaluate the model on the validation/test set.

    Args:
        model: The SAM2Base model (in eval mode).
        loader: DataLoader for the test subset.

    Returns:
        Average loss over the validation set.
    """
    model.eval()
    total_loss = 0.0
    device = cfg.DEVICE

    with torch.no_grad():
        for batch in loader:
            image = batch['image'].to(device)
            mask_gt = batch['mask'].to(device)
            points = batch['points'].to(device)
            labels = batch['labels'].to(device)

            backbone_out = model.forward_image(image)
            image_embeddings = backbone_out['vision_features']
            high_res_features = [
                backbone_out['backbone_fpn'][0],
                backbone_out['backbone_fpn'][1],
            ]

            sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
                points=(points, labels), boxes=None, masks=None,
            )
            image_pe = model.sam_prompt_encoder.get_dense_pe()

            low_res_multimasks, _, _, _ = model.sam_mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res_features,
            )

            pred = F.interpolate(
                low_res_multimasks,
                size=(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
                mode='bilinear',
                align_corners=False,
            )
            loss_bce = F.binary_cross_entropy_with_logits(pred, mask_gt)
            loss_dice = dice_loss(pred, mask_gt)
            total_loss += (loss_bce + loss_dice).item()

    return total_loss / len(loader)


def save_checkpoint(
    model: torch.nn.Module,
    class_head: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
    epoch: int,
    loss: float,
    best_loss: float,
) -> None:
    """Save LoRA adapter weights, class head, optimizer state, and training metadata.

    Args:
        model: The SAM2Base model (LoRA weights extracted from trunk).
        class_head: The classification head.
        optimizer: The optimizer (state saved for resume).
        path: Directory where checkpoint will be saved.
        epoch: Current epoch number.
        loss: Loss value at this checkpoint.
        best_loss: Best loss seen so far.
    """
    os.makedirs(path, exist_ok=True)
    lora_weights = model.image_encoder.trunk.state_dict()
    final_path = os.path.join(path, 'checkpoint.pt')
    tmp_path = final_path + '.tmp'
    torch.save({
        'epoch': epoch,
        'loss': loss,
        'best_loss': best_loss,
        'lora_weights': lora_weights,
        'class_head': class_head.state_dict(),
        'num_classes': cfg.NUM_CLASSES,
        'optimizer_state': optimizer.state_dict(),
    }, tmp_path)
    os.replace(tmp_path, final_path)
    print(f'  Checkpoint saved: {path}  (epoch={epoch}, loss={loss:.4f})')


def _find_resume_epoch() -> int:
    """Find the latest completed epoch from saved checkpoints.

    Returns:
        The epoch number to resume from (0 if no checkpoint found).
    """
    best_dir = os.path.join(cfg.OUTPUT_DIR, 'best')
    best_ckpt = os.path.join(best_dir, 'checkpoint.pt')
    if os.path.isfile(best_ckpt):
        state = torch.load(best_ckpt, map_location='cpu')
        return state.get('epoch', 0)
    return 0


def main() -> None:
    """Orchestrate the full training pipeline with crash-resilient resume."""
    print(f'Device: {cfg.DEVICE}')
    if cfg.DEVICE.type == 'cuda':
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision('high')
        props = torch.cuda.get_device_properties(0)
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
        print(f'  VRAM: {props.total_memory / 1e9:.1f} GB')

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print('Building model ...')
    model = build_model()

    print('Loading pretrained weights ...')
    load_checkpoint(model, str(cfg.CHECKPOINT_FILE))

    print('Applying LoRA ...')
    model = apply_lora(model)
    model = model.to(cfg.DEVICE)

    # Classification head
    class_head = ClassHead(in_channels=256, num_classes=cfg.NUM_CLASSES).to(cfg.DEVICE)
    print(f'Classification head: {cfg.NUM_CLASSES} classes')

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print('Loading dataset ...')
    split_file = str(cfg.DATA_DIR / 'train.txt')
    class_map_path = str(cfg.CLASS_MAP_FILE)
    train_dataset = BboxPointPromptDataset(
        str(cfg.DATA_DIR), split_file=split_file, class_map_path=class_map_path
    )
    # Use same dataset for test (no separate test split)
    test_dataset = train_dataset
    print(f'  Train samples: {len(train_dataset)}')
    print(f'  Test  samples: {len(test_dataset)}')

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=pad_collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=pad_collate,
    )

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        list(model.image_encoder.trunk.parameters()) + list(class_head.parameters()),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    # ------------------------------------------------------------------
    # Resume from checkpoint
    # ------------------------------------------------------------------
    start_epoch = 0
    best_loss = float('inf')
    resume_path = os.path.join(cfg.OUTPUT_DIR, 'best', 'checkpoint.pt')

    if os.path.isfile(resume_path):
        try:
            state = torch.load(resume_path, map_location='cpu')
            model.image_encoder.trunk.load_state_dict(state['lora_weights'])
            if 'optimizer_state' in state:
                optimizer.load_state_dict(state['optimizer_state'])
            start_epoch = state.get('epoch', 0)
            best_loss = state.get('best_loss', state.get('loss', float('inf')))
            print(f'Resumed from epoch {start_epoch}, best_loss={best_loss:.4f}')
        except Exception as exc:
            print(f'Warning: Could not load checkpoint ({exc}). Starting fresh.')
            os.remove(resume_path)

    # ------------------------------------------------------------------
    # Training loop (crash-resilient)
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, cfg.NUM_EPOCHS):
        try:
            train_loss = train_epoch(
                model, class_head, train_loader, optimizer, epoch, cfg.NUM_EPOCHS,
            )
            print(
                f'Epoch [{epoch + 1}/{cfg.NUM_EPOCHS}]  '
                f'train_loss={train_loss:.4f}'
            )

            test_loss = validate(model, test_loader)
            print(f'  test_loss={test_loss:.4f}')

            # Save checkpoint every epoch.
            if test_loss < best_loss:
                best_loss = test_loss
            save_checkpoint(
                model, class_head, optimizer,
                os.path.join(cfg.OUTPUT_DIR, 'best'),
                epoch + 1, test_loss, best_loss,
            )

        except RuntimeError as exc:
            if 'CUDA' in str(exc):
                print(
                    f'\n[CATCH] CUDA error at epoch {epoch + 1}, '
                    f'saving emergency checkpoint ...'
                )
                try:
                    save_checkpoint(
                        model, class_head, optimizer,
                        os.path.join(cfg.OUTPUT_DIR, 'best'),
                        epoch, float('nan'), best_loss,
                    )
                except Exception:
                    print('  Emergency save failed.')
                print('  Restart training to resume from last checkpoint.')
                return
            else:
                raise

    print(f'Training complete.  Best test loss: {best_loss:.4f}')


if __name__ == '__main__':
    main()
