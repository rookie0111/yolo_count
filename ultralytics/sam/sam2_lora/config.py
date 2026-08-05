"""Configuration for SAM 2.1 LoRA fine-tuning.

All hyperparameters and paths are centralized here for easy modification.
"""

import os
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path('/home/finn/ultralytics-main-count')
DATA_DIR = Path('/mnt/c/Users/finn.zhang/project/Datasets/sam_dataset')
WEIGHTS_DIR = Path('/home/finn/ultralytics-main-count/ultralytics/sam/weights')
OUTPUT_DIR = Path('/home/finn/ultralytics-main-count/ultralytics/sam/output_xutong_cls')
CHECKPOINT_FILE = WEIGHTS_DIR / 'sam2.1_hiera_base_plus.pt'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------------------
# Model architecture (SAM 2.1 Hiera-Base+)
# ---------------------------------------------------------------------------
IMAGE_SIZE = 1024
BACKBONE_STRIDE = 16

HIERA_CONFIG = {
    'embed_dim': 112,
    'num_heads': 2,
    'stages': [2, 3, 16, 3],
    'global_att_blocks': [23, 33, 43],
    'window_pos_embed_bkg_spatial_size': [14, 14],
    'window_spec': [8, 4, 16, 8],
}

FPN_CONFIG = {
    'backbone_channel_list': [896, 448, 224, 112],
    'd_model': 256,
    'fpn_top_down_levels': [2, 3],
    'fpn_interp_model': 'nearest',
}

# ---------------------------------------------------------------------------
# LoRA hyperparameters
# ---------------------------------------------------------------------------
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 1  # No accumulation; effective batch size = BATCH_SIZE
NUM_EPOCHS = 100
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0

# ---------------------------------------------------------------------------
# Point prompt sampling
# ---------------------------------------------------------------------------
MAX_FG_POINTS = 8     # Foreground points sampled from mask interior.
MAX_BG_POINTS = 8     # Background points sampled outside mask.
MAX_EDGE_POINTS = 4   # Points sampled near mask boundary.
MAX_POINTS = MAX_FG_POINTS + MAX_BG_POINTS + MAX_EDGE_POINTS  # Total.

# Checkpoint saving interval (epochs).
SAVE_EVERY_N_EPOCHS = 10

# Number of workers for DataLoader (0 for debugging, >0 for speed).
NUM_WORKERS = 4

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
NUM_CLASSES = 5
CLASS_MAP_FILE = DATA_DIR / 'class_map.json'
CLASS_LOSS_WEIGHT = 0.3  # Weight for classification loss vs segmentation loss
