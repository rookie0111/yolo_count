#!/usr/bin/env python3
"""
Step 1: Pre-compute SAM3 text embeddings and save to .npz file.

Usage:  python precompute_text_embed.py
Output: sam3_text_emb.npz
"""
import numpy as np
import torch
from ultralytics.models.sam.build_sam3 import build_sam3_image_model

TEXT_PROMPT = "filled with fertilizer bags"
SAM3_PT_PATH = "/home/finn/ultralytics-main-count/sam3.pt"
OUTPUT_PATH  = "/home/finn/ultralytics-main-count/ultralytics/labels/sam3_text_emb.npz"

print("Loading SAM3 model for text embedding...")
model = build_sam3_image_model(SAM3_PT_PATH)
model.eval().cuda()

with torch.no_grad():
    model.set_classes([TEXT_PROMPT])

feats = model.text_embeddings["language_features"].detach().cpu().numpy()
mask  = model.text_embeddings["language_mask"].detach().cpu().numpy()

np.savez_compressed(OUTPUT_PATH, language_features=feats, language_mask=mask)
print(f"Saved: {OUTPUT_PATH}")
print(f"  features: {feats.shape} dtype={feats.dtype}")
print(f"  mask:     {mask.shape} dtype={mask.dtype}")

del model
torch.cuda.empty_cache()
print("Done.")
