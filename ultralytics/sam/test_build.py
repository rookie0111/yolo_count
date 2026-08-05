#!/usr/bin/env python3
import os
import sys

os.chdir("/home/finn/ultralytics-main-count/sam2")
sys.path.insert(0, "/home/finn/ultralytics-main-count/sam2")

from hydra.core.global_hydra import GlobalHydra
GlobalHydra.instance().clear()

import sam2
print("path:", sam2.__path__)
print("file:", sam2.__file__)

from sam2.build_sam import build_sam2
model = build_sam2("sam2.1/sam2.1_hiera_b+.yaml", "checkpoints/sam2.1_hiera_base_plus.pt", device="cpu")
print("Model loaded successfully!")
