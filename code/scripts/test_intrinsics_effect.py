#!/usr/bin/env python3
"""Test effect of intrinsics on rendering"""

import sys
sys.path.insert(0, '.')

import torch
from src.hold.hold import HOLD
from omegaconf import OmegaConf

# Load Stage 2 checkpoint
config = OmegaConf.load('confs/ghop_stage1_rgb_only.yaml')

class Args:
    case = 'hold_bottle1_itw'
    n_images = 295
    num_sample = 2048
    infer_ckpt = 'logs/stage2_final.ckpt'
    ckpt_p = 'logs/stage2_final.ckpt'
    no_vis = False
    render_downsample = 2
    freeze_pose = False

args = Args()

# Load model
model = HOLD(config, args)
ckpt = torch.load('logs/stage2_final.ckpt', map_location='cpu')
model.load_state_dict(ckpt['state_dict'], strict=False)

print("Model loaded successfully")
print("\nChecking how model uses intrinsics...")

# Try to access intrinsics-related attributes
if hasattr(model, 'intrinsics'):
    print(f"Model has 'intrinsics' attribute: {model.intrinsics}")

# Check dataset's intrinsics
from src.datasets.utils import create_dataset
val_config = config.dataset.valid
val_dataset = create_dataset(val_config, args)

sample = val_dataset[0]
if 'intrinsics' in sample:
    print(f"\nDataset intrinsics shape: {sample['intrinsics'].shape}")
    print(f"Dataset intrinsics values: {sample['intrinsics']}")
else:
    print("\n⚠️  No 'intrinsics' key in dataset sample!")

print("\nTo check if intrinsics affect rendering, modify intrinsics.txt")
print("and re-render to see if output changes.")
