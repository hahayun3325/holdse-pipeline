#!/usr/bin/env python3
"""Debug intrinsics handling in ImageDataset"""

import sys
sys.path.insert(0, '.')

from src.datasets.image_dataset import ImageDataset
from omegaconf import OmegaConf
import torch

# Create minimal config
config = OmegaConf.create({
    'type': 'val',
    'batch_size': 1,
    'drop_last': False,
    'shuffle': False,
    'num_workers': 0,
})

# Create args
class Args:
    case = 'hold_bottle1_itw'
    n_images = 295
    num_sample = 2048
    render_downsample = 2
    no_vis = False

args = Args()

# Create dataset
dataset = ImageDataset(config, args)

# Get frame 234
sample = dataset[234]

print("=" * 70)
print("INTRINSICS DEBUGGING FOR FRAME 234")
print("=" * 70)

# Print intrinsics
if 'intrinsics' in sample:
    intrinsics = sample['intrinsics']
    print(f"\nIntrinsics shape: {intrinsics.shape}")
    print(f"Intrinsics values: {intrinsics}")
    
    if len(intrinsics) >= 4:
        fx, fy, cx, cy = intrinsics[:4]
        print(f"\nParsed intrinsics:")
        print(f"  fx (focal x): {fx:.4f}")
        print(f"  fy (focal y): {fy:.4f}")
        print(f"  cx (principal x): {cx:.4f}")
        print(f"  cy (principal y): {cy:.4f}")

# Print image size
if 'img_size' in sample:
    img_size = sample['img_size']
    print(f"\nImage size in batch: {img_size}")
    W, H = img_size[0], img_size[1]
    print(f"  Width: {W}")
    print(f"  Height: {H}")

# Check if image data exists
if 'rgb' in sample:
    print(f"\nRGB data shape: {sample['rgb'].shape}")

# Calculate expected intrinsics for different resolutions
print("\n" + "=" * 70)
print("EXPECTED INTRINSICS SCALING")
print("=" * 70)

original_W, original_H = 1000, 562
render_W, render_H = 512, 512

print(f"\nOriginal resolution: {original_W}×{original_H}")
print(f"Render resolution: {render_W}×{render_H}")

if 'intrinsics' in sample and len(sample['intrinsics']) >= 4:
    fx, fy, cx, cy = sample['intrinsics'][:4]
    
    # Compute scale factors
    scale_x = render_W / original_W
    scale_y = render_H / original_H
    
    print(f"\nScale factors:")
    print(f"  scale_x: {scale_x:.4f}")
    print(f"  scale_y: {scale_y:.4f}")
    
    # Expected scaled intrinsics
    fx_expected = fx * scale_x
    fy_expected = fy * scale_y
    cx_expected = cx * scale_x
    cy_expected = cy * scale_y
    
    print(f"\nExpected scaled intrinsics (if original):")
    print(f"  fx: {fx_expected:.4f}")
    print(f"  fy: {fy_expected:.4f}")
    print(f"  cx: {cx_expected:.4f}")
    print(f"  cy: {cy_expected:.4f}")
    
    print(f"\nActual intrinsics from dataset:")
    print(f"  fx: {fx:.4f}")
    print(f"  fy: {fy:.4f}")
    print(f"  cx: {cx:.4f}")
    print(f"  cy: {cy:.4f}")
    
    # Check if they match
    tolerance = 1.0
    fx_match = abs(fx - fx_expected) < tolerance
    fy_match = abs(fy - fy_expected) < tolerance
    cx_match = abs(cx - cx_expected) < tolerance
    cy_match = abs(cy - cy_expected) < tolerance
    
    print(f"\nScaling status:")
    print(f"  fx scaled correctly: {'✅' if fx_match else '❌'}")
    print(f"  fy scaled correctly: {'✅' if fy_match else '❌'}")
    print(f"  cx scaled correctly: {'✅' if cx_match else '❌'}")
    print(f"  cy scaled correctly: {'✅' if cy_match else '❌'}")
    
    if not all([fx_match, fy_match, cx_match, cy_match]):
        print("\n⚠️  WARNING: Intrinsics do not appear to be scaled!")
        print("   This could cause ray misprojection and rendering blur.")

print("\n" + "=" * 70)
