#!/usr/bin/env python3
"""
Script to inspect batch structure from ImageDataset.

Usage:
    cd ~/Projects/holdse/code
    python ./scripts/inspect_batch.py --case hold_mug1_itw
"""

import sys
import os

# Add code directory to path
code_dir = os.path.dirname(os.path.abspath(__file__))
if code_dir.endswith('scripts'):
    code_dir = os.path.dirname(code_dir)  # Go up one level from scripts to code
sys.path.insert(0, code_dir)

import torch
import argparse
from src.datasets.image_dataset import ImageDataset

def inspect_batch(case_name):
    """Inspect batch structure from dataset."""

    class Args:
        case = case_name
        num_sample = 1024
        debug = False
        datadir = f"./data/{case_name}"

    args = Args()

    try:
        dataset = ImageDataset(args)
    except Exception as e:
        print(f"ERROR: Failed to create dataset: {e}")
        print(f"Make sure you're in the code directory and data exists at ./data/{case_name}")
        return

    print("="*70)
    print(f"Dataset: {case_name}")
    print(f"Length: {len(dataset)} samples")
    print("=" * 70)

    # Get first batch
    batch = dataset[0]

    print("\nBatch structure:")
    print("=" * 70)

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{key:30s}: shape={str(value.shape):20s}, dtype={value.dtype}")
        elif isinstance(value, dict):
            print(f"{key:30s}: dict with {len(value)} keys")
        else:
            print(f"{key:30s}: {type(value).__name__}")

    print("=" * 70)

    # Search for MANO parameters
    print("\nSearching for MANO parameters (dim 45, 48, or 62):")
    print("="*70)

    found_candidates = []
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            shape = value.shape
            if len(shape) > 0 and shape[-1] in [45, 48, 62]:
                found_candidates.append((key, shape))
                status = "✓ TARGET" if shape[-1] in [45, 48] else "⚠ COMPOSITE"
                print(f"{status}: {key:30s} shape={shape}")

    if not found_candidates:
        print("✗ No MANO parameter candidates found!")

    print("="*70)

    # Recommendation
    print("\n💡 Extraction Priority Recommendation:")
    print("="*70)
    if any('right.full_pose' in k for k, _ in found_candidates):
        print("1. ✓ Use 'right.full_pose' (48 dims)")
    if any('right.pose' in k for k, _ in found_candidates):
        print("2. ✓ Use 'right.pose' (45 dims)")
    if any('right.params' in k for k, _ in found_candidates):
        print("3. ⚠ Extract from 'right.params' (62 dims → slice [0:48])")
    print("="*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=str, default='hold_mug1_itw')
    args = parser.parse_args()

    inspect_batch(args.case)