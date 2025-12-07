# scripts/inspect_resolution.py
import os
import sys
import argparse
import numpy as np
from pathlib import Path

os.environ["COMET_MODE"] = "disabled"
sys.path.insert(0, ".")
sys.path.insert(0, "../common")

from src.utils.parser import parser_args
from src.datasets.utils import create_dataset
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--case", required=True)
    args_local = parser.parse_args()

    # Build sys.argv for parser_args
    sys.argv = [
        sys.argv[0],
        "--config", args_local.config,
        "--case", args_local.case,
        "--num_epoch", "1",
        "--no-comet",
        "--gpu_id", "0"
    ]

    args, opt = parser_args()

    print("\n" + "=" * 70)
    print("RESOLUTION & INTRINSICS INSPECTION")
    print("=" * 70)

    # Create dataset - this returns a DataLoader in your setup
    trainset_or_loader = create_dataset(opt.dataset.train, args)

    print("\n[1] Dataset configuration:")
    print(f"  Returned type: {type(trainset_or_loader).__name__}")

    # Extract underlying dataset from DataLoader if needed
    if hasattr(trainset_or_loader, 'dataset'):
        trainset = trainset_or_loader.dataset
        print(f"  Underlying dataset: {type(trainset).__name__}")
    else:
        trainset = trainset_or_loader

    # Try to get data_dir from various sources
    data_dir = None
    if hasattr(opt.dataset.train, 'data_dir'):
        data_dir = Path(opt.dataset.train.data_dir)
    elif hasattr(trainset, 'data_dir'):
        data_dir = Path(trainset.data_dir)
    elif hasattr(trainset, 'dataset') and hasattr(trainset.dataset, 'data_dir'):
        data_dir = Path(trainset.dataset.data_dir)

    if data_dir:
        print(f"  Data directory: {data_dir}")
    else:
        print(f"  Data directory: (not found, searching...)")
        # Try to infer from case name
        data_dir = Path(f"data/{args_local.case}")
        if data_dir.exists():
            print(f"  Inferred data directory: {data_dir}")

    # Get first sample
    print("\n[2] Loading first sample...")
    try:
        if hasattr(trainset_or_loader, '__getitem__'):
            sample = trainset_or_loader[0]
        else:
            # It's a DataLoader
            sample = next(iter(trainset_or_loader))
            # DataLoader batches, so unbatch
            if isinstance(sample, dict):
                sample = {k: v[0] if torch.is_tensor(v) else v for k, v in sample.items()}

        print("  Sample keys:", list(sample.keys())[:15])

        print("\n[3] Image resolution from sample:")
        for key in ['img', 'hA', 'image', 'rgb']:
            if key in sample:
                img = sample[key]
                if torch.is_tensor(img):
                    print(f"  {key} shape: {img.shape}")
                    if img.ndim == 3:
                        print(f"    Format: [C, H, W] → Resolution: {img.shape[2]}×{img.shape[1]}")
                    elif img.ndim == 4:
                        print(f"    Format: [B, C, H, W] → Resolution: {img.shape[3]}×{img.shape[2]}")

        print("\n[4] Camera intrinsics from sample:")
        print("\n[4.5] Image size and sampling:")
        if 'img_size' in sample:
            img_size = sample['img_size']
            print(f"  img_size: {img_size}")
            if torch.is_tensor(img_size):
                print(f"    Shape: {img_size.shape}, Value: {img_size.tolist()}")

        if 'gt.rgb' in sample:
            gt_rgb = sample['gt.rgb']
            print(f"  gt.rgb shape: {gt_rgb.shape}")
            print(f"    This is the ground truth RGB for sampled rays")

        if 'uv' in sample:
            uv = sample['uv']
            print(f"  uv (sampled rays) shape: {uv.shape}")
        for key in ['K', 'intrinsics', 'cam_intr']:
            if key in sample:
                K = sample[key]
                if torch.is_tensor(K):
                    K = K.cpu().numpy()
                print(f"  {key}:\n{K}")
                if K.shape == (3, 3) or (K.ndim == 3 and K.shape[1:] == (3, 3)):
                    K_mat = K[0] if K.ndim == 3 else K
                    fx, fy = K_mat[0, 0], K_mat[1, 1]
                    cx, cy = K_mat[0, 2], K_mat[1, 2]
                    print(f"  Intrinsic parameters:")
                    print(f"    fx = {fx:.2f}")
                    print(f"    fy = {fy:.2f}")
                    print(f"    cx = {cx:.2f}")
                    print(f"    cy = {cy:.2f}")

    except Exception as e:
        print(f"  Error loading sample: {e}")
        import traceback
        traceback.print_exc()

    # Check raw data files
    if data_dir and data_dir.exists():
        print("\n[5] Raw data file inspection:")

        # Check for images
        img_dirs = [data_dir / "build" / "image", data_dir / "image", data_dir / "rgb"]
        for img_dir in img_dirs:
            if img_dir.exists():
                img_files = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))
                if img_files:
                    from PIL import Image
                    img_raw = Image.open(img_files[0])
                    print(f"  Raw image directory: {img_dir}")
                    print(f"  First image: {img_files[0].name}")
                    print(f"  Raw file resolution: {img_raw.size[0]}×{img_raw.size[1]} (W×H)")
                    break

        # Check data.npy more thoroughly
        data_npy = data_dir / "build" / "data.npy"
        if data_npy.exists():
            print(f"\n[6] Inspecting build/data.npy:")
            data = np.load(data_npy, allow_pickle=True).item()
            print(f"  Keys in data.npy: {list(data.keys())}")

            if 'img_hw' in data:
                print(f"  img_hw: {data['img_hw']}")
            if 'K' in data:
                K_file = data['K']
                print(f"  K from data.npy:\n{K_file}")
            if 'cam_intr' in data:
                print(f"  cam_intr: {data['cam_intr']}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()