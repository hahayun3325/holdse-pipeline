import os
import sys

sys.path.insert(0, '/home/fredcui/Projects/holdse/code')

from src.datasets.image_dataset import ImageDataset
from src.datasets.utils import load_image, load_mask
import numpy as np


# Config matching your training setup
class Args:
    root = "/home/fredcui/Projects/holdse/code/data/hold_MC1_ho3d"
    split = "train"
    num_sample = 2048
    img_size = [512, 512]  # Adjust if different


args = Args()

print("=" * 60)
print("TESTING ImageDataset MASK LOADING")
print("=" * 60)

try:
    # Initialize dataset
    dataset = ImageDataset(args)
    print(f"\n1. Dataset initialized successfully")
    print(f"   Number of images: {dataset.n_images}")
    print(f"   Number of mask paths: {len(dataset.mask_paths)}")

    # Check mask paths
    print(f"\n2. First 5 mask paths:")
    for i, path in enumerate(dataset.mask_paths[:5]):
        exists = os.path.exists(path) if path else False
        print(f"   [{i}] {path} (exists: {exists})")

    # Load a sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\n3. Sample loaded successfully")
        print(f"   Keys: {list(sample.keys())}")

        if 'gt.mask' in sample:
            mask = sample['gt.mask']
            print(f"\n4. Mask statistics:")
            print(f"   Shape: {mask.shape}")
            print(f"   Dtype: {mask.dtype}")
            print(f"   Unique values: {np.unique(mask)}")
            print(f"   Foreground ratio: {(mask > 0).mean():.4f}")

        if 'gt.rgb' in sample:
            rgb = sample['gt.rgb']
            print(f"\n5. RGB statistics:")
            print(f"   Shape: {rgb.shape}")
            print(f"   Value range: [{rgb.min():.2f}, {rgb.max():.2f}]")

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback

    traceback.print_exc()