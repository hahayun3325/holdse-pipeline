import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

data_root = "/home/fredcui/Projects/holdse/code/data/hold_MC1_ho3d"

print("=" * 60)
print("INSPECTING MASK LOADING")
print("=" * 60)

# 1. Check mask directory structure
print("\n1. Mask directory structure:")
build_mask_dir = os.path.join(data_root, "build", "mask")
sam_mask_dir = os.path.join(data_root, "sam", "object", "masks_processed")

print(f"   build/mask exists: {os.path.exists(build_mask_dir)}")
print(f"   sam/object/masks_processed exists: {os.path.exists(sam_mask_dir)}")

# 2. List mask files
if os.path.exists(build_mask_dir):
    build_masks = sorted(glob.glob(f"{build_mask_dir}/*.png"))
    print(f"\n2. build/mask contains {len(build_masks)} files")
    if len(build_masks) > 0:
        print(f"   First 5: {[os.path.basename(p) for p in build_masks[:5]]}")

if os.path.exists(sam_mask_dir):
    sam_masks = sorted(glob.glob(f"{sam_mask_dir}/*.png"))
    print(f"\n3. sam/object/masks_processed contains {len(sam_masks)} files")
    if len(sam_masks) > 0:
        print(f"   First 5: {[os.path.basename(p) for p in sam_masks[:5]]}")


# 3. Load and inspect actual mask values
def inspect_mask_values(mask_path, name):
    print(f"\n4. Inspecting {name}: {os.path.basename(mask_path)}")
    try:
        mask = np.array(Image.open(mask_path))
        print(f"   Shape: {mask.shape}")
        print(f"   Dtype: {mask.dtype}")
        print(f"   Unique values: {np.unique(mask)}")
        print(f"   Min: {mask.min()}, Max: {mask.max()}")
        print(f"   Mean: {mask.mean():.4f}")

        # Check if binary (0/1) or multi-class
        unique = np.unique(mask)
        if len(unique) == 2:
            print(f"   ✓ Binary mask detected")
            print(f"   Foreground pixel ratio: {(mask > 0).mean():.4f}")
        elif len(unique) > 2:
            print(f"   ⚠ Multi-class mask detected - check class semantics!")
            for val in unique:
                print(f"      Value {val}: {(mask == val).sum()} pixels")
        else:
            print(f"   ✗ Single value mask - no segmentation!")

        return mask
    except Exception as e:
        print(f"   ERROR loading mask: {e}")
        return None


# Inspect sample masks
if os.path.exists(build_mask_dir) and len(build_masks) > 0:
    build_mask = inspect_mask_values(build_masks[0], "build/mask")

if os.path.exists(sam_mask_dir) and len(sam_masks) > 0:
    sam_mask = inspect_mask_values(sam_masks[0], "sam/object/masks_processed")

# 4. Visualize comparison
if 'build_mask' in dir() and 'sam_mask' in dir() and build_mask is not None and sam_mask is not None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(build_mask, cmap='gray')
    axes[0].set_title("build/mask")
    axes[0].axis('off')

    axes[1].imshow(sam_mask, cmap='gray')
    axes[1].set_title("sam/object/masks_processed")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig("mask_comparison.png", dpi=150)
    print(f"\n5. Saved mask comparison to mask_comparison.png")