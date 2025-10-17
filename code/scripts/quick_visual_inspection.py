# File: quick_visual_inspection.py
# Quick script to inspect existing training visuals

import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def quick_visual_inspection(visuals_dir='logs/training_validation/visuals'):
    """
    Quick visual quality inspection of training outputs.

    Creates a comparison grid for manual inspection.
    """
    visuals_path = Path(visuals_dir)

    if not visuals_path.exists():
        print(f"ERROR: Visuals directory not found: {visuals_dir}")
        return

    print(f"[Inspection] Analyzing visuals in: {visuals_dir}")
    print("=" * 60)

    # Find all visual types
    visual_types = sorted([d.name for d in visuals_path.iterdir() if d.is_dir()])

    print(f"\nFound {len(visual_types)} visual types:")
    for i, vtype in enumerate(visual_types, 1):
        num_images = len(list((visuals_path / vtype).glob('*.png')))
        print(f"  {i}. {vtype}: {num_images} images")

    # Load latest image from each type (from epoch 2)
    print("\nLoading latest samples (step ~2999, epoch 2)...")

    images = {}
    for vtype in visual_types:
        vtype_dir = visuals_path / vtype
        image_files = sorted(vtype_dir.glob('step_000002999*.png'))

        if len(image_files) > 0:
            img_path = image_files[0]
            images[vtype] = {
                'image': Image.open(img_path),
                'path': img_path
            }
            print(f"  ✓ {vtype}: {img_path.name}")

    if len(images) == 0:
        print("ERROR: No images found!")
        return

    # Create comparison grid
    print("\nCreating visual comparison grid...")

    # Organize by category
    categories = {
        'Full Scene': ['rgb', 'bg_rgb'],
        'Hand': ['right.fg_rgb.vis', 'right.mask_prob', 'right.normal'],
        'Object': ['object.fg_rgb.vis', 'object.mask_prob', 'object.normal'],
        'Combined': ['fg_rgb.vis', 'mask_prob', 'normal', 'imap']
    }

    # Create figure with subplots per category
    fig = plt.figure(figsize=(20, 12))

    row = 0
    for category, vtypes in categories.items():
        # Filter available types
        available = [vt for vt in vtypes if vt in images]
        if len(available) == 0:
            continue

        print(f"\n{category} ({len(available)} images):")

        for col, vtype in enumerate(available):
            idx = row * 4 + col + 1
            ax = plt.subplot(len(categories), 4, idx)

            img = images[vtype]['image']
            ax.imshow(img)
            ax.set_title(f"{category}\n{vtype}", fontsize=8)
            ax.axis('off')

            # Basic quality check
            img_array = np.array(img)
            brightness = img_array.mean()
            contrast = img_array.std()

            print(f"  - {vtype}:")
            print(f"      Brightness: {brightness:.1f}/255")
            print(f"      Contrast: {contrast:.1f}")

            # Warn if image looks problematic
            if brightness < 10 or brightness > 245:
                print(f"      ⚠️ WARNING: Unusual brightness!")
            if contrast < 20:
                print(f"      ⚠️ WARNING: Low contrast (flat image)!")

        row += 1

    plt.tight_layout()
    plt.savefig('visual_inspection_grid.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Grid saved to: visual_inspection_grid.png")

    # Generate quality checklist
    print("\n" + "=" * 60)
    print("VISUAL QUALITY CHECKLIST")
    print("=" * 60)
    print("\nPlease inspect 'visual_inspection_grid.png' and answer:")
    print("\n1. HAND QUALITY (right.fg_rgb.vis):")
    print("   [ ] Hand geometry looks smooth (no holes/spikes)")
    print("   [ ] Fingers are well-defined")
    print("   [ ] Pose looks natural")
    print("\n2. OBJECT QUALITY (object.fg_rgb.vis):")
    print("   [ ] Object shape is recognizable")
    print("   [ ] Surface is smooth")
    print("   [ ] No major artifacts")
    print("\n3. HAND-OBJECT INTERACTION (rgb):")
    print("   [ ] Hand and object don't interpenetrate badly")
    print("   [ ] Contact looks plausible")
    print("   [ ] Relative positioning makes sense")
    print("\n4. MASKS (*.mask_prob):")
    print("   [ ] Clean boundaries")
    print("   [ ] No noise/speckles")
    print("   [ ] Hand and object well separated")
    print("\n5. NORMALS (*.normal):")
    print("   [ ] Smooth gradients (no discontinuities)")
    print("   [ ] Consistent across surface")
    print("   [ ] No black/white spikes")

    print("\nSCORING GUIDE:")
    print("  - All checks passed: Score = 5 (EXCELLENT)")
    print("  - 1-2 minor issues: Score = 4 (GOOD)")
    print("  - 3-4 issues: Score = 3 (ACCEPTABLE)")
    print("  - 5+ issues: Score = 2 (POOR)")
    print("  - Major problems: Score = 1 (BROKEN)")
    print("=" * 60)


if __name__ == '__main__':
    quick_visual_inspection()