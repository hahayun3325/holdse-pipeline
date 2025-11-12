#!/usr/bin/env python
"""Compare rendering quality between baseline and Phase 5 checkpoints"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def load_and_compare(baseline_dir, phase5_dir, output_dir):
    """
    Create side-by-side comparison of rendered images.

    Args:
        baseline_dir: Path to baseline rendering (epoch 20)
        phase5_dir: Path to Phase 5 rendering (epoch 25)
        output_dir: Output directory for comparison images
    """
    baseline_dir = Path(baseline_dir)
    phase5_dir = Path(phase5_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all RGB files
    baseline_rgb = sorted((baseline_dir / 'rgb').glob('*.png'))
    phase5_rgb = sorted((phase5_dir / 'rgb').glob('*.png'))

    print(f"Baseline images: {len(baseline_rgb)}")
    print(f"Phase 5 images: {len(phase5_rgb)}")

    # Compare each frame
    for i, (base_path, p5_path) in enumerate(zip(baseline_rgb, phase5_rgb)):
        # Load images
        base_img = cv2.imread(str(base_path))
        base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)

        p5_img = cv2.imread(str(p5_path))
        p5_img = cv2.cvtColor(p5_img, cv2.COLOR_BGR2RGB)

        # Create comparison figure
        fig = plt.figure(figsize=(15, 5))
        gs = GridSpec(1, 3, width_ratios=[1, 1, 1])

        # Baseline
        ax1 = fig.add_subplot(gs[0])
        ax1.imshow(base_img)
        ax1.set_title('Baseline (Epoch 20)\nPre-Phase 5', fontsize=12, fontweight='bold')
        ax1.axis('off')

        # Phase 5
        ax2 = fig.add_subplot(gs[1])
        ax2.imshow(p5_img)
        ax2.set_title('Phase 5 (Epoch 25)\nPost-Phase 5', fontsize=12, fontweight='bold')
        ax2.axis('off')

        # Difference map
        ax3 = fig.add_subplot(gs[2])
        diff = np.abs(base_img.astype(float) - p5_img.astype(float)).mean(axis=2)
        im = ax3.imshow(diff, cmap='hot', vmin=0, vmax=100)
        ax3.set_title(f'Difference Map\n(Avg: {diff.mean():.2f})', fontsize=12, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im, ax=ax3, fraction=0.046)

        plt.suptitle(f'Frame {i:03d} Comparison', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()

        # Save comparison
        plt.savefig(output_dir / f'comparison_{i:03d}.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Frame {i}: Difference = {diff.mean():.2f} (std: {diff.std():.2f})")

    print(f"\n✅ Saved {len(baseline_rgb)} comparison images to {output_dir}")


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print("Usage: python compare_renderings.py <baseline_dir> <phase5_dir> [output_dir]")
        sys.exit(1)

    baseline_dir = sys.argv[1]
    phase5_dir = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else '../deployment/hoi4d_phase5_v1.0/comparisons'

    load_and_compare(baseline_dir, phase5_dir, output_dir)

'''
cd ~/Projects/holdse/code

python scripts/evaluation/compare_renderings.py \
    logs/ad1f0073b/test_full_render/visuals \
    logs/6aaaf5002/test_full_render/visuals \
    ../deployment/hoi4d_phase5_v1.0/visual_comparison
'''