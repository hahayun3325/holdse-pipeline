# File: code/scripts/evaluation/quick_visual_check.py

import cv2
import numpy as np
from pathlib import Path
from loguru import logger


def check_visual_quality(visual_dir):
    """Quick automated quality check."""
    visual_dir = Path(visual_dir)

    issues = []

    # Check RGB images
    rgb_dir = visual_dir / 'rgb'
    if rgb_dir.exists():
        rgb_files = sorted(rgb_dir.glob('*.png'))
        logger.info(f"Found {len(rgb_files)} RGB images")

        # Sample first image
        if rgb_files:
            img = cv2.imread(str(rgb_files[0]))

            # Check for black images (failed render)
            if img.mean() < 10:
                issues.append("⚠️ RGB images are very dark (potential render failure)")

            # Check for all-white (overexposed)
            if img.mean() > 245:
                issues.append("⚠️ RGB images are overexposed")

            logger.info(f"✅ RGB mean brightness: {img.mean():.1f}")
    else:
        issues.append("❌ No RGB directory found")

    # Check meshes
    mesh_dir = visual_dir.parent / 'meshes'
    if mesh_dir.exists():
        mesh_files = sorted(mesh_dir.glob('*.obj'))
        logger.info(f"Found {len(mesh_files)} mesh files")

        if len(mesh_files) == 0:
            issues.append("❌ No meshes exported")
    else:
        issues.append("❌ No mesh directory found")

    # Summary
    logger.info("\n" + "=" * 70)
    if len(issues) == 0:
        logger.info("✅ Visual quality check passed!")
    else:
        logger.warning(f"⚠️ Found {len(issues)} potential issues:")
        for issue in issues:
            logger.warning(f"  {issue}")
    logger.info("=" * 70)

    return issues


if __name__ == '__main__':
    import sys

    visual_dir = sys.argv[1] if len(sys.argv) > 1 else 'logs/b2e4b039a/test/visuals'
    check_visual_quality(visual_dir)