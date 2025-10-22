# File: code/assess_render_quality.py
"""Quick automated quality assessment of rendered frames."""

import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

def assess_quality(render_dir):
    """Assess rendering quality across all frames."""
    render_dir = Path(render_dir)

    print("="*70)
    print("RENDERING QUALITY ASSESSMENT")
    print("="*70)

    results = defaultdict(dict)

    # Check RGB images
    rgb_dir = render_dir / 'rgb'
    if rgb_dir.exists():
        rgb_files = sorted(rgb_dir.glob('*.png'))
        print(f"\n📊 RGB Images: {len(rgb_files)} frames")

        # Sample analysis
        brightness_values = []
        for img_path in rgb_files[::10]:  # Every 10th frame
            img = cv2.imread(str(img_path))
            brightness = img.mean()
            brightness_values.append(brightness)

        mean_brightness = np.mean(brightness_values)
        std_brightness = np.std(brightness_values)

        print(f"  Mean brightness: {mean_brightness:.1f}")
        print(f"  Std brightness: {std_brightness:.1f}")

        if mean_brightness < 10:
            print(f"  ❌ FAIL: Images too dark (likely render failure)")
            results['rgb']['quality'] = 'FAIL'
        elif mean_brightness > 245:
            print(f"  ⚠️  WARN: Images overexposed")
            results['rgb']['quality'] = 'WARN'
        elif std_brightness < 5:
            print(f"  ⚠️  WARN: Very uniform brightness (no variation)")
            results['rgb']['quality'] = 'WARN'
        else:
            print(f"  ✅ PASS: Reasonable brightness range")
            results['rgb']['quality'] = 'PASS'

    # Check normal maps
    normal_dir = render_dir / 'normal'
    if normal_dir.exists():
        normal_files = sorted(normal_dir.glob('*.png'))
        print(f"\n📊 Normal Maps: {len(normal_files)} frames")

        # Check variance
        variances = []
        for img_path in normal_files[::10]:
            img = cv2.imread(str(img_path))
            variance = img.std()
            variances.append(variance)

        mean_variance = np.mean(variances)
        print(f"  Mean variance: {mean_variance:.1f}")

        if mean_variance < 10:
            print(f"  ❌ FAIL: Very low variance (likely uniform background)")
            results['normal']['quality'] = 'FAIL'
        else:
            print(f"  ✅ PASS: Sufficient detail present")
            results['normal']['quality'] = 'PASS'

    # Check instance maps
    imap_dir = render_dir / 'imap'
    if imap_dir.exists():
        imap_files = sorted(imap_dir.glob('*.png'))
        print(f"\n📊 Instance Maps: {len(imap_files)} frames")

        # Check number of unique colors
        unique_colors_list = []
        for img_path in imap_files[::10]:
            img = cv2.imread(str(img_path))
            unique_colors = len(np.unique(img.reshape(-1, 3), axis=0))
            unique_colors_list.append(unique_colors)

        mean_colors = np.mean(unique_colors_list)
        print(f"  Mean unique colors: {mean_colors:.1f}")

        if mean_colors < 3:
            print(f"  ❌ FAIL: Binary segmentation only (need 3+ classes)")
            results['imap']['quality'] = 'FAIL'
        elif mean_colors >= 3:
            print(f"  ✅ PASS: Multi-class segmentation present")
            results['imap']['quality'] = 'PASS'

    # Overall summary
    print("\n" + "="*70)
    print("OVERALL ASSESSMENT")
    print("="*70)

    passed = sum(1 for r in results.values() if r.get('quality') == 'PASS')
    failed = sum(1 for r in results.values() if r.get('quality') == 'FAIL')
    warned = sum(1 for r in results.values() if r.get('quality') == 'WARN')

    print(f"✅ Passed: {passed}")
    print(f"⚠️  Warnings: {warned}")
    print(f"❌ Failed: {failed}")

    if failed == 0 and passed >= 2:
        print(f"\n✅ Overall: ACCEPTABLE - Proceed with metrics computation")
        return True
    else:
        print(f"\n❌ Overall: NEEDS IMPROVEMENT - Review quality issues")
        return False

if __name__ == '__main__':
    import sys
    render_dir = sys.argv[1] if len(sys.argv) > 1 else 'logs/b2e4b039a/test_full_render/visuals'
    assess_quality(render_dir)
