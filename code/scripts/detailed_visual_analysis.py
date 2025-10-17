# File: code/scripts/detailed_visual_analysis.py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import json


def analyze_visual_grid(visuals_dir='logs/training_validation/visuals'):
    """Detailed quality analysis with scoring"""

    visuals_path = Path(visuals_dir)
    results = {'scores': {}, 'warnings': [], 'recommendations': []}

    # Define quality criteria per visual type
    criteria = {
        'rgb': {'brightness_range': (100, 200), 'min_contrast': 40},
        'right.fg_rgb.vis': {'brightness_range': (150, 240), 'min_contrast': 25},
        'object.fg_rgb.vis': {'brightness_range': (150, 240), 'min_contrast': 30},
        'right.mask_prob': {'brightness_range': (30, 150), 'min_contrast': 80},
        'object.mask_prob': {'brightness_range': (15, 100), 'min_contrast': 60},
        'right.normal': {'brightness_range': (100, 150), 'min_contrast': 18},
        'object.normal': {'brightness_range': (100, 150), 'min_contrast': 18},
        'normal': {'brightness_range': (80, 130), 'min_contrast': 45}
    }

    print("=" * 70)
    print("DETAILED VISUAL QUALITY ANALYSIS")
    print("=" * 70)

    for visual_type, thresholds in criteria.items():
        vtype_dir = visuals_path / visual_type
        if not vtype_dir.exists():
            continue

        # Load all epochs for this type
        images = sorted(vtype_dir.glob('*.png'))
        scores = []

        print(f"\n{visual_type}:")
        print("-" * 70)

        for img_path in images:
            img = np.array(Image.open(img_path))

            # Extract step number
            step = int(img_path.stem.split('_')[1])
            epoch = step // 1000

            # Compute metrics
            brightness = img.mean()
            contrast = img.std()

            # Score this image (0-10 scale)
            score = 10.0

            # Check brightness
            b_min, b_max = thresholds['brightness_range']
            if brightness < b_min or brightness > b_max:
                penalty = min(abs(brightness - b_min), abs(brightness - b_max)) / 50
                score -= min(penalty, 3.0)

            # Check contrast
            if contrast < thresholds['min_contrast']:
                penalty = (thresholds['min_contrast'] - contrast) / 10
                score -= min(penalty, 3.0)

            scores.append(score)

            status = "✓" if score >= 7.0 else "⚠️" if score >= 5.0 else "❌"
            print(f"  Epoch {epoch} (step {step:05d}): {status} Score={score:.1f}/10 "
                  f"(B={brightness:.1f}, C={contrast:.1f})")

            if score < 7.0:
                results['warnings'].append(
                    f"{visual_type} epoch {epoch}: Score {score:.1f}/10"
                )

        # Overall score for this type
        avg_score = np.mean(scores)
        results['scores'][visual_type] = avg_score

        print(f"  → Average: {avg_score:.1f}/10")

    # Calculate overall quality score
    overall_score = np.mean(list(results['scores'].values()))

    print("\n" + "=" * 70)
    print(f"OVERALL QUALITY SCORE: {overall_score:.1f}/10")
    print("=" * 70)

    # Generate recommendations
    if overall_score >= 8.0:
        recommendation = "EXCELLENT - Ready for production"
        next_step = "Proceed to GHOP multi-object testing"
    elif overall_score >= 6.5:
        recommendation = "GOOD - Minor improvements possible"
        next_step = "Proceed to testing OR apply optional refinements"
    elif overall_score >= 5.0:
        recommendation = "ACCEPTABLE - Some issues present"
        next_step = "Identify bottlenecks, apply targeted fixes"
    else:
        recommendation = "POOR - Critical issues need fixing"
        next_step = "Debug low-scoring components before proceeding"

    results['overall_score'] = overall_score
    results['recommendation'] = recommendation
    results['next_step'] = next_step

    print(f"\nRecommendation: {recommendation}")
    print(f"Next Step: {next_step}")

    # Save results
    with open('visual_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: visual_analysis_results.json")

    return results


if __name__ == '__main__':
    analyze_visual_grid()