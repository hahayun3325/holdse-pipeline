# File: code/scripts/evaluation/compare_checkpoints_metrics.py
"""
Compare quantitative metrics between baseline and Phase 5 checkpoints.
Modified from compare_checkpoints_metrics.py for checkpoint comparison.
"""

import torch
import numpy as np
from pathlib import Path
from loguru import logger
import json
import sys
import cv2

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import eval_modules


def load_predictions_from_render(render_dir):
    """
    Load model predictions from render output.

    Args:
        render_dir: Path to logs/xxx/test_full_render/

    Returns:
        Dictionary with predictions
    """
    render_dir = Path(render_dir)

    predictions = {}

    # Load RGB images
    rgb_dir = render_dir / 'visuals' / 'rgb'
    if rgb_dir.exists():
        rgb_files = sorted(rgb_dir.glob('*.png'))
        predictions['rgb_images'] = [cv2.imread(str(f)) for f in rgb_files]
        logger.info(f"Loaded {len(predictions['rgb_images'])} RGB images")

    # Load normal maps
    normal_dir = render_dir / 'visuals' / 'normal'
    if normal_dir.exists():
        normal_files = sorted(normal_dir.glob('*.png'))
        predictions['normal_maps'] = [cv2.imread(str(f)) for f in normal_files]
        logger.info(f"Loaded {len(predictions['normal_maps'])} normal maps")

    # Load meshes (if available)
    mesh_dir = render_dir / 'meshes'
    if mesh_dir.exists():
        hand_meshes = sorted(mesh_dir.glob('hand_*.obj'))
        obj_meshes = sorted(mesh_dir.glob('object_*.obj'))
        predictions['hand_meshes'] = [str(f) for f in hand_meshes]
        predictions['obj_meshes'] = [str(f) for f in obj_meshes]
        logger.info(f"Found {len(hand_meshes)} hand meshes, {len(obj_meshes)} object meshes")

    return predictions


def compute_image_quality_metrics(baseline_imgs, phase5_imgs):
    """
    Compute image quality metrics between baseline and Phase 5.

    Metrics:
    - PSNR: Peak Signal-to-Noise Ratio (higher is better)
    - SSIM: Structural Similarity (higher is better, 0-1)
    - MAE: Mean Absolute Error (lower is better)
    """
    from skimage.metrics import structural_similarity as ssim

    metrics = {
        'psnr': [],
        'ssim': [],
        'mae': []
    }

    for i, (base_img, p5_img) in enumerate(zip(baseline_imgs, phase5_imgs)):
        # Ensure same shape
        if base_img.shape != p5_img.shape:
            p5_img = cv2.resize(p5_img, (base_img.shape[1], base_img.shape[0]))

        # PSNR
        psnr = cv2.PSNR(base_img, p5_img)
        metrics['psnr'].append(psnr)

        # SSIM
        ssim_val = ssim(base_img, p5_img, multichannel=True, channel_axis=2, data_range=255)
        metrics['ssim'].append(ssim_val)

        # MAE
        mae = np.abs(base_img.astype(float) - p5_img.astype(float)).mean()
        metrics['mae'].append(mae)

        logger.debug(f"Frame {i}: PSNR={psnr:.2f}, SSIM={ssim_val:.4f}, MAE={mae:.2f}")

    return metrics


def compute_temporal_consistency(images):
    """
    Compute temporal consistency metrics from image sequence.

    Metrics:
    - Temporal variance: Pixel variance across time
    - Optical flow smoothness: How smooth motion is
    """
    if len(images) < 2:
        logger.warning("Need at least 2 frames for temporal metrics")
        return {}

    # Convert to grayscale for optical flow
    gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

    # Compute optical flow between consecutive frames
    flow_magnitudes = []
    flow_variations = []

    prev_flow = None
    for i in range(len(gray_images) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            gray_images[i], gray_images[i+1],
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        flow_magnitudes.append(magnitude.mean())

        if prev_flow is not None:
            variation = np.sqrt(
                (flow[..., 0] - prev_flow[..., 0])**2 +
                (flow[..., 1] - prev_flow[..., 1])**2
            ).mean()
            flow_variations.append(variation)

        prev_flow = flow

    # Compute pixel variance across time
    pixel_variance = np.stack([img.astype(float) for img in images], axis=0).var(axis=0).mean()

    metrics = {
        'avg_flow_magnitude': float(np.mean(flow_magnitudes)),
        'std_flow_magnitude': float(np.std(flow_magnitudes)),
        'avg_flow_variation': float(np.mean(flow_variations)) if flow_variations else 0.0,
        'jitter_score': float(np.std(flow_variations)) if flow_variations else 0.0,
        'pixel_variance': float(pixel_variance),
        'smoothness_score': 100.0 / (1.0 + np.std(flow_variations)) if flow_variations else 0.0
    }

    return metrics


def compare_checkpoints(baseline_dir, phase5_dir):
    """
    Compare two checkpoints comprehensively.

    Args:
        baseline_dir: Path to baseline (epoch 20) rendering
        phase5_dir: Path to Phase 5 (epoch 25) rendering

    Returns:
        Dictionary with comparison metrics
    """
    logger.info("=" * 70)
    logger.info("LOADING PREDICTIONS")
    logger.info("=" * 70)

    # Load predictions
    baseline_pred = load_predictions_from_render(baseline_dir)
    phase5_pred = load_predictions_from_render(phase5_dir)

    comparison = {
        'metadata': {
            'baseline_dir': str(baseline_dir),
            'phase5_dir': str(phase5_dir),
            'num_frames': len(baseline_pred.get('rgb_images', []))
        }
    }

    # ================================================================
    # 1. Image Quality Comparison
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("IMAGE QUALITY METRICS")
    logger.info("=" * 70)

    if 'rgb_images' in baseline_pred and 'rgb_images' in phase5_pred:
        quality_metrics = compute_image_quality_metrics(
            baseline_pred['rgb_images'],
            phase5_pred['rgb_images']
        )

        comparison['image_quality'] = {
            'psnr': {
                'mean': float(np.mean(quality_metrics['psnr'])),
                'std': float(np.std(quality_metrics['psnr'])),
                'values': quality_metrics['psnr']
            },
            'ssim': {
                'mean': float(np.mean(quality_metrics['ssim'])),
                'std': float(np.std(quality_metrics['ssim'])),
                'values': quality_metrics['ssim']
            },
            'mae': {
                'mean': float(np.mean(quality_metrics['mae'])),
                'std': float(np.std(quality_metrics['mae'])),
                'values': quality_metrics['mae']
            }
        }

        logger.info(f"PSNR (Phase 5 vs Baseline): {comparison['image_quality']['psnr']['mean']:.2f} ± {comparison['image_quality']['psnr']['std']:.2f} dB")
        logger.info(f"SSIM (Phase 5 vs Baseline): {comparison['image_quality']['ssim']['mean']:.4f} ± {comparison['image_quality']['ssim']['std']:.4f}")
        logger.info(f"MAE (Phase 5 vs Baseline): {comparison['image_quality']['mae']['mean']:.2f} ± {comparison['image_quality']['mae']['std']:.2f}")

    # ================================================================
    # 2. Temporal Consistency
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("TEMPORAL CONSISTENCY METRICS")
    logger.info("=" * 70)

    if 'rgb_images' in baseline_pred and len(baseline_pred['rgb_images']) > 1:
        baseline_temporal = compute_temporal_consistency(baseline_pred['rgb_images'])
        phase5_temporal = compute_temporal_consistency(phase5_pred['rgb_images'])

        comparison['temporal_consistency'] = {
            'baseline': baseline_temporal,
            'phase5': phase5_temporal,
            'improvement': {
                'jitter_reduction_pct': (
                    (baseline_temporal['jitter_score'] - phase5_temporal['jitter_score']) /
                    baseline_temporal['jitter_score'] * 100
                ) if baseline_temporal['jitter_score'] > 0 else 0.0,
                'smoothness_increase_pct': (
                    (phase5_temporal['smoothness_score'] - baseline_temporal['smoothness_score']) /
                    baseline_temporal['smoothness_score'] * 100
                ) if baseline_temporal['smoothness_score'] > 0 else 0.0
            }
        }

        logger.info("\nBaseline (Epoch 20):")
        logger.info(f"  Jitter score: {baseline_temporal['jitter_score']:.4f}")
        logger.info(f"  Smoothness score: {baseline_temporal['smoothness_score']:.2f}")

        logger.info("\nPhase 5 (Epoch 25):")
        logger.info(f"  Jitter score: {phase5_temporal['jitter_score']:.4f}")
        logger.info(f"  Smoothness score: {phase5_temporal['smoothness_score']:.2f}")

        logger.info("\nImprovement:")
        logger.info(f"  Jitter reduction: {comparison['temporal_consistency']['improvement']['jitter_reduction_pct']:+.2f}%")
        logger.info(f"  Smoothness increase: {comparison['temporal_consistency']['improvement']['smoothness_increase_pct']:+.2f}%")

    # ================================================================
    # 3. Summary
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 70)

    # Determine verdict
    if 'temporal_consistency' in comparison:
        jitter_reduction = comparison['temporal_consistency']['improvement']['jitter_reduction_pct']
        smoothness_increase = comparison['temporal_consistency']['improvement']['smoothness_increase_pct']

        if jitter_reduction > 10 and smoothness_increase > 5:
            verdict = "✅ SIGNIFICANT IMPROVEMENT"
        elif jitter_reduction > 5 or smoothness_increase > 3:
            verdict = "✅ MODERATE IMPROVEMENT"
        elif jitter_reduction > 0 and smoothness_increase > 0:
            verdict = "✅ SLIGHT IMPROVEMENT"
        else:
            verdict = "⚠️  NO CLEAR IMPROVEMENT"

        comparison['verdict'] = verdict
        logger.info(f"\nVerdict: {verdict}")

    return comparison


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compare baseline and Phase 5 checkpoints')
    parser.add_argument('--baseline_dir', type=str, required=True,
                        help='Path to baseline rendering (logs/xxx/test_full_render)')
    parser.add_argument('--phase5_dir', type=str, required=True,
                        help='Path to Phase 5 rendering (logs/yyy/test_full_render)')
    parser.add_argument('--output', type=str, default='../deployment/hoi4d_phase5_v1.0/checkpoint_comparison.json',
                        help='Output JSON file')
    args = parser.parse_args()

    # Run comparison
    results = compare_checkpoints(args.baseline_dir, args.phase5_dir)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✅ Comparison results saved to: {output_path}")

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Baseline checkpoint: {args.baseline_dir}")
    print(f"Phase 5 checkpoint: {args.phase5_dir}")
    print(f"Frames evaluated: {results['metadata']['num_frames']}")

    if 'image_quality' in results:
        print(f"\nImage Quality (Phase 5 vs Baseline):")
        print(f"  PSNR: {results['image_quality']['psnr']['mean']:.2f} dB")
        print(f"  SSIM: {results['image_quality']['ssim']['mean']:.4f}")

    if 'temporal_consistency' in results:
        print(f"\nTemporal Consistency:")
        print(f"  Jitter reduction: {results['temporal_consistency']['improvement']['jitter_reduction_pct']:+.2f}%")
        print(f"  Smoothness increase: {results['temporal_consistency']['improvement']['smoothness_increase_pct']:+.2f}%")

    if 'verdict' in results:
        print(f"\nOverall: {results['verdict']}")

    print("=" * 70)

'''
python scripts/evaluation/compare_checkpoints_metrics.py \
    --baseline_dir logs/ad1f0073b/test_full_render \
    --phase5_dir logs/6aaaf5002/test_full_render \
    --output ../deployment/hoi4d_phase5_v1.0/checkpoint_comparison.json
'''