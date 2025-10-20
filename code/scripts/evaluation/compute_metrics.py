# File: code/scripts/evaluation/compute_metrics.py
"""Compute quantitative evaluation metrics."""

import torch
import numpy as np
from pathlib import Path
from loguru import logger
import json
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import eval_modules


def load_predictions(pred_dir):
    """Load model predictions from test directory."""
    pred_dir = Path(pred_dir)

    # Load from checkpoint or exported data
    pred_file = pred_dir / 'predictions.pth'
    if pred_file.exists():
        return torch.load(pred_file)

    # Otherwise construct from meshes
    logger.warning("predictions.pth not found, loading from meshes...")
    mesh_dir = pred_dir / 'meshes'

    # Load hand meshes
    hand_meshes = sorted(mesh_dir.glob('hand_*.obj'))
    obj_meshes = sorted(mesh_dir.glob('object_*.obj'))

    # Parse meshes (simplified - full implementation needed)
    logger.info(f"Found {len(hand_meshes)} hand meshes")
    logger.info(f"Found {len(obj_meshes)} object meshes")

    return None  # Placeholder


def compute_all_metrics(data_pred, data_gt):
    """Compute all evaluation metrics."""
    logger.info("=" * 70)
    logger.info("COMPUTING METRICS")
    logger.info("=" * 70)

    metrics = {}

    # Hand metrics
    if 'j3d_ra.right' in data_pred and 'j3d_ra.right' in data_gt:
        metrics = eval_modules.eval_mpjpe_right(data_pred, data_gt, metrics)
        logger.info(f"✅ MPJPE (hand): {np.nanmean(metrics['mpjpe_ra_r']):.2f} mm")

    # Hand-object relation
    if 'root.right' in data_pred and 'root.object' in data_pred:
        metrics = eval_modules.eval_mrrpe_ho_right(data_pred, data_gt, metrics)
        logger.info(f"✅ MRRPE (hand-obj): {np.nanmean(metrics['mrrpe_ho']):.2f} mm")

    # Object geometry
    if 'v3d_ra.object' in data_pred and 'v3d_ra.object' in data_gt:
        metrics = eval_modules.eval_cd_f_ra(data_pred, data_gt, metrics)
        logger.info(f"✅ CD (object): {np.nanmean(metrics['cd_ra']):.2f} cm²")
        logger.info(f"✅ F-score@5mm: {np.nanmean(metrics['f5_ra']):.1f}%")
        logger.info(f"✅ F-score@10mm: {np.nanmean(metrics['f10_ra']):.1f}%")

    return metrics


def compute_temporal_metrics(data_pred):
    """Compute temporal consistency metrics."""
    if 'v3d_h_c' not in data_pred:
        return {}

    verts = data_pred['v3d_h_c']  # [N, V, 3]

    # Compute velocities
    velocities = torch.norm(verts[1:] - verts[:-1], dim=-1).mean(dim=-1)

    # Compute accelerations
    accelerations = torch.abs(velocities[1:] - velocities[:-1])

    temporal_metrics = {
        'velocity_mean': velocities.mean().item(),
        'velocity_std': velocities.std().item(),
        'velocity_max': velocities.max().item(),
        'acceleration_mean': accelerations.mean().item(),
        'acceleration_std': accelerations.std().item(),
        'jitter_score': accelerations.std().item() / (velocities.mean().item() + 1e-6),
    }

    logger.info("=" * 70)
    logger.info("TEMPORAL METRICS")
    logger.info("=" * 70)
    logger.info(f"Velocity mean: {temporal_metrics['velocity_mean']:.4f} m/frame")
    logger.info(f"Velocity std: {temporal_metrics['velocity_std']:.4f}")
    logger.info(f"Jitter score: {temporal_metrics['jitter_score']:.4f}")

    return temporal_metrics


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str, required=True)
    parser.add_argument('--gt_path', type=str, required=True)
    parser.add_argument('--output', type=str, default='metrics.json')
    args = parser.parse_args()

    # Load data
    data_gt = torch.load(args.gt_path)
    data_pred = load_predictions(args.pred_dir)

    if data_pred is None:
        logger.error("Failed to load predictions")
        sys.exit(1)

    # Compute metrics
    metrics = compute_all_metrics(data_pred, data_gt)
    temporal_metrics = compute_temporal_metrics(data_pred)

    # Combine and save
    all_metrics = {
        'spatial': {k: float(np.nanmean(v)) for k, v in metrics.items()},
        'temporal': temporal_metrics,
    }

    with open(args.output, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    logger.info(f"✅ Metrics saved to: {args.output}")