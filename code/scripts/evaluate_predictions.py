#!/usr/bin/env python3
# scripts/evaluate_predictions.py
"""Evaluate extracted predictions against ground truth."""

import argparse
import sys
import os.path as op
from pathlib import Path

sys.path.insert(0, '.')
sys.path.insert(0, str(Path(__file__).parent.parent))  # For common module

import numpy as np
import json
import torch
from tqdm import tqdm
import src.utils.eval_modules as eval_m
import src.utils.io.gt as gt


def evaluate_predictions(pred_path, compare_path=None, output_json=None):
    """
    Evaluate extracted predictions.

    Args:
        pred_path: Path to predictions .pkl file
        compare_path: Optional path to comparison metrics .json file
        output_json: Optional custom output path for metrics
    """

    print("=" * 75)
    print("HOLD PREDICTION EVALUATION")
    print("=" * 75)

    # Load predictions
    print(f"\nLoading predictions from: {pred_path}")
    data_pred = torch.load(pred_path)

    seq_name = data_pred['full_seq_name']
    n_valid = data_pred['is_valid'].sum()
    n_total = len(data_pred['is_valid'])

    print(f"  Sequence: {seq_name}")
    print(f"  Valid frames: {n_valid}/{n_total}")

    # Check required keys
    required_keys = ['v3d_c.right', 'j3d_c.right', 'v3d_c.object',
                     'j3d_ra.right', 'root.right', 'root.object']
    missing_keys = [k for k in required_keys if k not in data_pred]

    if missing_keys:
        print(f"\n❌ ERROR: Missing required keys: {missing_keys}")
        print("   Run coordinate transformation first!")
        return None

    print("  ✓ All required keys present")

    # Load ground truth
    print(f"\nLoading ground truth...")
    try:
        data_gt = gt.load_data(seq_name)
        print(f"  ✓ GT loaded: {len(data_gt['fnames'])} frames")
    except Exception as e:
        print(f"  ❌ Failed to load GT: {e}")
        return None

    # ========== HANDLE SAMPLED PREDICTIONS ==========
    if 'sampled_frame_indices' in data_pred and data_pred['sampled_frame_indices'] is not None:
        sampled_indices = data_pred['sampled_frame_indices']
        total_frames = data_pred['total_frames_in_sequence']

        print(f"\n{'='*75}")
        print(f"SAMPLED PREDICTIONS DETECTED")
        print(f"{'='*75}")
        print(f"  Prediction frames: {len(sampled_indices)} (sampled)")
        print(f"  GT frames: {total_frames} (full sequence)")

        # Show sampling pattern
        if len(sampled_indices) <= 10:
            print(f"  Sampled indices: {sampled_indices}")
        else:
            indices_str = f"{sampled_indices[:5].tolist()}...{sampled_indices[-3:].tolist()}"
            print(f"  Sampled indices: {indices_str}")

        # Convert to numpy array if torch tensor
        if torch.is_tensor(sampled_indices):
            sampled_indices = sampled_indices.cpu().numpy()
        sampled_indices = sampled_indices.astype(int)

        # Create new dictionary with subsetted GT (xdict doesn't allow reassignment)
        print(f"\nAligning GT to sampled frames...")
        aligned_count = 0
        subsetted_gt = {}

        for key in list(data_gt.keys()):
            value = data_gt[key]

            # Check if this is an array/tensor with frame dimension
            if isinstance(value, (np.ndarray, torch.Tensor)) and len(value) > 0:
                # Check if first dimension matches total frame count
                if len(value) == total_frames:
                    # Subset this array
                    if torch.is_tensor(value):
                        subsetted_gt[key] = value[sampled_indices]
                    else:
                        subsetted_gt[key] = value[sampled_indices]
                    print(f"  ✓ {key}: {len(value)} → {len(subsetted_gt[key])} frames")
                    aligned_count += 1
                else:
                    # Keep non-frame data as is
                    subsetted_gt[key] = value
            else:
                # Keep non-array data as is
                subsetted_gt[key] = value

        # Replace data_gt with subsetted version
        data_gt = subsetted_gt

        print(f"{'='*75}")
        print(f"✅ Aligned {aligned_count} GT arrays to {len(sampled_indices)} sampled frames")
        print(f"{'='*75}\n")
    else:
        print(f"  ℹ️ Full sequence evaluation (no sampling)\n")
    # ========== END FRAME ALIGNMENT ==========

    # Evaluation functions
    eval_fn_dict = {
        "mpjpe_ra_r": eval_m.eval_mpjpe_right,
        "mrrpe_ho": eval_m.eval_mrrpe_ho_right,
        "cd_f_ra": eval_m.eval_cd_f_ra,
        "cd_f_right": eval_m.eval_cd_f_right,
        "icp": eval_m.eval_icp_first_frame,
    }

    # Run evaluation
    print(f"\nEvaluating {len(eval_fn_dict)} metrics...")
    metric_dict = {}

    for name, fn in tqdm(eval_fn_dict.items(), desc="Evaluating"):
        try:
            metric_dict = fn(data_pred, data_gt, metric_dict)
        except Exception as e:
            print(f"  ⚠️ {name} failed: {str(e)[:80]}")

    # Compute means
    mean_metrics = {}
    for metric_name, values in metric_dict.items():
        mean_value = float(np.nanmean(values))
        mean_metrics[metric_name] = mean_value

    mean_metrics = dict(sorted(mean_metrics.items()))

    # Display results
    print("\n" + "=" * 75)
    print("EVALUATION RESULTS")
    print("=" * 75)
    print(f"{'Metric':<35} {'Value':>10}")
    print("-" * 75)

    for metric_name, mean_value in mean_metrics.items():
        print(f"{metric_name.upper():<35} {mean_value:>10.2f}")

    print("=" * 75)
    print("Units: Chamfer Distance (cm²), F-score (%), MPJPE/MRRPE (mm)")
    print("=" * 75)

    # Save metrics
    if output_json is None:
        output_json = str(pred_path) + ".metric.json"

    with open(output_json, 'w') as f:
        json.dump(mean_metrics, f, indent=4)
    print(f"\n✓ Metrics saved to: {output_json}")

    # Optional comparison
    if compare_path and op.exists(compare_path):
        print("\n" + "=" * 75)
        print(f"COMPARISON: Your Results vs Reference")
        print("=" * 75)

        with open(compare_path) as f:
            ref_metrics = json.load(f)

        print(f"{'Metric':<25} {'Yours':>10} {'Reference':>10} {'Diff':>12} {'%':>10}")
        print("-" * 75)

        key_metrics = ['mpjpe_ra_r', 'mrrpe_ho', 'cd_ra', 'f10_icp', 'f10_ra', 'f5_icp']
        for key in key_metrics:
            if key in mean_metrics and key in ref_metrics:
                yours = mean_metrics[key]
                ref = ref_metrics[key]
                diff = yours - ref
                pct = (diff / ref * 100) if ref != 0 else 0
                print(f"{key:<25} {yours:>10.2f} {ref:>10.2f} {diff:>+12.2f} {pct:>+9.1f}%")

        print("=" * 75)

        # Quality assessment
        mpjpe = mean_metrics.get('mpjpe_ra_r', 999)
        print("\nQUALITY ASSESSMENT:")
        if mpjpe < 40:
            print("  ✅ MPJPE < 40mm: GOOD quality")
            print("     → Ready for Stage 2 training")
        elif mpjpe < 60:
            print("  ⚠️  MPJPE 40-60mm: ACCEPTABLE quality")
            print("     → Can proceed, but consider retraining")
        else:
            print("  ❌ MPJPE > 60mm: POOR quality")
            print("     → Retraining strongly recommended")
        print("=" * 75)

    elif compare_path:
        print(f"\n⚠️  Comparison file not found: {compare_path}")

    return mean_metrics


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate HOLD predictions against ground truth',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python scripts/evaluate_predictions.py --predictions logs/MC1_predictions.pkl

  # With comparison to HOLD official
  python scripts/evaluate_predictions.py \\
      --predictions logs/MC1_predictions.pkl \\
      --compare ~/Projects/hold/code/logs/cb20a1702/checkpoints/last.ckpt.metric.json

  # Custom output location
  python scripts/evaluate_predictions.py \\
      --predictions logs/MC1_predictions.pkl \\
      --output results/my_metrics.json
        """
    )

    parser.add_argument('--predictions', '-p', required=True,
                        help='Path to predictions .pkl file')
    parser.add_argument('--compare', '-c', default=None,
                        help='Path to comparison metrics .json file (optional)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output path for metrics .json (default: <predictions>.metric.json)')

    args = parser.parse_args()

    # Validate input
    if not op.exists(args.predictions):
        print(f"❌ Error: Predictions file not found: {args.predictions}")
        sys.exit(1)

    # Run evaluation
    metrics = evaluate_predictions(args.predictions, args.compare, args.output)

    if metrics is None:
        sys.exit(1)

    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()

'''
Examples:
  # Basic evaluation
  python scripts/evaluate_predictions.py --predictions logs/evaluation_results/MC1_stage2_e30_SDS_predictions.pkl

  # With comparison to HOLD official
  python scripts/evaluate_predictions.py --predictions logs/evaluation_results/MC1_stage2_e30_SDS_predictions.pkl --compare ~/Projects/hold/code/logs/cb20a1702/checkpoints/last.ckpt.metric.json

  # Custom output location
  python scripts/evaluate_predictions.py \\
      --predictions logs/MC1_predictions.pkl \\
      --output results/my_metrics.json
'''