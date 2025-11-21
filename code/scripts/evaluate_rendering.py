#!/usr/bin/env python3
"""
Evaluate neural rendering quality using PSNR, SSIM, and LPIPS metrics.

Usage:
    python evaluate_rendering.py \
        --render_dir rgb_validation_renders/epoch_23/rgb \
        --gt_dir data/hold_bottle1_itw/images \
        --output_csv results/stage2_metrics.csv \
        --compare_dir rgb_validation_renders/epoch_10/rgb  # Optional: for Stage 1 vs Stage 2
"""

import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def load_image(path, resize=None):
    """Load image and convert to float32 in [0, 1] range."""
    img = Image.open(path).convert('RGB')
    if resize:
        img = img.resize(resize, Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


def compute_psnr(img1, img2):
    """Compute PSNR between two images."""
    return psnr(img1, img2, data_range=1.0)


def compute_ssim(img1, img2):
    """Compute SSIM between two images."""
    return ssim(img1, img2, multichannel=True, data_range=1.0, channel_axis=2)


def compute_lpips(img1, img2, lpips_fn):
    """Compute LPIPS between two images."""
    # Convert to tensor: [H, W, 3] -> [1, 3, H, W], range [-1, 1]
    img1_t = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    img2_t = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0) * 2 - 1

    # Move tensors to the same device as the LPIPS model
    device = next(lpips_fn.parameters()).device
    img1_t = img1_t.to(device)
    img2_t = img2_t.to(device)

    with torch.no_grad():
        result = lpips_fn(img1_t, img2_t).item()

    # Clear GPU cache if using CUDA
    if device.type == 'cuda':
        del img1_t, img2_t
        torch.cuda.empty_cache()

    return result

def get_frame_pairs(render_dir, gt_dir, max_frames=None):
    """
    Match rendered frames with ground truth frames.

    Returns list of tuples: (render_path, gt_path, frame_number)
    """
    render_dir = Path(render_dir)
    gt_dir = Path(gt_dir)

    # Get all rendered images
    render_files = sorted(render_dir.glob("*.png"))

    pairs = []
    for render_path in render_files:
        # Extract frame number from filename (e.g., "frame_100.png" -> 100)
        frame_name = render_path.stem
        frame_num = int(frame_name.split('_')[-1])

        # Find corresponding ground truth image
        # Try multiple naming conventions
        gt_candidates = [
            gt_dir / f"frame_{frame_num:06d}.png",
            gt_dir / f"frame_{frame_num:04d}.png",
            gt_dir / f"frame_{frame_num}.png",
            gt_dir / f"{frame_num:06d}.png",
            gt_dir / f"{frame_num:04d}.png",
        ]

        gt_path = None
        for candidate in gt_candidates:
            if candidate.exists():
                gt_path = candidate
                break

        if gt_path:
            pairs.append((str(render_path), str(gt_path), frame_num))
        else:
            print(f"Warning: No ground truth found for {render_path.name}")

    if max_frames:
        pairs = pairs[:max_frames]

    return pairs


def evaluate_directory(render_dir, gt_dir, lpips_fn, max_frames=None, resize=None, auto_resize=True):
    """
    Evaluate all frames in a directory.
    If auto_resize=True, automatically resizes GT to match render dimensions.
    """
    pairs = get_frame_pairs(render_dir, gt_dir, max_frames)

    if not pairs:
        raise ValueError(f"No valid frame pairs found between {render_dir} and {gt_dir}")

    print(f"Found {len(pairs)} frame pairs to evaluate")

    results = []
    for render_path, gt_path, frame_num in tqdm(pairs, desc="Evaluating frames"):
        # Load rendered image to get size
        render_img_pil = Image.open(render_path).convert('RGB')
        render_w, render_h = render_img_pil.size

        # Load GT and check size
        gt_img_pil = Image.open(gt_path).convert('RGB')
        gt_w, gt_h = gt_img_pil.size

        # Auto-resize if needed
        if auto_resize and (render_w, render_h) != (gt_w, gt_h):
            gt_img_pil = gt_img_pil.resize((render_w, render_h), Image.BICUBIC)

        # Convert to numpy
        render_img = np.array(render_img_pil, dtype=np.float32) / 255.0
        gt_img = np.array(gt_img_pil, dtype=np.float32) / 255.0

        # Apply manual resize if specified
        if resize:
            render_img = np.array(Image.fromarray((render_img * 255).astype(np.uint8)).resize(resize, Image.BICUBIC), dtype=np.float32) / 255.0
            gt_img = np.array(Image.fromarray((gt_img * 255).astype(np.uint8)).resize(resize, Image.BICUBIC), dtype=np.float32) / 255.0

        # Compute metrics
        psnr_val = compute_psnr(render_img, gt_img)
        ssim_val = compute_ssim(render_img, gt_img)
        lpips_val = compute_lpips(render_img, gt_img, lpips_fn)

        results.append({
            'frame': frame_num,
            'render_path': render_path,
            'gt_path': gt_path,
            'psnr': psnr_val,
            'ssim': ssim_val,
            'lpips': lpips_val
        })

    return pd.DataFrame(results)


def print_summary(df, stage_name="Stage"):
    """Print summary statistics with quality assessment."""
    print(f"\n{'=' * 70}")
    print(f"{stage_name} Rendering Quality Metrics")
    print(f"{'=' * 70}")
    print(f"Frames evaluated: {len(df)}")
    print(f"\nPSNR (Peak Signal-to-Noise Ratio):")
    print(f"  Mean:   {df['psnr'].mean():.2f} dB")
    print(f"  Std:    {df['psnr'].std():.2f} dB")
    print(f"  Min:    {df['psnr'].min():.2f} dB")
    print(f"  Max:    {df['psnr'].max():.2f} dB")

    # Quality assessment for PSNR
    mean_psnr = df['psnr'].mean()
    if mean_psnr >= 30:
        psnr_quality = "Excellent ✅"
    elif mean_psnr >= 25:
        psnr_quality = "Good ✅"
    elif mean_psnr >= 20:
        psnr_quality = "Acceptable ⚠️"
    else:
        psnr_quality = "Poor ❌"
    print(f"  Quality: {psnr_quality}")

    print(f"\nSSIM (Structural Similarity Index):")
    print(f"  Mean:   {df['ssim'].mean():.4f}")
    print(f"  Std:    {df['ssim'].std():.4f}")
    print(f"  Min:    {df['ssim'].min():.4f}")
    print(f"  Max:    {df['ssim'].max():.4f}")

    # Quality assessment for SSIM
    mean_ssim = df['ssim'].mean()
    if mean_ssim >= 0.90:
        ssim_quality = "Excellent ✅"
    elif mean_ssim >= 0.80:
        ssim_quality = "Good ✅"
    elif mean_ssim >= 0.70:
        ssim_quality = "Acceptable ⚠️"
    else:
        ssim_quality = "Poor ❌"
    print(f"  Quality: {ssim_quality}")

    print(f"\nLPIPS (Learned Perceptual Similarity):")
    print(f"  Mean:   {df['lpips'].mean():.4f}")
    print(f"  Std:    {df['lpips'].std():.4f}")
    print(f"  Min:    {df['lpips'].min():.4f}")
    print(f"  Max:    {df['lpips'].max():.4f}")

    # Quality assessment for LPIPS
    mean_lpips = df['lpips'].mean()
    if mean_lpips <= 0.05:
        lpips_quality = "Excellent ✅"
    elif mean_lpips <= 0.10:
        lpips_quality = "Good ✅"
    elif mean_lpips <= 0.20:
        lpips_quality = "Acceptable ⚠️"
    else:
        lpips_quality = "Poor ❌"
    print(f"  Quality: {lpips_quality}")

    print(f"\n{'=' * 70}")

    # Overall assessment
    print(f"\nOVERALL ASSESSMENT:")
    if mean_psnr >= 25 and mean_ssim >= 0.80 and mean_lpips <= 0.10:
        print("  ✅ High Quality - Ready for publication/deployment")
    elif mean_psnr >= 20 and mean_ssim >= 0.70 and mean_lpips <= 0.20:
        print("  ⚠️  Acceptable Quality - Suitable for Stage 3 training")
    else:
        print("  ❌ Low Quality - Consider re-training with adjusted hyperparameters")
    print(f"{'=' * 70}\n")


def compare_stages(df1, df2, stage1_name="Stage 1", stage2_name="Stage 2"):
    """Compare metrics between two stages."""
    print(f"\n{'=' * 70}")
    print(f"Comparison: {stage1_name} vs {stage2_name}")
    print(f"{'=' * 70}")

    metrics = ['psnr', 'ssim', 'lpips']
    better = {metric: 'higher' for metric in ['psnr', 'ssim']}
    better['lpips'] = 'lower'

    for metric in metrics:
        mean1 = df1[metric].mean()
        mean2 = df2[metric].mean()
        diff = mean2 - mean1
        pct_change = (diff / mean1) * 100

        if (better[metric] == 'higher' and diff > 0) or (better[metric] == 'lower' and diff < 0):
            symbol = "✅ Better"
        elif abs(pct_change) < 2:
            symbol = "≈ Similar"
        else:
            symbol = "⚠️  Worse"

        print(f"\n{metric.upper()}:")
        print(f"  {stage1_name}: {mean1:.4f}")
        print(f"  {stage2_name}: {mean2:.4f}")
        print(f"  Change: {diff:+.4f} ({pct_change:+.1f}%) {symbol}")

    print(f"\n{'=' * 70}\n")


def evaluate_single_pair(render_path, gt_path, lpips_fn, resize=None):
    """
    Evaluate a single pair of images.
    Automatically handles size mismatch by resizing GT to match render.
    """
    print(f"Evaluating single pair:")
    print(f"  Rendered: {render_path}")
    print(f"  Ground Truth: {gt_path}")

    # Load rendered image first (no resize)
    render_img = load_image(render_path, resize=None)
    render_h, render_w = render_img.shape[:2]

    # Load GT image
    gt_img_raw = Image.open(gt_path).convert('RGB')
    gt_h, gt_w = gt_img_raw.size[1], gt_img_raw.size[0]

    # Check if sizes match
    if (render_h, render_w) != (gt_h, gt_w):
        print(f"  Size mismatch detected:")
        print(f"    Rendered: {render_w}x{render_h}")
        print(f"    Ground Truth: {gt_w}x{gt_h}")
        print(f"  Resizing ground truth to match rendered image...")

        # Resize GT to match render size
        gt_img_raw = gt_img_raw.resize((render_w, render_h), Image.BICUBIC)

    gt_img = np.array(gt_img_raw, dtype=np.float32) / 255.0

    # Apply manual resize if specified
    if resize:
        print(f"  Applying manual resize to {resize[0]}x{resize[1]}...")
        render_img = np.array(Image.fromarray((render_img * 255).astype(np.uint8)).resize(resize, Image.BICUBIC), dtype=np.float32) / 255.0
        gt_img = np.array(Image.fromarray((gt_img * 255).astype(np.uint8)).resize(resize, Image.BICUBIC), dtype=np.float32) / 255.0

    # Compute metrics
    psnr_val = compute_psnr(render_img, gt_img)
    ssim_val = compute_ssim(render_img, gt_img)
    lpips_val = compute_lpips(render_img, gt_img, lpips_fn)

    return {
        'render_path': render_path,
        'gt_path': gt_path,
        'render_size': f"{render_w}x{render_h}",
        'gt_size': f"{gt_w}x{gt_h}",
        'psnr': psnr_val,
        'ssim': ssim_val,
        'lpips': lpips_val
    }


def print_single_result(result):
    """Print results for a single image pair."""
    print(f"\n{'='*70}")
    print(f"Single Image Comparison Results")
    print(f"{'='*70}")
    print(f"\nPSNR: {result['psnr']:.2f} dB")
    print(f"SSIM: {result['ssim']:.4f}")
    print(f"LPIPS: {result['lpips']:.4f}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate neural rendering quality")

    # Add mutually exclusive group for directory vs single image mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--render_dir', help="Directory with rendered images")
    mode_group.add_argument('--render_img', help="Single rendered image path")

    # Ground truth can be either directory or single image
    gt_group = parser.add_mutually_exclusive_group(required=True)
    gt_group.add_argument('--gt_dir', help="Directory with ground truth images")
    gt_group.add_argument('--gt_img', help="Single ground truth image path")

    parser.add_argument('--output_csv', default=None, help="Output CSV file for per-frame metrics")
    parser.add_argument('--compare_dir', default=None, help="Optional: Directory to compare against")
    parser.add_argument('--max_frames', type=int, default=None, help="Maximum frames to evaluate")
    parser.add_argument('--resize', type=int, nargs=2, default=None, help="Resize images to [width, height]")
    parser.add_argument('--stage_name', default="Stage 2", help="Name for this stage in output")
    parser.add_argument('--compare_name', default="Stage 1", help="Name for comparison stage")
    parser.add_argument('--no-auto-resize', action='store_true',
                        help="Disable automatic resizing of GT to match render size")

    args = parser.parse_args()

    # Initialize LPIPS
    print("Loading LPIPS model...")
    lpips_fn = lpips.LPIPS(net='alex')
    if torch.cuda.is_available():
        lpips_fn = lpips_fn.cuda()
        print("Using GPU for LPIPS")
    else:
        print("Using CPU for LPIPS")

    # Single image mode
    if args.render_img and args.gt_img:
        result = evaluate_single_pair(
            args.render_img,
            args.gt_img,
            lpips_fn,
            tuple(args.resize) if args.resize else None
        )
        print_single_result(result)

        if args.output_csv:
            df = pd.DataFrame([result])
            df.to_csv(args.output_csv, index=False)
            print(f"Results saved to {args.output_csv}")

        return

    # Evaluate main directory
    print(f"\nEvaluating {args.stage_name}...")
    df_main = evaluate_directory(
        args.render_dir,
        args.gt_dir,
        lpips_fn,
        args.max_frames,
        tuple(args.resize) if args.resize else None
    )

    # Print summary
    print_summary(df_main, args.stage_name)

    # Save results
    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
        df_main.to_csv(args.output_csv, index=False)
        print(f"Per-frame metrics saved to {args.output_csv}")

    # Optional: Compare with another stage
    if args.compare_dir:
        print(f"\nEvaluating {args.compare_name} for comparison...")
        df_compare = evaluate_directory(
            args.compare_dir,
            args.gt_dir,
            lpips_fn,
            args.max_frames,
            tuple(args.resize) if args.resize else None
        )

        print_summary(df_compare, args.compare_name)
        compare_stages(df_compare, df_main, args.compare_name, args.stage_name)

        if args.output_csv:
            compare_csv = args.output_csv.replace('.csv', '_comparison.csv')
            comparison_df = pd.DataFrame({
                'frame': df_main['frame'],
                f'{args.compare_name}_psnr': df_compare['psnr'],
                f'{args.stage_name}_psnr': df_main['psnr'],
                f'{args.compare_name}_ssim': df_compare['ssim'],
                f'{args.stage_name}_ssim': df_main['ssim'],
                f'{args.compare_name}_lpips': df_compare['lpips'],
                f'{args.stage_name}_lpips': df_main['lpips'],
            })
            comparison_df.to_csv(compare_csv, index=False)
            print(f"Comparison metrics saved to {compare_csv}")


if __name__ == "__main__":
    main()

'''
### Compare Two Specific Images

```bash
python evaluate_rendering.py \
    --render_img rgb_validation_renders/epoch_23/rgb/frame_100.png \
    --gt_img data/hold_bottle1_itw/images/frame_000100.png
```

### With Output CSV

```bash
python evaluate_rendering.py \
    --render_img rgb_validation_renders/epoch_23/rgb/frame_100.png \
    --gt_img data/hold_bottle1_itw/images/frame_000100.png \
    --output_csv results/frame_100_metrics.csv
```

### With Image Resizing

```bash
python evaluate_rendering.py \
    --render_img rgb_validation_renders/epoch_23/rgb/frame_100.png \
    --gt_img data/hold_bottle1_itw/images/frame_000100.png \
    --resize 512 512
```
'''