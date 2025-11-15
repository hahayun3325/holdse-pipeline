#!/usr/bin/env python
"""Validate RGB rendering from all training checkpoints"""

import torch
import sys
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
import argparse
import traceback

sys.path.insert(0, '.')
sys.path.insert(0, '../common')
os.environ['COMET_MODE'] = 'disabled'

from src.hold.hold import HOLD
from src.datasets.utils import create_dataset
from thing import thing2dev
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

def convert_batch_to_tensors(batch):
    """Convert batch data from numpy to torch tensors."""
    if isinstance(batch, dict):
        converted = {}
        for key, value in batch.items():
            if isinstance(value, np.ndarray):
                converted[key] = torch.from_numpy(value)
            elif isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
                converted[key] = torch.tensor(value, dtype=torch.long)
            elif isinstance(value, (np.float64, np.float32, np.float16)):
                converted[key] = torch.tensor(value, dtype=torch.float32)
            elif isinstance(value, dict):
                converted[key] = convert_batch_to_tensors(value)
            elif isinstance(value, list):
                converted[key] = [convert_batch_to_tensors(item) if isinstance(item, dict) else item for item in value]
            else:
                converted[key] = value
        return converted
    elif isinstance(batch, np.ndarray):
        return torch.from_numpy(batch)
    elif isinstance(batch, (np.int64, np.int32, np.int16, np.int8)):
        return torch.tensor(batch, dtype=torch.long)
    elif isinstance(batch, (np.float64, np.float32, np.float16)):
        return torch.tensor(batch, dtype=torch.float32)
    else:
        return batch

def ensure_batch_dimension(batch):
    """Ensure all tensors in batch have a batch dimension."""
    if isinstance(batch, dict):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                # Check if tensor is a scalar (0-dimensional)
                if value.ndim == 0:
                    batch[key] = value.unsqueeze(0)  # Add batch dimension
                # Check if it's missing batch dimension for specific fields
                elif key == 'idx' and value.shape == torch.Size([]):
                    batch[key] = value.unsqueeze(0)
            elif isinstance(value, dict):
                batch[key] = ensure_batch_dimension(value)
    return batch

def render_checkpoint(checkpoint_path, config_path, output_dir, frame_indices=None):
    """Render validation frames from a checkpoint."""

    if frame_indices is None:
        frame_indices = [0, 1, 2]  # Default frames
    # ✅ LOAD CHECKPOINT FIRST
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    # ✅ DETECT FRAME COUNT
    try:
        encoder_weight = ckpt['state_dict']['model.nodes.object.frame_latent_encoder.weight']
        detected_n_images = encoder_weight.shape[0]
        print(f"✓ Detected {detected_n_images} frames from checkpoint")
    except KeyError:
        print("⚠️ Could not detect frame count, using default 71")
        detected_n_images = 71

    # Load config
    opt = OmegaConf.load(config_path)
    
    if not hasattr(opt, 'model'):
        opt.model = OmegaConf.create({})
    if not hasattr(opt.model, 'scene_bounding_sphere'):
        opt.model.scene_bounding_sphere = 3.0
    
    # Setup args
    class Args:
        case = 'hold_bottle1_itw'
        n_images = 71  # Placeholder, will be overwritten
        num_sample = 2048
        infer_ckpt = checkpoint_path
        ckpt_p = checkpoint_path
        no_vis = False
        render_downsample = 2
        freeze_pose = False
        experiment = 'rgb_validation'
        log_every = 10
        log_dir = str(output_dir)
        barf_s = 0
        barf_e = 0
        no_barf = True
        shape_init = ""
        exp_key = 'rgb_validation'
        debug = False
    
    # ✅ ASSIGN DETECTED VALUE AFTER CLASS DEFINITION
    Args.n_images = detected_n_images

    args = Args()
    
    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model = HOLD(opt, args)

    # Disable all phases for pure rendering
    model.phase3_enabled = False
    model.phase4_enabled = False
    model.phase5_enabled = False
    model.ghop_enabled = False
    
    # Load checkpoint (should succeed now)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.cuda()
    model.eval()
    
    epoch = ckpt.get('epoch', -1)
    global_step = ckpt.get('global_step', -1)

    print(f"  Epoch: {epoch}, Step: {global_step}")

    # Create dataset
    val_config = opt.dataset.val if hasattr(opt.dataset, 'val') else opt.dataset.valid
    full_dataset = create_dataset(val_config, args)
    
    # Extract the underlying dataset
    if hasattr(full_dataset, 'dataset'):
        base_dataset = full_dataset.dataset
    else:
        base_dataset = full_dataset

    # Create a subset with specific frame indices
    subset_dataset = Subset(base_dataset, frame_indices)

    # Wrap in DataLoader
    subset_loader = DataLoader(subset_dataset, batch_size=1, shuffle=False)

    # Create output directories
    (output_dir / 'rgb').mkdir(parents=True, exist_ok=True)
    (output_dir / 'normal').mkdir(parents=True, exist_ok=True)

    rgb_stats = []

    print(f"Rendering {len(frame_indices)} specific frames: {frame_indices}")

    for i, batch in enumerate(tqdm(subset_loader, desc=f"Rendering epoch {epoch}")):
        actual_frame_idx = frame_indices[i]

        try:
            # Ensure all tensors have proper batch dimensions
            batch = ensure_batch_dimension(batch)

            with torch.no_grad():
                batch_cuda = thing2dev(batch, 'cuda')
                output = model.validation_step(batch_cuda)

                # Clean NaN if present
                if 'rgb' in output:
                    rgb = output['rgb']
                    if torch.isnan(rgb).any():
                        print(f"  ⚠️  Warning: NaN detected in frame {actual_frame_idx}, cleaning...")
                        rgb = torch.nan_to_num(rgb, nan=0.0)
                        output['rgb'] = rgb

                img_size = output['img_size']
                H, W = img_size[0], img_size[1]

                # Save RGB
                if 'rgb' in output:
                    rgb = output['rgb'].view(H, W, 3).cpu().numpy()

                    # Calculate statistics
                    rgb_stats.append({
                        'frame_idx': actual_frame_idx,
                        'mean': float(rgb.mean()),
                        'std': float(rgb.std()),
                        'min': float(rgb.min()),
                        'max': float(rgb.max())
                    })

                    # Save image
                    rgb_uint8 = (rgb * 255).clip(0, 255).astype(np.uint8)
                    output_path = output_dir / 'rgb' / f'frame_{actual_frame_idx:03d}.png'
                    cv2.imwrite(str(output_path), cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR))

                # Save normal map
                if 'normal' in output:
                    normal = output['normal'].view(H, W, 3).cpu().numpy()
                    normal = ((normal + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
                    normal_resized = cv2.resize(normal, (512, 512))
                    normal_path = output_dir / 'normal' / f'frame_{actual_frame_idx:03d}.png'
                    cv2.imwrite(str(normal_path), cv2.cvtColor(normal_resized, cv2.COLOR_RGB2BGR))

        except Exception as e:
            print(f"  ❌ Error rendering frame {actual_frame_idx}: {e}")
            traceback.print_exc()
            continue

    # Calculate statistics
    if rgb_stats:
        avg_stats = {
            'mean': np.mean([s['mean'] for s in rgb_stats]),
            'std': np.mean([s['std'] for s in rgb_stats]),
            'min': np.min([s['min'] for s in rgb_stats]),
            'max': np.max([s['max'] for s in rgb_stats]),
        }

        # Print per-frame stats for orientation analysis
        print(f"\nPer-Frame Statistics:")
        for stat in rgb_stats:
            print(f"  Frame {stat['frame_idx']:3d}: mean={stat['mean']:.4f}, std={stat['std']:.4f}")

        return epoch, avg_stats

    return epoch, None

def main():
    # argument parser
    parser = argparse.ArgumentParser(description='Validate RGB rendering from checkpoints')
    parser.add_argument('--frames', type=str, default='0,1,2',
                        help='Comma-separated frame indices (e.g., 50,100,150,200,250)')
    cmd_args = parser.parse_args()

    # Parse frame indices
    frame_indices = [int(x.strip()) for x in cmd_args.frames.split(',')]
    print(f"Target frames: {frame_indices}")

    # Configuration
    config_path = 'confs/ghop_stage1_rgb_only.yaml' # Stage 1 Configuration File(Stage 2 and 3 use the same. Phase 3, 4, 5 have no influence on rendering process)
    # config_path = '/home/fredcui/Projects/hold-master/code/confs/general.yaml' # HOLD Officail Configuration
    base_output_dir = Path('rgb_validation_renders')

    # All checkpoints from training session
    checkpoints = [
        # ghop_bottle_1 class
        # ('logs/78b326084/checkpoints/last.ckpt', 5),
        # ('logs/058f5c29f/checkpoints/last.ckpt', 10),
        # ('logs/a1fc80927/checkpoints/last.ckpt', 15),
        # ('logs/6c4787194/checkpoints/last.ckpt', 20),
        # ('logs/b87036a7c/checkpoints/last.ckpt', 50), # Stage 1 50 Epochs Checkpoint
        # ('logs/eece66acb/checkpoints/last.ckpt', 60), # Stage 2 Checkpoint
        # ('logs/e0ad72ec8/checkpoints/last.ckpt', 1), # Stage 3 Checkpoint
        # ('logs/61a9dc41c/checkpoints/last.ckpt', 2),
        # hold_bottle1_itw class
        ('logs/ab5edc20f/checkpoints/last.ckpt', 1), # Stage 1 Checkpoint
        # ('logs/75d213d30/checkpoints/last.ckpt', 2), # Stage 2 Checkpoint
        # ('logs/98b18938c/checkpoints/last.ckpt', 21),  # Stage 2 Checkpoint with new text prompt
        # GHOP Official Checkpoint
        # ('/home/fredcui/Projects/holdse/code/checkpoints/ghop/last.ckpt', 100),
        # HOLD Official Checkpoint(Case hold_bottle1_itw)
        # ('/home/fredcui/Projects/hold/code/logs/009c2e923/checkpoints/last.ckpt', 999),
    ]

    print("="*70)
    print("RGB RENDERING VALIDATION")
    print("="*70)
    print(f"Config: {config_path}")
    print(f"Output: {base_output_dir}/")
    print(f"Checkpoints to test: {len(checkpoints)}")
    print("="*70)
    print()

    results = {}

    for ckpt_path, expected_epoch in checkpoints:
        ckpt_path = Path(ckpt_path)

        if not ckpt_path.exists():
            print(f"⚠️  Checkpoint not found: {ckpt_path}")
            continue

        output_dir = base_output_dir / f'epoch_{expected_epoch:02d}'

        try:
            epoch, stats = render_checkpoint(ckpt_path, config_path, output_dir, frame_indices=frame_indices)

            if stats:
                results[epoch] = stats

                print(f"\nEpoch {epoch} Statistics:")
                print(f"  Mean: {stats['mean']:.4f}")
                print(f"  Std:  {stats['std']:.4f}")
                print(f"  Min:  {stats['min']:.4f}")
                print(f"  Max:  {stats['max']:.4f}")

                # Check if learned color
                if abs(stats['mean'] - 0.49) < 0.05 and stats['std'] < 0.02:
                    print(f"  ❌ GRAY (like old checkpoints - RGB not learned)")
                elif stats['std'] > 0.05:
                    print(f"  ✅ HAS COLOR VARIATION (RGB learned!)")
                else:
                    print(f"  ⚠️  BORDERLINE (needs more analysis)")

                print(f"  Saved to: {output_dir}/rgb/")

        except Exception as e:
            print(f"❌ Error rendering epoch {expected_epoch}: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    for epoch in sorted(results.keys()):
        stats = results[epoch]

        if abs(stats['mean'] - 0.49) < 0.05 and stats['std'] < 0.02:
            status = "❌ GRAY (NO COLOR)"
        elif stats['std'] > 0.05:
            status = "✅ COLOR LEARNED"
        else:
            status = "⚠️  UNCERTAIN"

        print(f"Epoch {epoch:2d}: {status} (mean={stats['mean']:.3f}, std={stats['std']:.3f})")

    print("\n" + "="*70)

    if any(s['std'] > 0.05 for s in results.values()):
        print("✅ SUCCESS: RGB loss is working! Model learned color!")
    else:
        print("❌ FAILURE: Model still producing gray images")
        print("   RGB loss configuration may need adjustment")

    print("\nVisual inspection:")
    print(f"  View renders: {base_output_dir}/")
    print(f"  Compare with: {base_output_dir}/epoch_*/rgb/frame_*.png")
    print("="*70)

if __name__ == '__main__':
    main()

'''
To use the script:
python scripts/render_validation_with_arg.py --frames 50,100,150,200,250
'''