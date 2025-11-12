#!/usr/bin/env python
"""
Render validation dataset with proper xdict handling
"""

import torch
import sys
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2

# Add paths for imports
sys.path.insert(0, '.')
sys.path.insert(0, '../common')

# Disable Comet
os.environ['COMET_MODE'] = 'disabled'

from src.hold.hold import HOLD
from src.datasets.utils import create_dataset
from thing import thing2dev
from omegaconf import OmegaConf

def main():
    # Configuration
    config_path = 'confs/ghop_production_chunked_20251027_131408.yaml'
    checkpoint_path = 'logs/6aaaf5002/checkpoints/last.ckpt'
    output_dir = Path('logs/6aaaf5002/validation_render')

    print("="*70)
    print("VALIDATION DATASET RENDERING")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_dir}")
    print("="*70 + "\n")

    # Load config
    opt = OmegaConf.load(config_path)

    # Add missing config keys
    if not hasattr(opt, 'model'):
        opt.model = OmegaConf.create({})
    if not hasattr(opt.model, 'scene_bounding_sphere'):
        opt.model.scene_bounding_sphere = 3.0
        print("⚠️  Added missing scene_bounding_sphere=3.0")

    # Complete Args class
    class Args:
        case = 'ghop_bottle_1'
        n_images = 71
        num_sample = 2048
        infer_ckpt = checkpoint_path
        ckpt_p = checkpoint_path
        no_vis = False
        render_downsample = 2
        freeze_pose = False
        experiment = 'validation_render'
        log_every = 10
        log_dir = str(output_dir)
        barf_s = 0
        barf_e = 0
        no_barf = True
        shape_init = ""
        exp_key = 'validation_render'
        debug = False

    args = Args()
    print(f"✓ Args configured\n")

    # Create model
    print("Creating model...")
    model = HOLD(opt, args)

    # Disable all GHOP phases
    model.phase3_enabled = False
    model.phase4_enabled = False
    model.phase5_enabled = False
    model.ghop_enabled = False
    print("✓ GHOP disabled\n")

    # Load checkpoint
    print("Loading checkpoint...")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.cuda()
    model.eval()
    print("✓ Checkpoint loaded\n")

    # Create validation dataset
    print("Creating validation dataset...")
    val_config = opt.dataset.val if hasattr(opt.dataset, 'val') else opt.dataset.valid
    val_dataset = create_dataset(val_config, args)
    print(f"✓ Validation dataset: {len(val_dataset)} samples\n")

    # Create output directories
    (output_dir / 'rgb').mkdir(parents=True, exist_ok=True)
    (output_dir / 'normal').mkdir(parents=True, exist_ok=True)

    # Render loop
    print("Starting rendering...")
    print("=" * 70 + "\n")

    frame_idx = 0
    nan_count = 0

    for batch_idx, batch in enumerate(tqdm(val_dataset, desc="Rendering")):
        with torch.no_grad():
            # Move to GPU
            batch_cuda = thing2dev(batch, 'cuda')

            # Run validation step
            output = model.validation_step(batch_cuda)

            # ================================================================
            # CRITICAL FIX: Handle NaN in xdict (can't overwrite keys)
            # ================================================================
            has_nan = False
            if 'rgb' in output:
                rgb = output['rgb']
                if torch.isnan(rgb).any():
                    has_nan = True
                    nan_count += 1
                    if nan_count <= 3:
                        print(f"\n⚠️  Frame {frame_idx}: RGB has NaN, replacing with zeros")

                    # Replace NaN values in-place (don't reassign to output)
                    rgb_cleaned = torch.nan_to_num(rgb, nan=0.0)

                    # Delete old key and add new one (xdict doesn't allow overwrite)
                    dict.__delitem__(output, 'rgb')
                    dict.__setitem__(output, 'rgb', rgb_cleaned)
            # ================================================================

            # Extract and save images
            img_size = output['img_size']
            H, W = img_size[0], img_size[1]

            # RGB
            if 'rgb' in output:
                rgb = output['rgb'].view(H, W, 3).cpu().numpy()
                rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
                rgb_resized = cv2.resize(rgb, (256, 256))
                cv2.imwrite(str(output_dir / 'rgb' / f'{frame_idx:05d}.png'),
                           cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2BGR))

            # Normal
            if 'normal' in output:
                normal = output['normal'].view(H, W, 3).cpu().numpy()
                normal = ((normal + 1) * 127.5).clip(0, 255).astype(np.uint8)
                normal_resized = cv2.resize(normal, (256, 256))
                cv2.imwrite(str(output_dir / 'normal' / f'{frame_idx:05d}.png'),
                           cv2.cvtColor(normal_resized, cv2.COLOR_RGB2BGR))

            frame_idx += 1

            # Diagnostic output for first frame
            if batch_idx == 0:
                print(f"\nFirst frame statistics:")
                if 'rgb' in output:
                    rgb_val = output['rgb']
                    print(f"  RGB: min={rgb_val.min().item():.4f}, "
                          f"max={rgb_val.max().item():.4f}, "
                          f"mean={rgb_val.mean().item():.4f}")
                    print(f"  Has NaN: {has_nan}")
                print()

    print(f"\n{'=' * 70}")
    print(f"✅ Rendering complete!")
    print(f"   Frames rendered: {frame_idx}")
    print(f"   Frames with NaN: {nan_count} ({100*nan_count/max(frame_idx,1):.1f}%)")
    print(f"   Output: {output_dir}")
    print(f"{'='*70}")

    # Final verification
    if frame_idx > 0:
        first_img = cv2.imread(str(output_dir / 'rgb' / '00000.png'))
        if first_img is not None:
            print(f"\nFirst image verification:")
            print(f"  Mean: {first_img.mean():.2f}")
            print(f"  Max: {first_img.max()}")
            print(f"  Non-zero: {np.count_nonzero(first_img)} / {first_img.size}")

            if first_img.mean() > 1:
                print(f"\n  ✅ SUCCESS! Validation rendering produced visible content!")
                print(f"  ✅ Model CAN render despite NaN issues!")
                print(f"\n  Next steps:")
                print(f"  1. Investigate why forward pass produces NaN")
                print(f"  2. Fix NaN source (likely in rendering network)")
                print(f"  3. Re-test rendering without NaN replacement")
            else:
                print(f"\n  ❌ Still black even with validation dataset and NaN replacement")
                print(f"\n  Conclusion: Model fundamentally broken")
                print(f"  All frames have NaN → model never learned to render")
                print(f"\n  Possible root causes:")
                print(f"  1. Training diverged early (check training logs)")
                print(f"  2. Learning rate too high")
                print(f"  3. Architectural issue in forward pass")
                print(f"  4. Data preprocessing issue")

if __name__ == '__main__':
    main()