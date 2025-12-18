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

def render_checkpoint(checkpoint_path, config_path, output_dir, frame_indices=None, downsample=1):
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
        case = 'hold_MC1_ho3d' # hold_GPMF12_ho3d
        n_images = 71  # Placeholder, will be overwritten
        num_sample = 2048
        infer_ckpt = checkpoint_path
        ckpt_p = checkpoint_path
        no_vis = False
        render_downsample = downsample
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
        # NEW: Extract the actual frame index from batch metadata
        if 'idx' in batch:
            actual_frame_idx = int(batch['idx'].item() if torch.is_tensor(batch['idx']) else batch['idx'])
            print(f"  Dataset position {frame_indices[i]} → Actual frame {actual_frame_idx}")
        elif 'frame_id' in batch:
            actual_frame_idx = int(batch['frame_id'].item() if torch.is_tensor(batch['frame_id']) else batch['frame_id'])
            print(f"  Dataset position {frame_indices[i]} → Actual frame {actual_frame_idx}")
        else:
            actual_frame_idx = frame_indices[i]
            print(f"  Warning: No 'idx' or 'frame_id' found in batch, using input index {actual_frame_idx}")

        try:
            # Ensure all tensors have proper batch dimensions
            batch = ensure_batch_dimension(batch)

            with torch.no_grad():
                batch_cuda = thing2dev(batch, 'cuda')
                # ============== ADD THIS DEBUG BLOCK ==============
                print(f"\n[Render Debug] Frame {actual_frame_idx}")

                # Check camera parameters
                if 'extrinsics' in batch_cuda:
                    extr = batch_cuda['extrinsics'][0]  # [4, 4]
                    camera_pos = extr[:3, 3]
                    print(f"  Camera position: {camera_pos.cpu().numpy()}")

                if 'intrinsics' in batch_cuda:
                    intr = batch_cuda['intrinsics'][0]  # [4, 4]
                    fx = intr[0, 0]
                    cx = intr[0, 2]
                    cy = intr[1, 2]
                    print(f"  Camera intrinsics: fx={fx:.1f}, cx={cx:.1f}, cy={cy:.1f}")

                # Check if model has object SDF
                if hasattr(model.model.nodes, 'object'):
                    obj_node = model.model.nodes['object']
                    print(f"  Object node exists: {type(obj_node).__name__}")

                    try:
                        with torch.no_grad():
                            # Get frame index from batch
                            frame_idx = batch_cuda.get('idx', torch.tensor([0], device='cuda'))
                            if frame_idx.dim() == 0:
                                frame_idx = frame_idx.unsqueeze(0)

                            print(f"  Querying object SDF for frame {frame_idx.item()}")

                            # Sample test points
                            test_pts = torch.tensor([
                                [0., 0., 0.],
                                [0.1, 0., 0.],
                                [0., 0.1, 0.],
                            ], device='cuda')

                            if hasattr(obj_node, 'implicit_network'):
                                # Get frame latent code (conditioning)
                                if hasattr(obj_node, 'implicit_network') and hasattr(obj_node, 'frame_latent_encoder'):
                                    try:
                                        frame_latents = obj_node.frame_latent_encoder(frame_idx)
                                        print(f"  Frame latent shape: {frame_latents.shape}")

                                        # Check what conditioning the network expects
                                        if hasattr(obj_node.implicit_network, 'cond'):
                                            cond_mode = obj_node.implicit_network.cond
                                            print(f"  Network conditioning mode: '{cond_mode}'")
                                        else:
                                            cond_mode = "unknown"
                                            print(f"  ⚠️  Network has no 'cond' attribute!")

                                        # Test single point
                                        test_pt = torch.tensor([[0., 0., 0.]], device='cuda')  # [1, 3]
                                        B = frame_latents.shape[0]

                                        pts_batched = test_pt.unsqueeze(0)  # [1, 1, 3]

                                        # Create conditioning as DICTIONARY (the correct format!)
                                        cond_tensor = frame_latents.unsqueeze(1)  # [1, 1, 32]

                                        # Build dict with all possible conditioning keys
                                        cond_dict = {
                                            "frame": cond_tensor,
                                            "pose": torch.zeros(1, 1, 48, device='cuda'),  # Dummy pose if needed
                                        }

                                        print(f"  Query shapes: pts={pts_batched.shape}, cond['frame']={cond_dict['frame'].shape}")

                                        try:
                                            # Pass dictionary, not raw tensor!
                                            sdf_output = obj_node.implicit_network(pts_batched, cond_dict)
                                            print(f"  ✓ Dict-conditioned SDF succeeded: shape={sdf_output.shape}")

                                            # Extract value
                                            sdf_val = sdf_output.flatten()[0].item()
                                            print(f"  Object SDF at origin: {sdf_val:.4f}")

                                            # Test multiple points
                                            test_pts = torch.tensor([
                                                [0., 0., 0.],
                                                [0.1, 0., 0.],
                                                [0., 0.1, 0.],
                                                [0., 0., 0.1],
                                                [-0.1, 0., 0.],
                                            ], device='cuda')

                                            N = test_pts.shape[0]
                                            pts_multi = test_pts.unsqueeze(0)  # [1, 5, 3]

                                            # ✅ FIX: Conditioning should be [B, C], NOT [B, N, C]!
                                            # The implicit network will handle expansion to [B, N, C] internally
                                            cond_multi = {
                                                "frame": frame_latents,  # [1, 32] - 2D only!
                                                "pose": frame_latents,   # [1, 32] - 2D only!
                                            }

                                            print(f"  Multi-point query: pts={pts_multi.shape}, cond['pose']={cond_multi['pose'].shape}")

                                            try:
                                                sdf_multi = obj_node.implicit_network(pts_multi, cond_multi)
                                                print(f"  ✓ Multi-point SDF succeeded! Output shape: {sdf_multi.shape}")

                                                # Extract SDF from output (first channel of 257)
                                                if sdf_multi.shape[-1] > 1:
                                                    sdf_vals = sdf_multi[..., 0]  # Get SDF channel only: [1, 5]
                                                else:
                                                    sdf_vals = sdf_multi.squeeze(-1)  # [1, 5]

                                                print(f"  Multi-point SDF shape: {sdf_vals.shape}")
                                                print(f"  Object SDF samples:")
                                                for i in range(N):
                                                    pt = test_pts[i].cpu().numpy()
                                                    val = sdf_vals[0, i].item()
                                                    print(f"    {pt}: {val:.4f}")

                                                # Statistics
                                                sdf_flat = sdf_vals.flatten()
                                                mean_val = sdf_flat.mean().item()
                                                std_val = sdf_flat.std().item()
                                                min_val = sdf_flat.min().item()
                                                max_val = sdf_flat.max().item()

                                                print(f"  SDF stats: mean={mean_val:.4f}, std={std_val:.4f}, min={min_val:.4f}, max={max_val:.4f}")

                                                if std_val < 0.01:
                                                    print(f"    ⚠️  Near-constant SDF - object geometry degenerate!")
                                                elif abs(mean_val) > 2.0:
                                                    print(f"    ⚠️  Large SDF values - object outside bounds!")
                                                else:
                                                    print(f"    ✓ SDF values reasonable - object geometry exists!")

                                            except Exception as e:
                                                print(f"  ❌ Multi-point query failed: {e}")
                                                import traceback
                                                traceback.print_exc()
                                        except Exception as e:
                                            print(f"  ❌ Dict-conditioned query failed: {e}")
                                            traceback.print_exc()
                                    except Exception as e:
                                        print(f"  ❌ SDF query error: {e}")
                                        exc_type, exc_value, exc_tb = sys.exc_info()
                                        print("  Traceback:")
                                        for line in traceback.format_exception(exc_type, exc_value, exc_tb):
                                            print(f"    {line.rstrip()}")
                                else:
                                    print(f"  ❌ Object node missing frame_latent_encoder!")
                            else:
                                print(f"  ❌ Object node has no implicit_network!")
                    except Exception as e:
                        print(f"  ❌ SDF query failed: {type(e).__name__}: {e}")
                        print("  Full traceback:")
                        traceback.print_exc()
                else:
                    print(f"  ❌ No object node in model!")

                # Check hand parameters
                if 'right.fullpose' in batch_cuda:
                    hand_pose = batch_cuda['right.fullpose'][0]
                    print(f"  Hand pose shape: {hand_pose.shape}, mean: {hand_pose.mean():.4f}")

                # ============== END DEBUG BLOCK ==============
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
    parser.add_argument('--downsample', type=int, default=1,  # ← NEW
                        help='Render downsample factor (1=full 640x480, 2=half 320x240)')
    cmd_args = parser.parse_args()

    # Parse frame indices
    frame_indices = [int(x.strip()) for x in cmd_args.frames.split(',')]
    print(f"Target frames: {frame_indices}")
    print(f"Render downsample: {cmd_args.downsample}")  # ← NEW

    # Configuration
    # config_path = 'confs/stage1_hold_MC1_ho3d.yaml' # Case hold_MC1_ho3d Stage 1 Configuration File
    # config_path = 'confs/stage1_hold_MC1_ho3d_8layer_implicit.yaml' # Case hold_MC1_ho3d Stage 1 8-layer implicitnet MANO Enabled Configuration File
    # config_path = 'confs/stage1_hold_MC1_ho3d_8layer_implicit_official_match_fixed.yaml' # Case hold_MC1_ho3d Stage 1 8-layer implicitnet MANO Enabled New Configuration File
    # config_path = 'confs/stage1_hold_MC1_ho3d_8layer_implicit_joint002.yaml' # Case hold_MC1_ho3d Stage 1 8-layer implicitnet Joint Supervision Configuration File
    # config_path = 'confs/ghop_stage1_rgb_only.yaml' # Case hold_bottle1_itw Stage 1 Configuration File(Stage 2 and 3 use the same. Phase 3, 4, 5 have no influence on rendering process)
    # config_path = '/home/fredcui/Projects/hold-master/code/confs/general.yaml' # HOLD Officail Configuration
    config_path = 'confs/stage1_hold_MC1_ho3d_sds_from_official.yaml' # Case hold_MC1_ho3d Stage 2 on Official Checkpoint
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
        # ('logs/ab5edc20f/checkpoints/last.ckpt', 1), # Stage 1 Checkpoint
        # ('logs/75d213d30/checkpoints/last.ckpt', 2), # Stage 2 Checkpoint
        # ('logs/98b18938c/checkpoints/last.ckpt', 21),  # Stage 2 Checkpoint with new text prompt
        # ('logs/4d248833e/checkpoints/last.ckpt', 22),  # Stage 2 Checkpoint with refined SDF calculation(1 epoch test)
        # ('logs/20d1ec1e8/checkpoints/last.ckpt', 23),  # Stage 2 Checkpoint with refined SDF calculation(20 epoch test)
        # ('logs/b3b7ca677/checkpoints/last.ckpt', 24),  # Stage 2 Checkpoint with refined SDF calculation(16 epoch test) Run A
        # ('logs/f70ec4323/checkpoints/last.ckpt', 24),  # Stage 2 Checkpoint with refined SDF calculation(20 epoch test) Run A
        # ('logs/4848a499d/checkpoints/last.ckpt', 24),  # Stage 2 Checkpoint with refined SDF calculation(25 epoch test) Run A
        # ('logs/a249430cb/checkpoints/last.ckpt', 25),
        # Stage 2 Checkpoint with refined SDF calculation(16 epoch test) Run B
        # ('logs/2abee7631/checkpoints/last.ckpt', 26),
        # Stage 2 Checkpoint with refined SDF calculation(16 epoch test) Run C
        # ('logs/07080837c/checkpoints/last.ckpt', 3),  # Stage 3 Checkpoint
        # hold_MC1_ho3d case
        # ('logs/140dc5c18/checkpoints/last.ckpt', 1), # Stage 1 Checkpoint
        # ('logs/e1c95c0d0/checkpoints/last.ckpt', 11), # Stage 1 Checkpoint 8-layer implicitnet 200 epochs
        # ('logs/8d40bcd35/checkpoints/last.ckpt', 12), # Stage 1 Checkpoint 8-layer implicitnet MANO Enabled joint 0.01 supervision 20 epochs
        # ('logs/d839b2738/checkpoints/last.ckpt', 13),  # Stage 1 Checkpoint 8-layer implicitnet 70-epoch MANO Enabled
        # ('logs/6fc82956f/checkpoints/last.ckpt', 14),  # Stage 1 Checkpoint 8-layer implicitnet 70-epoch MANO Enabled New Config
        # ('logs/a0419ab35/checkpoints/last.ckpt', 15), # Stage 1 Checkpoint 8-layer implicitnet 20-epoch MANO Updated
        # ('logs/7a34708ef/checkpoints/last.ckpt', 15), # Stage 1 Checkpoint 8-layer implicitnet 100-epoch MANO Updated
        # ('logs/4fa8bb20d/checkpoints/last.ckpt', 2), # Stage 2 Checkpoint
        # ('logs/a0c32d3e8/checkpoints/last.ckpt', 21), # Stage 2 Checkpoint 15 epochs Refiend SDS
        # ('logs/70d907fbb/checkpoints/last.ckpt', 22),  # Stage 2 Checkpoint 30 epochs Refiend SDS
        # ('logs/482915ef4/checkpoints/last.ckpt', 23),  # Stage 2 Checkpoint 1-epoch on Official Checkpoint
        ('logs/eb4395048/checkpoints/last.ckpt', 24),  # Stage 2 Checkpoint 30-epoch on Official Checkpoint
        # ('logs/adfdabdc0/checkpoints/last.ckpt', 25),  # Stage 2 Checkpoint 70-epoch on Official Checkpoint
        # ('logs/afb17c622/checkpoints/last.ckpt', 26),  # Stage 2 Checkpoint 70-epoch(full SDS) on Official Checkpoint
        # ('logs/19a598d7e/checkpoints/last.ckpt', 3),  # Stage 3 Checkpoint
        # ('logs/fafeb1145/checkpoints/last.ckpt', 31),  # Stage 3 Checkpoint with updated SDS
        # ('logs/8ceebe9d0/checkpoints/last.ckpt', 3),  # Stage 3 Checkpoint 30-epoch
        # ('logs/ac71c88b7/checkpoints/last.ckpt', 31),  # Stage 3 Checkpoint 40-epoch
        # ('logs/75def08b1/checkpoints/last.ckpt', 32),  # Stage 3 Checkpoint Refined Phase 4 30-epoch
        # ('logs/33c12e63d/checkpoints/last.ckpt', 33),  # Stage 3 Checkpoint Refined GHOP ckpt loading 30-epoch
        # hold_GPMF12_ho3d case
        # ('logs/fadb8ec38/checkpoints/last.ckpt', 20),  # Stage 2 Checkpoint 10-epoch(full SDS) on Official Checkpoint
        # GHOP Official Checkpoint
        # ('/home/fredcui/Projects/holdse/code/checkpoints/ghop/last.ckpt', 100),
        # HOLD Official Checkpoint(Case hold_bottle1_itw)
        # ('/home/fredcui/Projects/hold/code/logs/009c2e923/checkpoints/last.ckpt', 999), # hold_bottle1_itw
        # ('/home/fredcui/Projects/hold/code/logs/cb20a1702/checkpoints/last.ckpt', 999), # hold_MC1_ho3d
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
            epoch, stats = render_checkpoint(ckpt_path, config_path, output_dir, frame_indices=frame_indices, downsample=cmd_args.downsample)

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
# Full resolution (640×480)
python scripts/render_validation_with_arg.py --frames 0,50,100 --downsample 1
python scripts/render_validation_with_arg.py --frames 10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30 --downsample 1
python scripts/render_validation_with_arg.py --frames 0 --downsample 1 2>&1 | tee logs/evaluation_results/MC1_official_rendering$(date +%d%H%M%S).log
# Half resolution (320×240) - faster for testing
python scripts/render_validation_with_arg.py --frames 0,50,100 --downsample 2
'''