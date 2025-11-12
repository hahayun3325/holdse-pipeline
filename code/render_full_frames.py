# File: code/render_full_frames.py
"""
Custom rendering script that bypasses validation_epoch_end
and directly saves full frames.
"""

import torch
import os
from src.hold.hold import HOLD
from src.utils.parser import parser_args
import os.path as op
from common.torch_utils import reset_all_seeds
import numpy as np
from pprint import pprint
import sys

sys.path = [".."] + sys.path
from src.datasets.utils import create_dataset
import common.thing as thing
from tqdm import tqdm
from pathlib import Path
import cv2


def extract_img_size_robust(img_size_raw):
    """
    Robust extraction of H, W from various img_size structures.

    Handles:
    - TempoDataset: [[tensor([H]), tensor([W])], ...]
    - Regular: tensor([H, W]) or [H, W]
    """
    try:
        # TempoDataset nested list structure
        if isinstance(img_size_raw, list) and len(img_size_raw) > 0:
            first_frame = img_size_raw[0]
            if isinstance(first_frame, list) and len(first_frame) >= 2:
                H = int(first_frame[0].cpu().item() if isinstance(first_frame[0], torch.Tensor) else first_frame[0])
                W = int(first_frame[1].cpu().item() if isinstance(first_frame[1], torch.Tensor) else first_frame[1])
                return H, W

        # Regular tensor/array structure
        img_size_flat = np.array(img_size_raw).flatten()
        if len(img_size_flat) >= 2:
            return int(img_size_flat[0]), int(img_size_flat[1])

    except Exception:
        pass

    return 512, 512  # Default fallback


def save_full_frame(outputs, frame_idx, output_dir):
    """
    Save full rendered frame (not pixel samples).

    Args:
        outputs: Model output dict with rendered images
        frame_idx: Frame number
        output_dir: Base output directory
    """
    output_dir = Path(output_dir)

    # Create subdirectories
    dirs = {
        'rgb': output_dir / 'rgb',
        'normal': output_dir / 'normal',
        'imap': output_dir / 'imap',
        'mask_prob': output_dir / 'mask_prob',
        'fg_rgb': output_dir / 'fg_rgb',
        'bg_rgb': output_dir / 'bg_rgb',
        'hand_rgb': output_dir / 'hand_rgb',
        'object_rgb': output_dir / 'object_rgb',
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Extract and save each modality
    # RGB (composite)
    if 'rgb' in outputs:
        rgb = outputs['rgb']  # Should be [H, W, 3]
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.cpu().numpy()
        if rgb.max() <= 1.0:
            rgb = (rgb * 255).astype(np.uint8)
        cv2.imwrite(str(dirs['rgb'] / f'{frame_idx:05d}.png'),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    # Normal map
    if 'normal' in outputs:
        normal = outputs['normal']
        if isinstance(normal, torch.Tensor):
            normal = normal.cpu().numpy()
        # Convert from [-1, 1] to [0, 255]
        normal = ((normal + 1) / 2 * 255).astype(np.uint8)
        cv2.imwrite(str(dirs['normal'] / f'{frame_idx:05d}.png'),
                    cv2.cvtColor(normal, cv2.COLOR_RGB2BGR))

    # Instance map
    if 'imap' in outputs or 'instance_map' in outputs:
        imap_key = 'instance_map' if 'instance_map' in outputs else 'imap'
        imap = outputs[imap_key]
        if isinstance(imap, torch.Tensor):
            imap = imap.cpu().numpy()
        if imap.dtype == np.float32:
            imap = (imap * 255).astype(np.uint8)
        cv2.imwrite(str(dirs['imap'] / f'{frame_idx:05d}.png'), imap)

    # Mask probability
    if 'mask_prob' in outputs:
        mask = outputs['mask_prob']
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        if mask.max() <= 1.0:
            mask = (mask * 255).astype(np.uint8)
        cv2.imwrite(str(dirs['mask_prob'] / f'{frame_idx:05d}.png'), mask)

    # Foreground RGB
    if 'fg_rgb' in outputs:
        fg_rgb = outputs['fg_rgb']
        if isinstance(fg_rgb, torch.Tensor):
            fg_rgb = fg_rgb.cpu().numpy()
        if fg_rgb.max() <= 1.0:
            fg_rgb = (fg_rgb * 255).astype(np.uint8)
        cv2.imwrite(str(dirs['fg_rgb'] / f'{frame_idx:05d}.png'),
                    cv2.cvtColor(fg_rgb, cv2.COLOR_RGB2BGR))

    # Background RGB
    if 'bg_rgb' in outputs:
        bg_rgb = outputs['bg_rgb']
        if isinstance(bg_rgb, torch.Tensor):
            bg_rgb = bg_rgb.cpu().numpy()
        if bg_rgb.max() <= 1.0:
            bg_rgb = (bg_rgb * 255).astype(np.uint8)
        cv2.imwrite(str(dirs['bg_rgb'] / f'{frame_idx:05d}.png'),
                    cv2.cvtColor(bg_rgb, cv2.COLOR_RGB2BGR))


def extract_full_frame_from_inference(batch_output, img_size, render_downsample=2):
    """
    Extract full frame image from inference_step output.

    Args:
        batch_output: Output from model.inference_step()
        img_size: Original image dimensions [H, W]
        render_downsample: Downsampling factor (default 2)

    Returns:
        dict with full rendered images
    """
    # Apply downsampling to image size
    H, W = img_size
    H_render = H // render_downsample
    W_render = W // render_downsample

    print(f"  Original size: {H}×{W}")
    print(f"  Render size: {H_render}×{W_render} (downsample={render_downsample})")

    full_frame = {}

    # RGB
    if 'rgb' in batch_output:
        rgb = batch_output['rgb']  # [N_rays, 3]
        if len(rgb.shape) == 2:
            # Verify size matches
            expected_rays = H_render * W_render
            actual_rays = rgb.shape[0]

            if actual_rays != expected_rays:
                # Auto-detect actual size
                H_render = int(np.sqrt(actual_rays))
                W_render = int(np.sqrt(actual_rays))
                print(f"  WARNING: Size mismatch, auto-detected: {H_render}×{W_render}")

            rgb = rgb.reshape(H_render, W_render, 3)
        full_frame['rgb'] = rgb

    # Normal
    if 'normal' in batch_output:
        normal = batch_output['normal']
        if len(normal.shape) == 2:
            normal = normal.reshape(H_render, W_render, 3)
        full_frame['normal'] = normal

    # Instance map
    if 'imap' in batch_output:
        imap = batch_output['imap']
        if len(imap.shape) == 1:
            imap = imap.reshape(H_render, W_render)
        full_frame['imap'] = imap

    # Mask probability
    if 'mask_prob' in batch_output:
        mask = batch_output['mask_prob']
        if len(mask.shape) == 1:
            mask = mask.reshape(H_render, W_render)
        full_frame['mask_prob'] = mask

    return full_frame


def export_mesh(vertices, faces, filepath):
    """Export mesh to OBJ file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        # Write vertices
        for v in vertices:
            f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')

        # Write faces (OBJ is 1-indexed)
        for face in faces:
            f.write(f'f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n')


def main():
    # ================================================================
    # CRITICAL: Disable Comet BEFORE parser_args()
    # ================================================================
    if '--no-comet' in sys.argv:
        print("\n" + "=" * 70)
        print("⚠️  COMET LOGGING DISABLED (render_full_frames.py)")
        print("=" * 70)
        os.environ['COMET_MODE'] = 'disabled'

    device = "cuda:0"
    args, opt = parser_args()

    print("Working dir:", os.getcwd())
    exp_key = args.load_ckpt.split("/")[1]
    args.log_dir = op.join("logs", exp_key, "test_full_render")

    pprint(args)

    # ================================================================
    # CRITICAL FIX: Force training checkpoint for GHOP
    # ================================================================
    print("\n" + "="*70)
    print("CHECKPOINT PATH CONFIGURATION")
    print("="*70)

    # Determine checkpoint path BEFORE creating model
    original_infer_ckpt = args.infer_ckpt if hasattr(args, 'infer_ckpt') else ""

    # Check if deployment checkpoint is being used
    is_deployment_ckpt = (
        'deployment' in original_infer_ckpt or
        'hoi4d_phase5' in original_infer_ckpt
    )

    if is_deployment_ckpt:
        # Deployment checkpoints don't work - use training checkpoint instead
        training_ckpt = op.join(args.load_ckpt, "checkpoints", "last.ckpt")

        print(f"⚠️  Deployment checkpoint detected:")
        print(f"   {original_infer_ckpt}")
        print(f"\n✓ Switching to training checkpoint:")
        print(f"   {training_ckpt}")
        print(f"\nReason: Deployment checkpoints missing GHOP structure.")
        print(f"Training checkpoints contain all components correctly.")

        # Override args BEFORE model creation
        args.infer_ckpt = training_ckpt
        args.ckpt_p = training_ckpt

    elif original_infer_ckpt and op.exists(original_infer_ckpt):
        print(f"✓ Using checkpoint: {original_infer_ckpt}")
        args.ckpt_p = original_infer_ckpt

    elif args.load_ckpt:
        # Auto-construct from load_ckpt
        training_ckpt = op.join(args.load_ckpt, "checkpoints", "last.ckpt")
        print(f"✓ Auto-constructed checkpoint path:")
        print(f"   {training_ckpt}")
        args.infer_ckpt = training_ckpt
        args.ckpt_p = training_ckpt

    else:
        raise ValueError("No checkpoint specified (use --infer_ckpt or --load_ckpt)")

    # Verify checkpoint exists
    ckpt_path = args.infer_ckpt if hasattr(args, 'infer_ckpt') and args.infer_ckpt else args.ckpt_p
    if not op.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"\n✓ Checkpoint verified:")
    print(f"  Path: {ckpt_path}")
    print(f"  Size: {op.getsize(ckpt_path) / (1024**2):.1f} MB")
    print("="*70 + "\n")

    # ================================================================
    # Load model (GHOP will now load correct checkpoint during init)
    # ================================================================
    print("Creating HOLD model...")
    print("  (GHOP components will load from checkpoint during initialization)")
    model = HOLD(opt, args)
    print("✓ Model created\n")

    # ================================================================
    # Load dataset
    # ================================================================
    print("="*70)
    print("DATASET LOADING")
    print("="*70)

    # Priority 1: Test dataset (TestDataset - single frames)
    if hasattr(opt.dataset, 'test'):
        print("Using test dataset configuration (single-frame TestDataset)")
        dataset_config = opt.dataset.test
        testset = create_dataset(dataset_config, args)

    # Priority 2: Validation dataset (ValDataset - single frames)
    elif hasattr(opt.dataset, 'valid') or hasattr(opt.dataset, 'val'):
        print("⚠️  No test config, using validation dataset (single-frame ValDataset)")
        dataset_config = getattr(opt.dataset, 'valid', None) or opt.dataset.val
        testset = create_dataset(dataset_config, args)

    # Priority 3: Create test config from train config
    else:
        print("⚠️  No test/valid config, creating test config from train...")
        from easydict import EasyDict
        test_config = EasyDict(opt.dataset.train.copy())
        test_config.type = "test"
        test_config.shuffle = False
        test_config.drop_last = False
        test_config.num_workers = 0
        testset = create_dataset(test_config, args)

    # Verify dataset type
    actual_dataset = testset.dataset if hasattr(testset, 'dataset') else testset
    dataset_class = type(actual_dataset).__name__

    print(f"\n✅ Dataset loaded:")
    print(f"  Class: {dataset_class}")
    print(f"  Total samples: {len(testset)}")

    if 'Tempo' in dataset_class:
        raise RuntimeError("TempoDataset loaded - use single-frame dataset for rendering")
    else:
        print(f"  ✅ Confirmed: Single-frame dataset ({dataset_class})")

    print("="*70 + "\n")

    # ================================================================
    # Load checkpoint weights (for non-GHOP components)
    # ================================================================
    print("="*70)
    print("LOADING CHECKPOINT WEIGHTS")
    print("="*70)

    reset_all_seeds(1)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt["state_dict"]
    print(f"  Loaded {len(sd)} parameters")

    # Load into model (GHOP already loaded during init, this loads everything else)
    print(f"\nLoading state dict into model...")
    missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)

    # Check status
    ghop_missing = [k for k in missing_keys if 'ghop' in k.lower() or 'vqvae' in k.lower() or 'unet' in k.lower()]

    if len(ghop_missing) > 0:
        print(f"\n⚠️  {len(ghop_missing)} GHOP keys not loaded via load_state_dict")
        print(f"   (This is normal - GHOP loaded during model init)")

    if len(missing_keys) > len(ghop_missing):
        print(f"\n⚠️  {len(missing_keys) - len(ghop_missing)} non-GHOP keys missing!")
    else:
        print(f"\n✅ All non-GHOP components loaded successfully")

    model.to(device)
    model.eval()

    print(f"✓ Model loaded and set to eval mode")
    print("="*70 + "\n")

    # Disable barf masks
    nodes = model.model.nodes
    for node in nodes.values():
        node.implicit_network.embedder_obj.eval()
    model.model.background.bg_implicit_network.embedder_obj.eval()
    model.model.background.bg_rendering_network.embedder_obj.eval()

    # ================================================================
    # CUSTOM FULL-FRAME RENDERING LOOP
    # ================================================================
    output_dir = Path(args.log_dir) / 'visuals'
    mesh_dir = Path(args.log_dir) / 'meshes'
    mesh_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🔄 Rendering {len(testset)} batches...")
    print(f"   Output: {output_dir}")

    frame_count = 0

    for batch_idx, batch in enumerate(tqdm(testset, desc="Rendering frames")):
        with torch.no_grad():
            batch = thing.thing2dev(batch, device)

            # Extract image size
            H, W = extract_img_size_robust(batch.get('img_size', None))

            # Run inference
            out = model.inference_step(batch)

            # DEBUG: First frame only
            if batch_idx == 0:
                print(f"\n[DEBUG] inference_step() output keys:")
                for key in sorted(out.keys()):
                    value = out[key]
                    if isinstance(value, torch.Tensor):
                        print(f"  {key}: shape={value.shape}, device={value.device}")
                    elif isinstance(value, np.ndarray):
                        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                    else:
                        print(f"  {key}: type={type(value)}")

                # Check RGB values
                if 'rgb' in out:
                    rgb = out['rgb']
                    print(f"\n[DEBUG] RGB statistics:")
                    print(f"  Min: {rgb.min().item():.6f}")
                    print(f"  Max: {rgb.max().item():.6f}")
                    print(f"  Mean: {rgb.mean().item():.6f}")
                    print(f"  Non-zero: {torch.count_nonzero(rgb).item()} / {rgb.numel()}")

            # Extract full frame from ray-based output
            full_frame = extract_full_frame_from_inference(out, (H, W))

            # DEBUG: First frame only
            if batch_idx == 0:
                print(f"\n[DEBUG] Extracted full_frame:")
                if len(full_frame) > 0:
                    for key, value in full_frame.items():
                        if isinstance(value, (torch.Tensor, np.ndarray)):
                            shape = value.shape if hasattr(value, 'shape') else len(value)
                            print(f"  {key}: shape={shape}")
                else:
                    print("  ⚠️  EMPTY!")
                print()

            # Save full frame
            if len(full_frame) > 0:
                save_full_frame(full_frame, frame_count, output_dir)

            # Export meshes if available
            if 'hand_verts' in out:
                hand_verts = out['hand_verts']
                if isinstance(hand_verts, torch.Tensor):
                    hand_verts = hand_verts.cpu().numpy()
                    if len(hand_verts.shape) == 3:
                        hand_verts = hand_verts[0]

                if hasattr(model, 'mano_faces'):
                    hand_faces = model.mano_faces.cpu().numpy()
                elif 'hand_faces' in out:
                    hand_faces = out['hand_faces'].cpu().numpy()
                else:
                    hand_faces = None

                if hand_faces is not None:
                    export_mesh(hand_verts, hand_faces,
                                mesh_dir / f'hand_{frame_count:05d}.obj')

            if 'object_verts' in out:
                obj_verts = out['object_verts']
                if isinstance(obj_verts, torch.Tensor):
                    obj_verts = obj_verts.cpu().numpy()
                    if len(obj_verts.shape) == 3:
                        obj_verts = obj_verts[0]

                if 'object_faces' in out:
                    obj_faces = out['object_faces'].cpu().numpy()
                    export_mesh(obj_verts, obj_faces,
                                mesh_dir / f'object_{frame_count:05d}.obj')

            frame_count += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"  Rendered {frame_count} frames")

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print(f"\n✅ Rendering complete!")
    print(f"   Total frames rendered: {frame_count}")
    print(f"   Output directory: {args.log_dir}")

    rgb_files = list((output_dir / 'rgb').glob('*.png'))
    normal_files = list((output_dir / 'normal').glob('*.png'))
    mesh_files = list(mesh_dir.glob('*.obj'))

    print(f"\n📊 Output summary:")
    print(f"   RGB images: {len(rgb_files)}")
    print(f"   Normal maps: {len(normal_files)}")
    print(f"   Meshes: {len(mesh_files)}")

    if len(rgb_files) < frame_count:
        print(f"\n⚠️  WARNING: Expected {frame_count} frames, got {len(rgb_files)} RGB images")
    else:
        print(f"\n✅ All {frame_count} frames rendered successfully!")


if __name__ == "__main__":
    main()

'''
cd ~/Projects/holdse/code

# Baseline (Epoch 20)
python render_full_frames.py \
    --case ghop_bottle_1 \
    --config confs/ghop_production_chunked_20251027_131408.yaml \
    --load_ckpt logs/ad1f0073b \
    --infer_ckpt ../deployment/hoi4d_phase5_v1.0/hoi4d_phase5_pre_v1.0.ckpt \
    --gpu_id 0 \
    --render_downsample 2 \
    --use_ghop \
    --no-comet > render_full_output.txt 2>&1

# Phase 5 (Epoch 25)
python render_full_frames.py \
    --case ghop_bottle_1 \
    --config confs/ghop_production_chunked_20251027_131408.yaml \
    --load_ckpt logs/6aaaf5002 \
    --infer_ckpt logs/6aaaf5002/checkpoints/last.ckpt \
    --gpu_id 0 \
    --render_downsample 2 \
    --no-comet \
    --agent_id -1 > render_full_output.txt 2>&1
'''