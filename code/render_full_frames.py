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
    device = "cuda:0"
    args, opt = parser_args()

    print("Working dir:", os.getcwd())
    exp_key = args.load_ckpt.split("/")[1]
    args.log_dir = op.join("logs", exp_key, "test_full_render")

    pprint(args)

    # Load model
    model = HOLD(opt, args)

    # Load dataset
    try:
        testset = create_dataset(opt.dataset.test, args)
        print("✅ Using test dataset")
    except AttributeError:
        print("⚠️  No test dataset config, using train dataset")
        testset = create_dataset(opt.dataset.train, args)

    print("\nDataset info:")
    img_paths = np.array(testset.dataset.dataset.img_paths)
    print(f"  Total images: {len(img_paths)}")
    print(f"  First 3: {img_paths[:3]}")
    print(f"  Last 3: {img_paths[-3:]}")

    # Load checkpoint
    reset_all_seeds(1)
    ckpt_path = None if args.ckpt_p == "" else args.ckpt_p
    sd = torch.load(ckpt_path)["state_dict"]
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()

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
            # Move to device
            batch = thing.thing2dev(batch, device)

            # Get image size
            if 'img_size' in batch:
                img_size = batch['img_size']
                if isinstance(img_size, torch.Tensor):
                    img_size = img_size.cpu().numpy()
                H, W = int(img_size[0]), int(img_size[1])
            else:
                H, W = 512, 512  # Default

            # Run inference
            out = model.inference_step(batch)

            # Extract full frame from ray-based output
            full_frame = extract_full_frame_from_inference(out, (H, W))

            # Save full frame
            save_full_frame(full_frame, frame_count, output_dir)

            # Export meshes if available
            if 'hand_verts' in out:
                hand_verts = out['hand_verts']
                if isinstance(hand_verts, torch.Tensor):
                    hand_verts = hand_verts.cpu().numpy()
                    if len(hand_verts.shape) == 3:
                        hand_verts = hand_verts[0]  # Remove batch dim

                # Get hand faces
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

            # Progress update
            if (batch_idx + 1) % 10 == 0:
                print(f"  Rendered {frame_count} frames")

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print(f"\n✅ Rendering complete!")
    print(f"   Total frames: {frame_count}")
    print(f"   Output directory: {args.log_dir}")

    # Verify outputs
    rgb_files = list((output_dir / 'rgb').glob('*.png'))
    mesh_files = list(mesh_dir.glob('*.obj'))

    print(f"\n📊 Output summary:")
    print(f"   RGB images: {len(rgb_files)}")
    print(f"   Meshes: {len(mesh_files)}")

    if len(rgb_files) < len(img_paths):
        print(f"\n⚠️  WARNING: Expected {len(img_paths)} frames, got {len(rgb_files)}")
    else:
        print(f"\n✅ All frames rendered successfully!")


if __name__ == "__main__":
    main()
