#!/usr/bin/env python3
"""
Reconstruct full sequence from HOLD checkpoint for evaluation.

This script extracts per-frame predictions (meshes + MANO parameters)
from a trained HOLD checkpoint, formatting them for evaluate.py.

Usage:
    python scripts/reconstruct_sequence.py \
        --checkpoint logs/2abee7631/checkpoints/last.ckpt \
        --case hold_bottle1_itw \
        --output_dir evaluation/stage2_sequence \
        --save_meshes \
        --save_params
"""

#!/usr/bin/env python3
import sys
import os
import torch
import numpy as np
import trimesh
import argparse
from tqdm import tqdm
from loguru import logger

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Parent of scripts/
sys.path.insert(0, project_root)

# Now imports work
from src.hold.hold import HOLD
from src.datasets.hold_dataset import HOLDDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to HOLD checkpoint')
    parser.add_argument('--case', type=str, default='hold_bottle1_itw',
                        help='Dataset case name')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for predictions')
    parser.add_argument('--save_meshes', action='store_true',
                        help='Export OBJ meshes for each frame')
    parser.add_argument('--save_params', action='store_true',
                        help='Save MANO parameters separately')
    parser.add_argument('--resolution', type=int, default=128,
                        help='Marching cubes resolution')
    parser.add_argument('--frames', type=str, default=None,
                        help='Comma-separated frame indices (default: all)')
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()


def load_model(checkpoint_path, device):
    """Load HOLD model from checkpoint."""
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    model = HOLD.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    model.to(device)

    return model


def load_dataset(model, case):
    """Load dataset to get frame indices and ground truth."""
    # Get dataset configuration from model
    dataset_path = f"./data/{case}/build"

    # Check if dataset exists
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset not found: {dataset_path}")

    dataset = HOLDDataset(
        data_path=dataset_path,
        split='train',  # Use full dataset
        subsample=1  # No subsampling
    )

    logger.info(f"Loaded dataset: {len(dataset)} frames")
    return dataset


def extract_frame_prediction(model, dataset, frame_idx, resolution, device):
    """Extract prediction for a single frame.

    Returns:
        hand_verts: [778, 3] or high-res MANO vertices
        hand_faces: [1538, 3] MANO faces
        obj_verts: [V, 3] object vertices
        obj_faces: [F, 3] object faces
        hand_pose: [48] MANO pose
        hand_shape: [10] MANO shape
        hand_trans: [3] hand translation
    """
    with torch.no_grad():
        # Get batch for this frame
        batch = dataset[frame_idx]

        # Add batch dimension and move to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].unsqueeze(0).to(device)

        # Add frame index
        batch['idx'] = torch.tensor([[frame_idx]]).to(device)

        # Run inference
        model.eval()

        # ================================================================
        # Extract Hand Mesh + Parameters
        # ================================================================
        hand_verts, hand_faces = model._extract_hand_mesh(batch)

        # Extract MANO parameters from batch or nodes
        hand_node = None
        for node in model.model.nodes.values():
            if 'right' in node.node_id.lower():
                hand_node = node
                break

        if hand_node is None:
            raise ValueError("No hand node found")

        # Get parameters
        params = hand_node.params(batch['idx'])

        # Extract pose and shape
        if 'fullpose' in params:
            hand_pose = params['fullpose'].cpu().squeeze()
        elif 'pose' in params and 'global_orient' in params:
            hand_pose = torch.cat([
                params['global_orient'].cpu().squeeze(),
                params['pose'].cpu().squeeze()
            ])
        else:
            raise ValueError("Cannot find hand pose in params")

        if 'betas' in params:
            hand_shape = params['betas'].cpu().squeeze()
        else:
            # Default shape if not optimized
            hand_shape = torch.zeros(10)

        if 'transl' in params:
            hand_trans = params['transl'].cpu().squeeze()
        else:
            hand_trans = torch.zeros(3)

        # ================================================================
        # Extract Object Mesh
        # ================================================================
        obj_verts_list, obj_faces_list = model._extract_object_mesh_from_sdf(batch)
        obj_verts = obj_verts_list[0].cpu()
        obj_faces = obj_faces_list[0].cpu()

        # Squeeze batch dimension from hand mesh
        hand_verts = hand_verts.cpu().squeeze()
        hand_faces = hand_faces.cpu()

    return {
        'hand_verts': hand_verts,
        'hand_faces': hand_faces,
        'obj_verts': obj_verts,
        'obj_faces': obj_faces,
        'hand_pose': hand_pose,
        'hand_shape': hand_shape,
        'hand_trans': hand_trans,
    }


def save_frame_meshes(frame_data, frame_idx, output_dir):
    """Save meshes as OBJ files."""
    mesh_dir = os.path.join(output_dir, 'meshes')
    os.makedirs(mesh_dir, exist_ok=True)

    # Save hand mesh
    hand_mesh = trimesh.Trimesh(
        vertices=frame_data['hand_verts'].numpy(),
        faces=frame_data['hand_faces'].numpy()
    )
    hand_path = os.path.join(mesh_dir, f'hand_frame_{frame_idx:04d}.obj')
    hand_mesh.export(hand_path)

    # Save object mesh
    if frame_data['obj_verts'].shape[0] > 0:
        obj_mesh = trimesh.Trimesh(
            vertices=frame_data['obj_verts'].numpy(),
            faces=frame_data['obj_faces'].numpy()
        )
        obj_path = os.path.join(mesh_dir, f'object_frame_{frame_idx:04d}.obj')
        obj_mesh.export(obj_path)


def package_evaluation_data(all_predictions, seq_name, output_dir):
    """Package predictions in format expected by evaluate.py"""

    # Stack per-frame data
    hand_verts_seq = torch.stack([p['hand_verts'] for p in all_predictions])
    hand_poses_seq = torch.stack([p['hand_pose'] for p in all_predictions])
    hand_shapes_seq = torch.stack([p['hand_shape'] for p in all_predictions])
    hand_trans_seq = torch.stack([p['hand_trans'] for p in all_predictions])

    # Object vertices (variable topology per frame)
    obj_verts_seq = [p['obj_verts'] for p in all_predictions]
    obj_faces_seq = [p['obj_faces'] for p in all_predictions]

    # Hand faces (constant topology)
    hand_faces = all_predictions[0]['hand_faces']

    # Create evaluation dictionary
    eval_data = {
        'full_seq_name': seq_name,
        'hand_verts': hand_verts_seq,  # [T, 778, 3]
        'hand_faces': hand_faces,  # [1538, 3]
        'object_verts': obj_verts_seq,  # List of [V_i, 3]
        'object_faces': obj_faces_seq,  # List of [F_i, 3]
        'hand_pose': hand_poses_seq,  # [T, 48]
        'hand_shape': hand_shapes_seq,  # [T, 10]
        'hand_trans': hand_trans_seq,  # [T, 3]
    }

    # Save as PyTorch checkpoint
    save_path = os.path.join(output_dir, f'{seq_name}.pth')
    torch.save(eval_data, save_path)

    logger.info(f"Saved evaluation data: {save_path}")
    logger.info(f"  Frames: {hand_verts_seq.shape[0]}")
    logger.info(f"  Hand verts shape: {hand_verts_seq.shape}")
    logger.info(f"  Hand pose shape: {hand_poses_seq.shape}")

    return save_path


def main():
    args = parse_args()

    # Load model
    model = load_model(args.checkpoint, args.device)

    # Load dataset
    dataset = load_dataset(model, args.case)

    # Determine frames to process
    if args.frames:
        frame_indices = [int(f) for f in args.frames.split(',')]
    else:
        frame_indices = list(range(len(dataset)))

    logger.info(f"Processing {len(frame_indices)} frames")

    # Extract predictions for each frame
    all_predictions = []

    for frame_idx in tqdm(frame_indices, desc="Extracting frames"):
        try:
            frame_data = extract_frame_prediction(
                model, dataset, frame_idx, args.resolution, args.device
            )

            all_predictions.append(frame_data)

            # Optionally save meshes
            if args.save_meshes:
                save_frame_meshes(frame_data, frame_idx, args.output_dir)

        except Exception as e:
            logger.error(f"Failed to extract frame {frame_idx}: {e}")
            import traceback
            traceback.print_exc()

    # Package for evaluation
    if len(all_predictions) > 0:
        eval_path = package_evaluation_data(
            all_predictions, args.case, args.output_dir
        )

        # Optionally save parameters separately
        if args.save_params:
            poses = torch.stack([p['hand_pose'] for p in all_predictions])
            shapes = torch.stack([p['hand_shape'] for p in all_predictions])
            np.save(os.path.join(args.output_dir, 'poses.npy'), poses.numpy())
            np.save(os.path.join(args.output_dir, 'shapes.npy'), shapes.numpy())

        print(f"\n{'=' * 70}")
        print(f"✅ Sequence reconstruction complete!")
        print(f"{'=' * 70}")
        print(f"Evaluation file: {eval_path}")
        print(f"Frames processed: {len(all_predictions)}")
        if args.save_meshes:
            print(f"Meshes saved to: {os.path.join(args.output_dir, 'meshes')}")
        print(f"\nNext step:")
        print(f"python evaluate.py --sd_p {eval_path}")
        print(f"{'=' * 70}")
    else:
        logger.error("No frames successfully extracted")


if __name__ == '__main__':
    main()

'''
# Make script executable
chmod +x scripts/reconstruct_sequence.py

# Run for all three checkpoints
for run_id in b3b7ca677 a249430cb 2abee7631; do
    python scripts/reconstruct_sequence.py \
        --checkpoint logs/$run_id/checkpoints/last.ckpt \
        --case hold_bottle1_itw \
        --output_dir logs/evaluation/${run_id}_sequence \
        --save_meshes \
        --save_params \
        --resolution 128
done
'''