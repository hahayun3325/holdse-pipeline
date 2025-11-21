#!/usr/bin/env python3
"""
Prepare exported meshes for HOLD evaluation.
"""
import os
import numpy as np
import trimesh
import argparse
import torch

def load_mesh(obj_path):
    """Load OBJ file as vertices and faces."""
    mesh = trimesh.load(obj_path, process=False)
    return mesh.vertices, mesh.faces


def prepare_evaluation_data(mesh_dir, output_dir, seq_name, step=32000):
    """Package exported meshes for evaluate.py"""

    # Load final epoch meshes
    hand_path = os.path.join(mesh_dir, f"mesh_cano_right_step_{step}.obj")
    obj_path = os.path.join(mesh_dir, f"mesh_cano_object_step_{step}.obj")

    if not os.path.exists(hand_path):
        raise FileNotFoundError(f"Hand mesh not found: {hand_path}")
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"Object mesh not found: {obj_path}")

    print(f"Loading hand mesh: {hand_path}")
    hand_verts, hand_faces = load_mesh(hand_path)

    print(f"Loading object mesh: {obj_path}")
    obj_verts, obj_faces = load_mesh(obj_path)

    # Package for evaluation
    eval_data = {
        'full_seq_name': seq_name,
        'hand_verts': hand_verts,
        'hand_faces': hand_faces,
        'object_verts': [obj_verts],  # List format for multi-frame
        'object_faces': [obj_faces],
        'hand_pose': torch.stack(all_poses),        # [T, 48]
        'hand_shape': torch.stack(all_shapes),      # [T, 10]
    }

    # Save
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, seq_name + '.npy')
    # np.save(save_path, eval_data)
    torch.save(eval_data, save_path)  # Fixed: PyTorch format

    print(f"\n{'=' * 70}")
    print(f"✅ Evaluation data prepared!")
    print(f"{'=' * 70}")
    print(f"Saved as PyTorch checkpoint: {save_path}")
    print(f"Hand vertices: {hand_verts.shape}")
    print(f"Hand faces: {hand_faces.shape}")
    print(f"Object vertices: {obj_verts.shape}")
    print(f"Object faces: {obj_faces.shape}")
    print(f"\nNext step:")
    print(f"python evaluate.py --sd_p {save_path}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_dir', type=str, required=True,
                        help='Directory containing exported meshes')
    parser.add_argument('--output_dir', type=str, default='evaluation',
                        help='Output directory for evaluation data')
    parser.add_argument('--seq', type=str, default='hold_bottle1_itw',
                        help='Sequence name')
    parser.add_argument('--step', type=int, default=32000,
                        help='Training step to use (default: 32000 = final)')
    args = parser.parse_args()

    prepare_evaluation_data(args.mesh_dir, args.output_dir, args.seq, args.step)

'''
# Run for all three checkpoints
python scripts/prepare_evaluation_data.py \
    --mesh_dir logs/b3b7ca677/mesh_cano \
    --output_dir logs/evaluation/stage2_runA \
    --seq hold_bottle1_itw \
    --step 32000

python scripts/prepare_evaluation_data.py \
    --mesh_dir logs/a249430cb/mesh_cano \
    --output_dir logs/evaluation/stage2_runB \
    --seq hold_bottle1_itw \
    --step 32000

python scripts/prepare_evaluation_data.py \
    --mesh_dir logs/2abee7631/mesh_cano \
    --output_dir logs/evaluation/stage2_runC \
    --seq hold_bottle1_itw \
    --step 32000
'''