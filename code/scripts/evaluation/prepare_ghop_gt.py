# File: code/scripts/evaluation/prepare_ghop_gt.py
"""
Simplified ground truth extraction - uses pose parameters directly.
For full mesh-based evaluation, MANO layer integration needed.
"""

import numpy as np
import torch
from pathlib import Path
import trimesh
from loguru import logger
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))


def compute_bounding_box_centers(vertices):
    """Compute bounding box centers."""
    if vertices.ndim == 3:
        bbox_min = vertices.min(dim=1)[0]
        bbox_max = vertices.max(dim=1)[0]
    else:
        bbox_min = vertices.min(dim=0)[0]
        bbox_max = vertices.max(dim=0)[0]
    return (bbox_min + bbox_max) / 2

def invert_transform(T):
    """Invert 4x4 transformation matrices [N, 4, 4]."""
    R = T[:, :3, :3]
    t = T[:, :3, 3:4]
    R_inv = R.transpose(1, 2)
    t_inv = -R_inv @ t
    T_inv = torch.zeros_like(T)
    T_inv[:, :3, :3] = R_inv
    T_inv[:, :3, 3:4] = t_inv
    T_inv[:, 3, 3] = 1.0
    return T_inv

def load_ghop_ground_truth(ghop_dir):
    """Load GHOP ground truth (simplified - pose parameters only)."""
    ghop_dir = Path(ghop_dir).expanduser()

    logger.info("="*70)
    logger.info(f"Loading GHOP ground truth from: {ghop_dir}")
    logger.info("="*70)

    # Load hand parameters
    hands = np.load(ghop_dir / 'hands.npz')
    hA = torch.from_numpy(hands['hA']).float().squeeze(1)
    beta = torch.from_numpy(hands['beta']).float().squeeze(1)
    N = len(hA)
    logger.info(f"✅ Hand poses: {hA.shape}")

    # Load cameras
    cameras = np.load(ghop_dir / 'cameras_hoi.npz')
    if 'wTc' in cameras:
        c2w = torch.from_numpy(cameras['wTc']).float()
    else:
        cTw = torch.from_numpy(cameras['cTw']).float()
        c2w = invert_transform(cTw)
    K = torch.from_numpy(cameras['K_pix']).float()
    logger.info(f"✅ Camera poses: {c2w.shape}")

    # Load object mesh
    obj_mesh = trimesh.load(ghop_dir / 'oObj.obj')
    v3d_obj = torch.from_numpy(obj_mesh.vertices).float()
    faces_obj = torch.from_numpy(obj_mesh.faces).long()
    logger.info(f"✅ Object mesh: {v3d_obj.shape[0]} vertices")

    # Object for all frames
    root_obj = compute_bounding_box_centers(
        v3d_obj.unsqueeze(0).repeat(N, 1, 1)
    )
    v3d_obj_all = v3d_obj.unsqueeze(0).repeat(N, 1, 1)

    # Ground truth dictionary
    data_gt = {
        # Hand parameters (main GT)
        'hA': hA,
        'right.betas': beta,

        # Object
        'v3d_o_c': v3d_obj_all,
        'v3d_ra.object': v3d_obj_all - root_obj.unsqueeze(1),
        'faces.object': faces_obj,
        'root.object': root_obj,

        # Camera
        'c2w': c2w,
        'intrinsics': K,

        # Validity
        'is_valid': torch.ones(N),

        # Metadata
        'fnames': [f'{i:05d}.png' for i in range(N)],
        'full_seq_name': ghop_dir.name,
    }

    logger.info("="*70)
    logger.info("Ground Truth Summary:")
    logger.info(f"  Frames: {N}")
    logger.info(f"  Hand pose params: {hA.shape}")
    logger.info(f"  Object vertices: {v3d_obj_all.shape}")
    logger.info("="*70)

    return data_gt

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ghop_dir', type=str, required=True)
    parser.add_argument('--output', type=str, default='ghop_gt.pth')
    args = parser.parse_args()

    data_gt = load_ghop_ground_truth(args.ghop_dir)

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Save
    torch.save(data_gt, args.output)
    logger.info(f"✅ Saved to: {args.output}")