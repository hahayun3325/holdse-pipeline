import torch
import os
import os.path as op
import numpy as np
from PIL import Image
import sys

sys.path = [".."] + sys.path
from loguru import logger

from src.hold.hold import HOLD
from src.utils.parser import parser_args
from src.datasets.utils import create_dataset
import common.thing as thing
from common.torch_utils import reset_all_seeds


def save_normal_image(normal_array, save_path):
    """Save normal map as RGB image with white background."""
    # Ensure [0, 255] range
    if normal_array.dtype != np.uint8:
        normal_img = (normal_array * 255).clip(0, 255).astype(np.uint8)
    else:
        normal_img = normal_array
    Image.fromarray(normal_img).save(save_path)


def compute_normals_from_vertices(vertices, faces=None):
    """
    Compute vertex normals from point cloud or mesh.
    vertices: [N, 3] or [B, N, 3]
    Returns: [N, 3] normalized normals
    """
    if faces is not None:
        # Mesh-based normal computation would go here
        # For now, use point cloud normals
        pass

    # Estimate normals from point cloud using PCA on neighborhoods
    # Simple approximation: use vertex positions as proxy (works for sphere-like objects)
    # Better: use nearest neighbors to fit local planes
    normals = vertices.clone()
    # Normalize to unit length
    norm = torch.norm(normals, dim=-1, keepdim=True) + 1e-8
    normals = normals / norm
    return normals


def render_point_cloud_normals(vertices, cam_intrinsics, cam_pose, img_size, device='cuda'):
    """
    Render normal map from point cloud using simple projection.
    vertices: [N, 3] in world space
    cam_intrinsics: [3, 3] camera intrinsics
    cam_pose: [4, 4] camera extrinsics (world to cam)
    img_size: [H, W]
    Returns: [H, W, 3] normal map, [H, W] mask
    """
    H, W = img_size

    # Transform to camera space
    vertices_h = torch.cat([vertices, torch.ones(vertices.shape[0], 1, device=device)], dim=-1)
    vertices_cam = (cam_pose @ vertices_h.T).T[:, :3]

    # Project to image plane
    vertices_2d = (cam_intrinsics @ vertices_cam.T).T
    # Perspective divide
    vertices_2d = vertices_2d[:, :2] / (vertices_2d[:, 2:3] + 1e-8)

    # Filter points in front of camera
    valid_depth = vertices_cam[:, 2] > 0

    # Filter points within image bounds
    valid_x = (vertices_2d[:, 0] >= 0) & (vertices_2d[:, 0] < W)
    valid_y = (vertices_2d[:, 1] >= 0) & (vertices_2d[:, 1] < H)
    valid = valid_depth & valid_x & valid_y

    # Create normal map
    normals_cam = compute_normals_from_vertices(vertices_cam)
    # Convert from camera space normals to RGB
    # Camera space: -Z is forward, X right, Y down
    # Normal XYZ -> RGB (shifted to [0, 1])
    normals_rgb = (normals_cam + 1.0) / 2.0  # [-1, 1] -> [0, 1]

    # Rasterize to image
    normal_map = torch.ones(H, W, 3, device=device) * 1.0  # White background
    mask = torch.zeros(H, W, device=device)

    valid_verts = vertices_2d[valid].long()
    valid_normals = normals_rgb[valid]
    valid_depths = vertices_cam[valid, 2]

    # Z-buffer: keep closest point
    for i, (px, py) in enumerate(valid_verts):
        px, py = int(px), int(py)
        if py < H and px < W and valid_depths[i] < 10.0:  # reasonable depth
            if mask[py, px] == 0 or valid_depths[i] < mask[py, px]:
                normal_map[py, px] = valid_normals[i]
                mask[py, px] = valid_depths[i]

    # Set background to white (where mask is 0)
    bg_mask = mask == 0
    normal_map[bg_mask] = 1.0  # White

    return normal_map, (~bg_mask).float()


def extract_v3d_cano(checkpoint_path, device='cuda'):
    """Extract v3d_cano from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt['state_dict']

    # Find v3d_cano key
    v3d_key = None
    for key in state_dict.keys():
        if 'v3d_cano' in key:
            v3d_key = key
            break

    if v3d_key is None:
        raise ValueError("v3d_cano not found in checkpoint")

    v3d_cano = state_dict[v3d_key].to(device)
    logger.info(f"Loaded v3d_cano: {v3d_cano.shape}, bbox: [{v3d_cano.min():.4f}, {v3d_cano.max():.4f}]")
    return v3d_cano


def get_object_pose(model, batch, device='cuda'):
    """Extract object pose from model for current batch."""
    with torch.no_grad():
        # Forward pass to get object pose
        # The model should output object pose parameters
        out = model.inference_step(batch)

        # Try to extract object pose from output
        # Common keys: 'obj_pose', 'object_pose', 'pose_obj', 'T_obj'
        obj_pose = None
        for key in ['obj_pose', 'object_pose', 'pose_obj', 'T_obj', 'Rt_obj']:
            if key in out:
                obj_pose = out[key]
                break

        if obj_pose is None:
            # Try to get from model's internal state
            if hasattr(model, 'model') and hasattr(model.model, 'nodes'):
                nodes = model.model.nodes
                if 'object' in nodes:
                    obj_node = nodes['object']
                    if hasattr(obj_node, 'get_pose'):
                        obj_pose = obj_node.get_pose()
                    elif hasattr(obj_node, 'pose'):
                        obj_pose = obj_node.pose

        return obj_pose, out


def apply_pose_to_vertices(v3d_cano, pose):
    """
    Apply SE(3) pose to canonical vertices.
    v3d_cano: [N, 3]
    pose: [4, 4] or [3, 4] or dict with 'R' and 't'
    Returns: [N, 3] transformed vertices
    """
    if isinstance(pose, dict):
        R = pose['R']  # [3, 3]
        t = pose['t']  # [3]
        v3d_world = (R @ v3d_cano.T).T + t
    elif pose.shape[-2:] == (4, 4):
        # Homogeneous transformation
        v3d_h = torch.cat([v3d_cano, torch.ones(v3d_cano.shape[0], 1, device=v3d_cano.device)], dim=-1)
        v3d_world = (pose @ v3d_h.T).T[:, :3]
    elif pose.shape[-2:] == (3, 4):
        # [R|t] format
        R = pose[:, :3]
        t = pose[:, 3]
        v3d_world = (R @ v3d_cano.T).T + t
    else:
        raise ValueError(f"Unknown pose format: {pose.shape}")

    return v3d_world


def main():
    args, opt = parser_args()
    device = "cuda:0"

    # Fix test data path
    if hasattr(opt.dataset, 'dataset_path'):
        base_path = opt.dataset.dataset_path
        if not hasattr(opt.dataset, 'test'):
            from easydict import EasyDict
            opt.dataset.test = EasyDict()
        opt.dataset.test.data_dir = op.join(base_path, args.case)
        opt.dataset.test.seq_name = args.case

    # Setup output directories
    exp_key = args.load_ckpt.split("/")[1]
    output_dir = op.join("logs", exp_key, "test", "normals_v3d")

    hand_dir = op.join(output_dir, "hand")
    object_dir = op.join(output_dir, "object")
    combined_dir = op.join(output_dir, "combined")

    for d in [hand_dir, object_dir, combined_dir]:
        os.makedirs(d, exist_ok=True)

    logger.info(f"Test data dir: {opt.dataset.test.data_dir}")
    logger.info(f"Saving v3d-based normals to: {output_dir}")

    # Initialize model
    model = HOLD(opt, args)
    testset = create_dataset(opt.dataset.test, args)
    logger.info(f"Rendering {len(testset)} frames...")

    # Load checkpoint
    ckpt = torch.load(args.load_ckpt, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.to(device)
    model.eval()

    # Extract v3d_cano from checkpoint
    v3d_cano = extract_v3d_cano(args.load_ckpt, device)

    # Render loop
    reset_all_seeds(1)

    for idx, batch in enumerate(testset):
        with torch.no_grad():
            batch = thing.thing2dev(batch, device)

            # Get camera parameters from batch
            img_size = batch.get('img_size', [320, 240])  # Default if not present
            if isinstance(img_size, torch.Tensor):
                img_size = img_size.cpu().numpy().tolist()

            # Get intrinsics
            K = batch.get('K', batch.get('intrinsics'))
            if K is None:
                logger.warning(f"No intrinsics found for frame {idx}, skipping")
                continue

            # Get object pose
            obj_pose, out = get_object_pose(model, batch, device)

            if obj_pose is None:
                logger.warning(f"Could not extract object pose for frame {idx}")
                continue

            # Transform v3d to world space
            v3d_world = apply_pose_to_vertices(v3d_cano, obj_pose)

            # Get camera pose (world to camera)
            # Usually batch contains 'cam_pose' or 'extrinsics'
            cam_pose = batch.get('cam_pose', batch.get('extrinsics'))
            if cam_pose is None:
                # Default camera at origin looking at -Z
                cam_pose = torch.eye(4, device=device)

            # Render object normals
            obj_normal_map, obj_mask = render_point_cloud_normals(
                v3d_world, K, cam_pose, img_size, device
            )

            # Extract hand normals from model output (SDF-based, since hand works)
            hand_normal = None
            if "right.normal" in out:
                hand_norm = out["right.normal"]
                if isinstance(hand_norm, torch.Tensor):
                    # Reshape if needed
                    if hand_norm.dim() == 2:
                        if hand_norm.shape[0] == 3:
                            hand_norm = hand_norm.permute(1, 0)
                        H, W = img_size[0], img_size[1]
                        hand_norm = hand_norm.view(H, W, 3)
                    elif hand_norm.dim() == 3 and hand_norm.shape[0] == 3:
                        hand_norm = hand_norm.permute(1, 2, 0)
                    hand_normal = hand_norm

            # Save object normals
            obj_path = op.join(object_dir, f"frame_{idx:04d}.png")
            save_normal_image(obj_normal_map.cpu().numpy(), obj_path)

            # Save hand normals if available
            if hand_normal is not None:
                hand_path = op.join(hand_dir, f"frame_{idx:04d}.png")
                # Convert to [0, 1] range if in [-1, 1]
                hand_img = (hand_normal.cpu().numpy() + 1.0) / 2.0
                # White background where invalid
                hand_norm_mag = np.linalg.norm(hand_normal.cpu().numpy(), axis=-1)
                bg_mask = hand_norm_mag < 0.1
                hand_img[bg_mask] = 1.0  # White
                save_normal_image(hand_img, hand_path)

            # Combine hand and object normals
            # Simple combination: object where obj_mask > 0, hand where hand_mask > 0, white background
            if hand_normal is not None:
                combined = obj_normal_map.clone()
                hand_mask = ~bg_mask
                # Where hand is valid and object is not, use hand
                combined[hand_mask & (obj_mask.cpu().numpy() == 0)] = torch.from_numpy(
                    (hand_normal.cpu().numpy()[hand_mask & (obj_mask.cpu().numpy() == 0)] + 1.0) / 2.0
                ).float().to(device)

                combined_path = op.join(combined_dir, f"frame_{idx:04d}.png")
                save_normal_image(combined.cpu().numpy(), combined_path)
            else:
                # Just copy object to combined
                combined_path = op.join(combined_dir, f"frame_{idx:04d}.png")
                save_normal_image(obj_normal_map.cpu().numpy(), combined_path)

            if idx % 10 == 0:
                logger.info(f"Rendered frame {idx}/{len(testset)}")

    logger.info(f"\\nComplete! Saved v3d-based normal maps to {output_dir}")


if __name__ == "__main__":
    main()

"""
Usage:
export COMET_API_KEY="4hhuylWTxYQBirmxKwuwGv4Q5"
export COMET_WORKSPACE="cloudy"
python render_normals_v3d.py \
  --case hold_MC1_ho3d \
  --load_ckpt logs/69366de27_000010000/checkpoints/last.ckpt \
  --config confs/render_stage3_hold_MC1_ho3d_sds_from_official.yaml \
  --mute \
  --agent_id -1
"""
