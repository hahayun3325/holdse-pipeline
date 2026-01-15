#!/usr/bin/env python3
"""Extract predictions using HOLD's official methodology."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from glob import glob
import os.path as op
from common.xdict import xdict
from src.model.mano.server import MANOServer
from src.model.obj.server import ObjectServer
from src.utils.eval_modules import compute_bounding_box_centers


def map_deform2eval(verts, scale, normalize_shift):
    """Transform from deformation space to evaluation space."""
    conversion_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    _normalize_shift = normalize_shift.copy()
    _normalize_shift[0] *= -1

    src_verts = np.copy(verts)
    src_verts = np.dot(src_verts, conversion_matrix)
    src_verts *= scale
    src_verts += _normalize_shift
    return src_verts


def load_data_hold_style(ckpt_path):
    """
    Load predictions from checkpoint using HOLD's exact methodology.

    This replicates src/utils/io/ours.py from hold-master.
    """
    device = "cuda:0"
    print(f"Loading checkpoint: {ckpt_path}")

    # Load checkpoint
    data = torch.load(ckpt_path, map_location="cpu")
    sd = xdict(data["state_dict"])

    # Find misc folder - FIX path handling for absolute paths
    ckpt_dir = Path(ckpt_path).parent.parent
    misc_dir = ckpt_dir / "misc"

    if not misc_dir.exists():
        raise FileNotFoundError(f"misc/ not found at {misc_dir}")

    misc_ps = sorted(glob(str(misc_dir / "*.npy")))
    if not misc_ps:
        raise FileNotFoundError(f"No .npy files in {misc_dir}")

    misc = np.load(misc_ps[-1], allow_pickle=True).item()

    # Extract metadata
    fnames = misc["img_paths"]
    K = torch.FloatTensor(misc["K"]).to(device).view(1, 4, 4)[:, :3, :3]
    scale = misc["scale"]
    scale = torch.tensor([scale]).float().to(device)
    mesh_c_o = misc.get("mesh_c_o") or misc.get("object_cano")

    # Extract node IDs
    node_ids = []
    for key in sd.keys():
        if ".nodes." not in key:
            continue
        node_id = key.split(".")[2]
        node_ids.append(node_id)
    node_ids = list(set(node_ids))

    # Extract parameters for each node
    params = {}
    for node_id in node_ids:
        params[node_id] = sd.search(".params.").search(node_id)
        params[node_id]["scene_scale"] = scale
        params[node_id] = params[node_id].to(device)

    # Get object scale if present
    scale_key = "model.nodes.object.server.object_model.obj_scale"
    obj_scale = sd[scale_key] if scale_key in sd.keys() else None

    # Get sequence name
    seq_name = fnames[0].split("/")[2]

    # Create servers
    servers = {}
    faces = {}
    for node_id in node_ids:
        if "right" in node_id or "left" in node_id:
            hand = "right" if "right" in node_id else "left"
            is_right = hand == "right"
            server = MANOServer(betas=None, is_rhand=is_right).to(device)
            myfaces = torch.LongTensor(server.faces.astype(np.int64)).to(device)
        elif "object" in node_id:
            server = ObjectServer(seq_name, template=mesh_c_o)
            server.object_model.obj_scale = obj_scale
            server.to(device)
            myfaces = torch.LongTensor(mesh_c_o.faces).to(device)
        else:
            raise ValueError(f"Unknown node id: {node_id}")

        servers[node_id] = server
        faces[node_id] = myfaces

    if obj_scale is not None:
        servers["object"].object_model.obj_scale = obj_scale.to(device)

    # Forward pass through servers
    out = xdict()
    for node_id in node_ids:
        out.merge(
            xdict(servers[node_id].forward_param(params[node_id])).postfix(f".{node_id}")
        )

    # ==================== CORRECTED STRUCTURE ====================
    # Subset joints to HO3D's 16-joint format BEFORE mapping
    HO3D_JOINT_INDICES = [0, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]

    for node_id in node_ids:
        if 'right' in node_id or 'left' in node_id:
            joint_key = f'jnts.{node_id}'
            if joint_key in out.keys():
                original_joints = out[joint_key]
                print(f"✓ Subsetting {joint_key}: {original_joints.shape[1]} → 16 joints")
                # Delete and reassign to avoid xdict error
                del out[joint_key]
                out[joint_key] = original_joints[:, HO3D_JOINT_INDICES, :]

    # Map to evaluation space (helper function)
    def map_deform2eval_batch(verts, inverse_scale, normalize_shift):
        return np.array([
            map_deform2eval(v, inverse_scale, normalize_shift)
            for v in verts.cpu().detach().numpy()
        ])

    # Load dataset for normalize_shift
    dataset = np.load(f"data/{seq_name}/build/data.npy", allow_pickle=True).item()
    normalize_shift = dataset["normalize_shift"]

    # Map predictions to evaluation space
    inverse_scale = float(1.0 / scale[0])
    print(f"✓ Applying coordinate transformation (inverse_scale={inverse_scale:.6f})")

    for key, val in out.search("verts.").items():
        out[key.replace("verts.", "v3d_c.")] = map_deform2eval_batch(
            val, inverse_scale, normalize_shift
        )
    for key, val in out.search("jnts.").items():
        out[key.replace("jnts.", "j3d_c.")] = map_deform2eval_batch(
            val, inverse_scale, normalize_shift
        )

    # Compute root-relative coordinates (from 16 joints)
    print(f"✓ Computing root-aligned coordinates")
    for key, val in out.search("j3d_c.").items():
        out[key.replace("j3d_c.", "root.")] = val[:, :1].squeeze(1)
        out[key.replace("j3d_c.", "j3d_ra.")] = val - val[:, :1]

    # Object root (bounding box center)
    out["root.object"] = compute_bounding_box_centers(out["v3d_c.object"])
    out["v3d_ra.object"] = out["v3d_c.object"] - out["root.object"][:, None, :]

    # Object relative to hand
    out["v3d_right.object"] = out["v3d_c.object"] - out["root.right"][:, None, :]
    if "root.left" in out.keys():
        out["v3d_left.object"] = out["v3d_c.object"] - out["root.left"][:, None, :]

    # Package output
    out_dict = xdict()
    out_dict["fnames"] = fnames
    out_dict.merge(out)
    out_dict["faces"] = faces
    out_dict["servers"] = servers
    out_dict["K"] = K.cpu().numpy()
    out_dict["full_seq_name"] = seq_name
    out_dict["is_valid"] = torch.ones(len(fnames), dtype=torch.bool)

    print(f"✓ Loaded {len(fnames)} frames")
    out_dict = out_dict.to_torch()
    return out_dict


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path')
    parser.add_argument('--output', required=True, help='Output .pkl path')
    args = parser.parse_args()

    predictions = load_data_hold_style(args.checkpoint)

    # Save in compatible format
    torch.save(predictions, args.output)
    print(f"\n✓ Saved predictions to: {args.output}")


if __name__ == '__main__':
    main()

'''
Here's the correct extraction command:
# Extract using HOLD's methodology
python scripts/extract_from_checkpoint_hold_style.py \
    --checkpoint /home/fredcui/Projects/hold/code/logs/cb20a1702/checkpoints/last.ckpt \
    --output logs/evaluation_results/MC1_official_hold_style_predictions_$(date +%Y%m%d_%H%M%S).pkl \
    2>&1 | tee logs/evaluation_results/MC1_official_hold_style_extraction_$(date +%Y%m%d_%H%M%S).log
'''