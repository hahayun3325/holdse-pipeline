#!/usr/bin/env python
import os
import glob
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger

from pytorch3d.loss import chamfer_distance

# If your project is a package, adjust these imports accordingly
from src.model.obj.object_model import GeometryEncoder3D   # same class used in ObjectModel


# -----------------------------
# Simple implicit SDF decoder
# -----------------------------
class ImplicitSDFDecoder(nn.Module):
    """
    Simple implicit SDF decoder used only for offline pretraining.

    Input:
        x: [B, N, 3] canonical points in [-1, 1]^3
        z: [B, D] geometry latent from GeometryEncoder3D

    We concatenate x and broadcasted z, then apply an MLP → SDF value.
    (The main HOLD pipeline will use a FiLM-based implicit_network;
     this decoder is only to teach the encoder a good geometry prior.)
    """
    def __init__(self, latent_dim=128, hidden_dim=128):
        super().__init__()
        in_dim = 3 + latent_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, z):
        # x: [B, N, 3], z: [B, D]
        B, N, _ = x.shape
        z_expanded = z.unsqueeze(1).expand(B, N, -1)  # [B, N, D]
        inp = torch.cat([x, z_expanded], dim=-1)      # [B, N, 3+D]
        sdf = self.mlp(inp)                           # [B, N, 1]
        return sdf


# -----------------------------
# Utility: mesh → SDF grid 64³
# -----------------------------
def mesh_to_sdf_grid(vertices, grid_resolution=64, padding=0.1, device="cuda"):
    """
    Convert a single mesh (vertices [N,3]) to an approximate signed distance
    field on a regular 64³ grid in canonical space [-1,1]³.
    This mirrors _mesh_to_sdf_grid in HOLD and is used as encoder input.
    """
    if vertices is None or vertices.shape[0] == 0:
        logger.error("[Pretrain] Cannot voxelize empty mesh")
        return None

    vertices = vertices.to(device)

    # Normalize to [-1, 1]³
    v_min = vertices.min(dim=0)[0]
    v_max = vertices.max(dim=0)[0]
    v_center = (v_min + v_max) / 2
    v_scale = (v_max - v_min).max() * (1 + padding)
    v_scale = torch.clamp(v_scale, min=1e-6)

    vertices_norm = (vertices - v_center) / (v_scale / 2)

    # Grid in [-1,1]³
    grid_coords = torch.linspace(-1, 1, grid_resolution, device=device)
    gx, gy, gz = torch.meshgrid(grid_coords, grid_coords, grid_coords)
    grid_points = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)

    # Unsigned distances (batched)
    batch_size = 5000
    dists_all = []
    for i in range(0, grid_points.shape[0], batch_size):
        pts = grid_points[i:i + batch_size]
        d = torch.cdist(pts, vertices_norm)
        d_min = d.min(dim=1)[0]
        dists_all.append(d_min)
    unsigned = torch.cat(dists_all, dim=0).reshape(
        grid_resolution, grid_resolution, grid_resolution
    )

    # Coarse sign via centroid heuristic
    centroid = vertices_norm.mean(dim=0)
    grid_to_centroid = torch.norm(grid_points - centroid, dim=1).reshape(
        grid_resolution, grid_resolution, grid_resolution
    )
    median_dist = grid_to_centroid.median()
    is_inside = grid_to_centroid < median_dist * 0.7

    signed = unsigned.clone()
    signed[is_inside] = -signed[is_inside]

    # For encoder: [1,1,R,R,R]
    sdf_grid = signed.unsqueeze(0).unsqueeze(0).contiguous()
    return sdf_grid  # [1,1,64,64,64]


# -----------------------------
# Dataset: load template meshes
# -----------------------------
class TemplateMeshDataset(torch.utils.data.Dataset):
    """
    Very simple dataset that loads template/object meshes.

    Assumes each .pt file under templates_dir contains a tensor [N,3]
    of canonical vertices (like HOLD's object templates).
    """
    def __init__(self, templates_dir):
        super().__init__()
        self.files = sorted(glob.glob(os.path.join(templates_dir, "*.pt")))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .pt templates found in {templates_dir}")
        logger.info(f"[Pretrain] Found {len(self.files)} templates in {templates_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        verts = torch.load(path)  # adapt if you store dicts
        if isinstance(verts, dict):
            # e.g. {'verts': tensor}
            verts = verts.get("verts", None)
        if verts is None:
            raise ValueError(f"Template {path} has no vertices")
        return verts.float()


# -----------------------------
# Loss helpers
# -----------------------------
def eikonal_loss(sdf, x):
    """
    Eikonal loss: ||∇_x sdf||_2 − 1 → 0.
    sdf: [B, N, 1], x: [B, N, 3]
    """
    grad = torch.autograd.grad(
        outputs=sdf.sum(),
        inputs=x,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]  # [B, N, 3]
    grad_norm = torch.linalg.norm(grad, dim=-1)
    return ((grad_norm - 1.0) ** 2).mean()


def normal_consistency_loss(pred_verts, pred_faces):
    """
    Optional: simple mesh normal smoothness using face normals.
    For brevity, we use a basic neighbor-normal penalty.
    You can replace with pytorch3d.mesh_normal_consistency if desired.
    """
    try:
        from pytorch3d.structures import Meshes
        from pytorch3d.loss import mesh_normal_consistency

        mesh = Meshes(verts=[pred_verts], faces=[pred_faces])
        return mesh_normal_consistency(mesh)
    except Exception as e:
        logger.warning(f"[Pretrain] normal_consistency skipped: {e}")
        return pred_verts.new_tensor(0.0)


# -----------------------------
# Main training loop
# -----------------------------
def train_encoder(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Dataset & loader
    dataset = TemplateMeshDataset(args.templates_dir)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
    )

    # Models: encoder + temporary decoder
    encoder = GeometryEncoder3D(
        in_channels=1,
        base_channels=32,
        latent_dim=args.latent_dim,
    ).to(device)

    decoder = ImplicitSDFDecoder(
        latent_dim=args.latent_dim,
        hidden_dim=128,
    ).to(device)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=args.lr)

    logger.info(f"[Pretrain] Encoder params: {sum(p.numel() for p in encoder.parameters())}")
    logger.info(f"[Pretrain] Decoder params: {sum(p.numel() for p in decoder.parameters())}")

    for epoch in range(args.epochs):
        encoder.train()
        decoder.train()

        running_loss = 0.0
        for verts in loader:
            verts = verts.to(device)  # [N,3] per sample; batch of templates is small
            optimizer.zero_grad()

            # For simplicity, treat each template independently in batch=1
            # (you can extend to real batching if needed)
            if verts.ndim == 3:
                # [B, N, 3] → loop; here we keep B=1 for clarity
                verts = verts[0]

            # 1) Build voxelized SDF grid 64³ as encoder input
            sdf_grid = mesh_to_sdf_grid(
                vertices=verts,
                grid_resolution=64,
                padding=0.1,
                device=device,
            )  # [1,1,64,64,64]

            # 2) Encode to latent z_geo
            z_geo = encoder(sdf_grid)  # [1, D]

            # 3) Sample canonical query points x in [-1,1]³ for implicit decoder
            num_points = args.num_points
            x = torch.rand(1, num_points, 3, device=device) * 2.0 - 1.0  # [1,N,3]

            # 4) Decode SDF
            x.requires_grad_(True)
            sdf_pred = decoder(x, z_geo)  # [1,N,1]

            # 5) Build a coarse ground-truth SDF at these points from the voxel grid
            #    (trilinear interpolation using grid_sample)
            #    sdf_grid: [1,1,D,H,W] with D=H=W=64 in [-1,1]³
            #    coords for grid_sample: [B,N,1,1,3] in [-1,1]
            coords = x[..., [2, 1, 0]].unsqueeze(2).unsqueeze(2)  # [1,N,1,1,3], zyx
            sdf_gt = torch.nn.functional.grid_sample(
                sdf_grid,
                coords,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            ).view(1, num_points, 1)  # [1,N,1]

            # 6) Losses
            # SDF reconstruction (L1)
            loss_sdf = (sdf_pred - sdf_gt).abs().mean()

            # Eikonal
            loss_eik = eikonal_loss(sdf_pred, x)

            # Optional Chamfer: compare decoded zero-level set to template verts
            # For efficiency, we only evaluate on a low-res grid here.
            # (You can increase resolution or use marching cubes if desired.)
            with torch.no_grad():
                # Low-res grid for prediction
                res_chamfer = 32
                coords_c = torch.linspace(-1, 1, res_chamfer, device=device)
                gx, gy, gz = torch.meshgrid(coords_c, coords_c, coords_c)
                pts_grid = torch.stack([gx, gy, gz], dim=-1).reshape(1, -1, 3)  # [1,M,3]
                sdf_grid_pred = decoder(pts_grid, z_geo)  # [1,M,1]
                # Take near-surface points as predicted "point cloud"
                mask = sdf_grid_pred[0, :, 0].abs() < 0.02
                pts_pred = pts_grid[0, mask] if mask.any() else pts_grid[0]

            # Normalize verts to [-1,1]³ as ground-truth points
            v_min = verts.min(dim=0)[0]
            v_max = verts.max(dim=0)[0]
            v_center = (v_min + v_max) / 2
            v_scale = (v_max - v_min).max() * 1.001
            v_scale = torch.clamp(v_scale, min=1e-6)
            verts_norm = (verts - v_center) / (v_scale / 2)

            # Chamfer distance between predicted surface samples and GT surface points
            chamfer_loss, _ = chamfer_distance(
                pts_pred.unsqueeze(0),      # [1, P, 3]
                verts_norm.unsqueeze(0),    # [1, Q, 3]
            )

            # Optional simple normal smoothness on predicted surface (not GT normals)
            loss_normal = normal_consistency_loss(
                pred_verts=pts_pred.detach(),   # detach to avoid 2nd grad
                pred_faces=torch.empty(0, 3, dtype=torch.long, device=device),  # dummy
            )
            # If normal loss fails / dummy, it will be ~0.

            # Total loss
            total_loss = (
                args.w_chamfer * chamfer_loss
                + args.w_eikonal * loss_eik
                + args.w_sdf * loss_sdf
                + args.w_normal * loss_normal
            )

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        avg_loss = running_loss / max(len(loader), 1)
        logger.info(
            f"[Pretrain] Epoch {epoch+1}/{args.epochs} "
            f"avg_loss={avg_loss:.6f}"
        )

        # TODO: you can periodically evaluate cd_icp / f10_icp if you have ICP code.

    # Save only encoder weights
    os.makedirs(os.path.dirname(args.out_ckpt), exist_ok=True)
    torch.save(
        {"encoder_state_dict": encoder.state_dict()},
        args.out_ckpt,
    )
    logger.info(f"[Pretrain] Saved encoder checkpoint to {args.out_ckpt}")


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain GeometryEncoder3D on templates")
    parser.add_argument("--templates_dir", type=str, required=True,
                        help="Directory with .pt template meshes (verts [N,3])")
    parser.add_argument("--out_ckpt", type=str, required=True,
                        help="Path to save pretrained encoder checkpoint")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_points", type=int, default=8192)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--w_chamfer", type=float, default=1.0)
    parser.add_argument("--w_eikonal", type=float, default=0.1)
    parser.add_argument("--w_sdf", type=float, default=1.0)
    parser.add_argument("--w_normal", type=float, default=0.01)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-pin-memory", dest="no_pin_memory", action="store_true")
    parser.set_defaults(no_pin_memory=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_encoder(args)

'''
python pretrain_geometry_encoder.py \
  --templates_dir data/templates \
  --out_ckpt pretrained/geo_encoder.pt \
  --epochs 50 \
  --latent_dim 128
  
  
  mkdir -p pretrained

python pretrain_geometry_encoder.py \
  --templates_dir data/templates \
  --out_ckpt pretrained/geo_encoder.pt \
  --epochs 50 \
  --batch_size 1 \
  --num_workers 4 \
  --num_points 8192 \
  --latent_dim 128 \
  --lr 1e-4

'''