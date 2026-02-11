import sys

import torch
import torch.nn as nn
import numpy as np

sys.path = [".."] + sys.path
from common.rot import axis_angle_to_matrix
from loguru import logger

class ObjectModel(nn.Module):
    def __init__(self, seq_name, template=None, grid_res=64):
        super(ObjectModel, self).__init__()

        data = np.load(f"./data/{seq_name}/build/data.npy", allow_pickle=True).item()["entities"]["object"]
        if template is None:
            v3d_cano = torch.FloatTensor(data["pts.cano"])
        else:
            v3d_cano = torch.FloatTensor(template.vertices)
        # Debug: check canonical vertices for non-finite values
        if not torch.isfinite(v3d_cano).all():
            logger.error(
                "[ObjectModel] v3d_cano contains non-finite values: "
                f"any_nan={torch.isnan(v3d_cano).any().item()}, "
                f"any_inf={torch.isinf(v3d_cano).any().item()}"
            )
            v3d_cano_finite = torch.nan_to_num(v3d_cano)
            logger.error(
                "[ObjectModel] v3d_cano finite stats: "
                f"min={v3d_cano_finite.min().item():.4e}, "
                f"max={v3d_cano_finite.max().item():.4e}, "
                f"std={v3d_cano_finite.std().item():.4e}"
            )
        else:
            logger.info(
                "[ObjectModel] v3d_cano stats: "
                f"min={v3d_cano.min().item():.4e}, "
                f"max={v3d_cano.max().item():.4e}, "
                f"mean={v3d_cano.mean().item():.4e}, "
                f"std={v3d_cano.std().item():.4e}"
            )

        # Keep scale and normalization as non-trainable buffers
        self.register_buffer(
            "obj_scale", torch.FloatTensor(np.array([data["obj_scale"]]))
        )
        # self.register_buffer("v3d_cano", v3d_cano)
        # ✅ Make canonical object vertices learnable
        self.v3d_cano = nn.Parameter(v3d_cano, requires_grad=True)

        # Initialize SDF grid from vertices
        with torch.no_grad():
            logger.info(
             f"[ObjectModel] Initializing SDF grid from vertices; grid_res={grid_res}, "
             f"v3d_cano.shape={self.v3d_cano.shape}"
            )
            sdf_grid = self._initialize_sdf_from_vertices(self.v3d_cano.detach(), grid_res)

            # Debug: check SDF grid for non-finite values BEFORE unsqueeze/Parameter
            if not torch.isfinite(sdf_grid).all():
                logger.error("[ObjectModel] SDF grid contains non-finite values right after init!")
                logger.error(
                    f"  any_nan={torch.isnan(sdf_grid).any().item()}, "
                    f"any_inf={torch.isinf(sdf_grid).any().item()}"
                )
                sdf_finite = torch.nan_to_num(sdf_grid)
                logger.error(
                    "[ObjectModel] SDF finite stats: "
                    f"min={sdf_finite.min().item():.4e}, "
                    f"max={sdf_finite.max().item():.4e}, "
                    f"mean={sdf_finite.mean().item():.4e}, "
                    f"std={sdf_finite.std().item():.4e}"
                )
            else:
                logger.info(
                    "[ObjectModel] SDF grid stats after init: "
                    f"shape={sdf_grid.shape}, "
                    f"min={sdf_grid.min().item():.4e}, "
                    f"max={sdf_grid.max().item():.4e}, "
                    f"mean={sdf_grid.mean().item():.4e}, "
                    f"std={sdf_grid.std().item():.4e}"
                )

            # Add batch dimension so shape is [1, D, H, W]
            sdf_grid = sdf_grid.unsqueeze(0)

        self.sdf_grid = nn.Parameter(sdf_grid, requires_grad=True)

        self.register_buffer("norm_mat", torch.FloatTensor(data["norm_mat"]))
        self.register_buffer("denorm_mat", torch.inverse(self.norm_mat))

    def forward(self, rot, trans, scene_scale=None):
        device = self.v3d_cano.device

        batch_size = rot.shape[0]
        if scene_scale is None:
            scene_scale = torch.ones(batch_size).to(device)
        else:
            scene_scale = scene_scale.view(batch_size)
        rot_mat = axis_angle_to_matrix(rot).view(batch_size, 3, 3)

        # cano to camera
        batch_size = rot_mat.shape[0]
        tf_mats = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        tf_mats[:, :3, :3] = rot_mat
        tf_mats[:, :3, 3] = trans.view(batch_size, 3)

        scale_mat = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        scale_mat *= scene_scale[:, None, None]
        scale_mat[:, 3, 3] = 1
        v3d_cano_pad = torch.cat(
            [self.v3d_cano, torch.ones(self.v3d_cano.shape[0], 1, device=device)],
            dim=1,
        )
        v3d_cano_pad = v3d_cano_pad[None, :, :].repeat(batch_size, 1, 1)

        obj_scale = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        obj_scale *= self.obj_scale
        obj_scale[:, 3, 3] = 1

        # deformalize
        tf_mats = torch.matmul(scale_mat, tf_mats)
        tf_mats = torch.matmul(tf_mats, obj_scale)
        tf_mats = torch.matmul(
            tf_mats, self.denorm_mat[None, :, :].repeat(batch_size, 1, 1)
        )

        vertices = torch.bmm(tf_mats, v3d_cano_pad.permute(0, 2, 1)).permute(0, 2, 1)
        verts = vertices[:, :, :3] / vertices[:, :, 3:4]
        out = {}
        out["vertices"] = verts
        out["T"] = tf_mats
        return out

    def _initialize_sdf_from_vertices(self, vertices, grid_res=64,
                                      max_verts_for_init: int = 5000,
                                      gp_chunk: int = 8192,
                                      v_chunk: int = 4096):
        """
        Initialize an SDF grid from a vertex point cloud in a memory-safe way.

        vertices: [N, 3] tensor (CPU or GPU)
        Returns: [grid_res, grid_res, grid_res] float32 tensor on same device.
        """
        device = vertices.device
        verts = vertices.float()

        # Optional: subsample very dense meshes to keep cost reasonable
        # (64^3 grid doesn't need all ~70k verts to get a decent SDF)
        if verts.shape[0] > max_verts_for_init:
            idx = torch.randperm(verts.shape[0], device=device)[:max_verts_for_init]
            verts = verts[idx]

        # Bounding box with small padding
        bbox_min = verts.min(dim=0)[0] - 0.05
        bbox_max = verts.max(dim=0)[0] + 0.05

        # Coordinate grid
        xs = torch.linspace(bbox_min[0], bbox_max[0], grid_res, device=device)
        ys = torch.linspace(bbox_min[1], bbox_max[1], grid_res, device=device)
        zs = torch.linspace(bbox_min[2], bbox_max[2], grid_res, device=device)

        # Older PyTorch (1.8.x) does not support indexing="ij"
        # Default behavior is already "ij"-style for multiple 1D inputs.
        grid_x, grid_y, grid_z = torch.meshgrid(xs, ys, zs)

        grid_points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)  # [res^3, 3]
        M = grid_points.shape[0]
        N = verts.shape[0]

        # Compute unsigned distance to nearest vertex in chunks:
        #   - chunk over grid_points (gp_chunk)
        #   - and over verts (v_chunk)
        min_dists_flat = torch.full((M,), float("inf"), device=device)

        for i in range(0, M, gp_chunk):
            gp_chunk_pts = grid_points[i:i + gp_chunk]                     # [m, 3]
            m = gp_chunk_pts.shape[0]
            chunk_min = torch.full((m,), float("inf"), device=device)

            for j in range(0, N, v_chunk):
                v_chunk_pts = verts[j:j + v_chunk]                         # [n, 3]
                # [m, n] distances for this block
                d_block = torch.cdist(gp_chunk_pts, v_chunk_pts)
                # Update minimum over vertices seen so far
                chunk_min = torch.minimum(chunk_min, d_block.min(dim=1)[0])

            min_dists_flat[i:i + m] = chunk_min

        min_dists = min_dists_flat.reshape(grid_res, grid_res, grid_res)

        # Simple signed heuristic: negative near object center, positive far away
        center = verts.mean(dim=0, keepdim=True)            # [1, 3]
        center_dists = torch.norm(grid_points - center, dim=-1)
        center_dists = center_dists.reshape(grid_res, grid_res, grid_res)

        thresh = center_dists.median()
        inside_mask = center_dists <= thresh

        sdf = torch.where(inside_mask, -min_dists, min_dists)

        return sdf