import torch.nn as nn
import torch

import src.engine.volsdf_utils as volsdf_utils
from src.engine.rendering import render_color

from ...engine.density import LaplaceDensity
from ...engine.ray_sampler import ErrorBoundSampler
from ...networks.shape_net import ImplicitNet
from ...networks.texture_net import RenderingNet
from loguru import logger


class Node(nn.Module):
    def __init__(
        self,
        args,
        opt,
        specs,
        sdf_bounding_sphere,
        implicit_network_opt,
        rendering_network_opt,
        deformer,
        server,
        class_id,
        node_id,
        params,
    ):
        super(Node, self).__init__()
        self.args = args
        self.specs = specs
        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.implicit_network = ImplicitNet(implicit_network_opt, args, specs)
        self.rendering_network = RenderingNet(rendering_network_opt, args, specs)
        self.ray_sampler = ErrorBoundSampler(
            self.sdf_bounding_sphere, inverse_sphere_bg=True, **opt.ray_sampler
        )
        self.density = LaplaceDensity(**opt.density)
        self.deformer = deformer
        self.server = server
        self.class_id = class_id
        self.node_id = node_id
        self.params = params

    def meshing_cano(self, pose=None):
        return None

    def sample_points(self, input):
        raise NotImplementedError("Derived classes should implement this method.")

    def forward(self, input):
        if "time_code" in input:
            time_code = input["time_code"]
        else:
            time_code = None
        sample_dict = self.sample_points(input)

        # ✅ FIX: Guard against NaN sampled points
        if torch.isnan(sample_dict['points']).any():
            logger.info(f"[Node.forward] ⚠️ sample_dict['points'] has NaN, replacing with zeros")
            sample_dict['points'] = torch.nan_to_num(sample_dict['points'], nan=0.0)
        if 'z_vals' in sample_dict and torch.isnan(sample_dict['z_vals']).any():
            logger.info(f"[Node.forward] ⚠️ sample_dict['z_vals'] has NaN, replacing with 1.0")
            sample_dict['z_vals'] = torch.nan_to_num(sample_dict['z_vals'], nan=1.0)

        # ✅ PRESERVE VIEW-SPACE POINTS before any deformation
        view_points = sample_dict["points"]  # [B, N, 3] in camera/world space

        # ✅ ONLY apply deformer/canonicalization for object nodes
        if self.node_id == "object":
            # Object: query SDF in canonical space
            sdf_fn = self.implicit_network

            sdf_output, canonical_points, feature_vectors = volsdf_utils.sdf_func_with_deformer(
                self.deformer,
                sdf_fn,  # Use conditional function
                self.training,
                view_points.reshape(-1, 3),  # input is view-space
                sample_dict["deform_info"],
            )
            # Use canonical for SDF query, but view_points for rendering
            render_points = view_points
            sdf_query_points = canonical_points

        else:
            # Hand/MANO: stay in view space entirely
            cond = sample_dict["deform_info"]["cond"]
            output = self.implicit_network(view_points, cond)  # Pass [B, N, 3] directly
            sdf_output = output[..., :1]  # Gives [B, N, 1] - extracts SDF channel
            canonical_points = view_points.view(-1, 3)  # identity (no canonicalization)
            feature_vectors = None
            render_points = view_points
            sdf_query_points = view_points.reshape(-1, 3)

        # ✅ DEBUG
        logger.info(f"\n[Node.forward] After processing:")
        logger.info(f"  sdf_output has_nan: {torch.isnan(sdf_output).any().item()}")
        logger.info(f"  canonical_points has_nan: {torch.isnan(canonical_points).any().item()}")

        num_samples = sample_dict["z_vals"].shape[1]

        # ✅ Pass both view_points (for shading) and sdf_query_points (for SDF)
        color, normal, semantics = self.render(
            sample_dict,
            num_samples,
            render_points,        # view space for rendering
            sdf_query_points,     # canonical for object, view for hand
            feature_vectors,
            time_code
        )
        self.device = color.device

        density = self.density(sdf_output).view(-1, num_samples, 1)

        # Store both for downstream use
        sample_dict["canonical_pts"] = canonical_points.view(
            sample_dict["batch_size"], -1, num_samples, 3
        )
        sample_dict["view_pts"] = view_points.view(
            sample_dict["batch_size"], -1, num_samples, 3
        )

        factors = {
            "color": color,
            "normal": normal,
            "density": density,
            "semantics": semantics,
            "z_vals": sample_dict["z_vals"],
        }

        return factors, sample_dict

    def render(
        self, sample_dict, num_samples, view_points, sdf_query_points, feature_vectors, time_code
    ):
        color, normal, semantics = render_color(
            self.deformer,
            self.implicit_network,
            self.rendering_network,
            sample_dict["ray_dirs"],
            sample_dict["cond"],
            sample_dict["tfs"],
            view_points,           # ✅ view space for shading
            sdf_query_points,      # canonical for object SDF
            feature_vectors,
            self.training,
            num_samples,
            self.class_id if hasattr(self, 'class_id') else 0,
            time_code,
        )
        return color, normal, semantics

    def step_embedding(self):
        # ✅ Check if embedder exists (multires=0 has no embedder)
        if hasattr(self.implicit_network, 'embedder_obj') and self.implicit_network.embedder_obj is not None:
            self.implicit_network.embedder_obj.step()

    def query_sdf_grid(self, x_c, cond):
        """Query SDF from stored grid using trilinear interpolation."""
        sdf_grid = self.server.object_model.sdf_grid  # [1, D, H, W] or [D, H, W]

        # Handle shape variations
        if sdf_grid.dim() == 3:
            sdf_grid = sdf_grid.unsqueeze(0)  # [1, D, H, W]

        # Store original shape (all dims except the last coordinate dimension)
        original_shape = x_c.shape[:-1]
        N = x_c.shape[0] if x_c.dim() == 2 else x_c.shape[0] * x_c.shape[1]

        # Normalize canonical coordinates to [-1, 1] for grid_sample
        if hasattr(self.server.object_model, 'norm_mat'):
            # Use the stored normalization matrix
            x_c_h = torch.cat([x_c, torch.ones(*x_c.shape[:-1], 1, device=x_c.device)], dim=-1)
            # Use ... to handle both 2D [N, 4] and 3D [B, N, 4] cases
            x_c_norm = torch.matmul(x_c_h, self.server.object_model.norm_mat.T)[..., :3]
        else:
            # Fallback: no normalization
            x_c_norm = x_c

        # Flatten for grid_sample: need [1, 1, N, 1, 3]
        x_c_flat = x_c_norm.view(1, 1, N, 1, 3)  # [1, 1, N, 1, 3]

        # Interpolate
        sdf_vals = torch.nn.functional.grid_sample(
            sdf_grid.unsqueeze(0),  # [1, 1, D, H, W]
            x_c_flat,
            align_corners=True,
            mode='bilinear',
            padding_mode='border',
        )  # Output: [1, 1, 1, 1, N]

        # Reshape to [batch=1, N, 1] to match expected 3D format
        sdf_vals = sdf_vals.view(1, N, 1)  # [1, N, SDF_dim]

        # Create dummy feature vector
        feature_dim = self.implicit_network.opt.feature_vector_size
        features = torch.zeros(1, N, feature_dim, device=x_c.device)

        # Concatenate: [1, N, output_dim]
        output = torch.cat([sdf_vals, features], dim=-1)
        return output
