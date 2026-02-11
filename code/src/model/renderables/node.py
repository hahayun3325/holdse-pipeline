import torch.nn as nn
import torch

import src.engine.volsdf_utils as volsdf_utils
from src.engine.rendering import render_color

from ...engine.density import LaplaceDensity
from ...engine.ray_sampler import ErrorBoundSampler
from ...networks.shape_net import ImplicitNet
from ...networks.texture_net import RenderingNet


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
            print(f"[Node.forward] ⚠️ sample_dict['points'] has NaN, replacing with zeros")
            sample_dict['points'] = torch.nan_to_num(sample_dict['points'], nan=0.0)
        if 'z_vals' in sample_dict and torch.isnan(sample_dict['z_vals']).any():
            print(f"[Node.forward] ⚠️ sample_dict['z_vals'] has NaN, replacing with 1.0")
            sample_dict['z_vals'] = torch.nan_to_num(sample_dict['z_vals'], nan=1.0)

        # ✅ NEW DEBUG: Check deform_info structure
        print(f"\n[Node.forward] {self.node_id} - About to call sdf_func_with_deformer:")
        print(f"  points shape: {sample_dict['points'].shape}")
        if 'deform_info' in sample_dict and sample_dict['deform_info'] is not None:
            deform_info = sample_dict['deform_info']
            print(f"  deform_info type: {type(deform_info)}")
            if isinstance(deform_info, dict):
                print(f"  deform_info keys: {list(deform_info.keys())}")
                for k, v in deform_info.items():
                    if isinstance(v, torch.Tensor):
                        print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
                    elif isinstance(v, dict):
                        print(f"    {k}: (nested dict with keys {list(v.keys())})")
                    else:
                        print(f"    {k}: type={type(v)}")
            else:
                print(f"  deform_info: {deform_info}")
        else:
            print(f"  ⚠️  deform_info is None or missing!")
        # In Node.forward, before calling sdf_func_with_deformer:
        # Choose SDF source based on training mode
        if (not self.training and
            self.node_id == "object" and  # <-- CRITICAL: Only for object node
            hasattr(self, 'server') and
            hasattr(self.server.object_model, 'sdf_grid')):
            # Inference: use optimized SDF grid (bypasses drifted implicit network)
            sdf_fn = lambda x_c, cond: self.query_sdf_grid(x_c, cond)
        else:
            # Training: keep original behavior (implicit network)
            sdf_fn = self.implicit_network

        # Then call with the selected function:
        sdf_output, canonical_points, feature_vectors = volsdf_utils.sdf_func_with_deformer(
            self.deformer,
            sdf_fn,  # Use conditional function instead of hard-coded self.implicit_network
            self.training,
            sample_dict["points"].reshape(-1, 3),
            sample_dict["deform_info"],
        )

        # ✅ NEW DEBUG: Check immediately after sdf_func_with_deformer
        print(f"\n[Node.forward] After sdf_func_with_deformer:")
        print(f"  sdf_output has_nan: {torch.isnan(sdf_output).any().item()}")
        print(f"  canonical_points has_nan: {torch.isnan(canonical_points).any().item()}")
        print(f"  feature_vectors has_nan: {torch.isnan(feature_vectors).any().item()}")
        if torch.isnan(canonical_points).any():
            print(f"  ❌ canonical_points is NaN right after deformer!")
            # Also check input points
            print(f"  input points (sample_dict['points']) has_nan: {torch.isnan(sample_dict['points']).any().item()}")
        # After line ~100 where sdf_output is computed
        print(f"\n[Node.forward] {self.node_id} SDF statistics:")
        print(f"  sdf_output shape: {sdf_output.shape}")
        print(f"  sdf min/max: {sdf_output.min().item():.4f} / {sdf_output.max().item():.4f}")
        print(f"  sdf mean: {sdf_output.mean().item():.4f}")
        print(f"  sdf std: {sdf_output.std().item():.4f}")

        # Count how many points are inside (SDF < 0) vs outside (SDF > 0)
        inside_count = (sdf_output < 0).sum().item()
        outside_count = (sdf_output > 0).sum().item()
        print(f"  Points inside surface (SDF<0): {inside_count}")
        print(f"  Points outside surface (SDF>0): {outside_count}")
        num_samples = sample_dict["z_vals"].shape[1]
        color, normal, semantics = self.render(
            sample_dict, num_samples, canonical_points, feature_vectors, time_code
        )
        self.device = color.device

        num_samples = color.shape[1]
        density = self.density(sdf_output).view(-1, num_samples, 1)
        print(f"\n[Node.forward] {self.node_id} Density statistics:")
        print(f"  density shape: {density.shape}")
        print(f"  density min/max: {density.min().item():.6f} / {density.max().item():.6f}")
        print(f"  density mean: {density.mean().item():.6f}")
        if density.max().item() < 0.001:
            print(f"  ⚠️  Density is near-zero! Object will be invisible!")
        sample_dict["canonical_pts"] = canonical_points.view(
            sample_dict["batch_size"], sample_dict["num_pixels"], num_samples, 3
        )
        # color, normal, density, semantics
        factors = {
            "color": color,
            "normal": normal,
            "density": density,
            "semantics": semantics,
            "z_vals": sample_dict["z_vals"],
        }

        return factors, sample_dict

    def render(
        self, sample_dict, num_samples, canonical_points, feature_vectors, time_code
    ):
        color, normal, semantics = render_color(
            self.deformer,
            self.implicit_network,
            self.rendering_network,
            sample_dict["ray_dirs"],
            sample_dict["cond"],
            sample_dict["tfs"],
            canonical_points,
            feature_vectors,
            self.training,
            num_samples,
            self.class_id,
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
