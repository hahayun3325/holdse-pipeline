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

        # ✅ NEW DEBUG: Check inputs to sdf_func_with_deformer
        print(f"\n[Node.forward] Before sdf_func_with_deformer:")
        print(f"  sample_dict['points'] has_nan: {torch.isnan(sample_dict['points']).any().item()}")
        if 'deform_info' in sample_dict and sample_dict['deform_info'] is not None:
            # deform_info is typically a dict with 'tfs', 'cond', etc.
            if isinstance(sample_dict['deform_info'], dict):
                for k, v in sample_dict['deform_info'].items():
                    if isinstance(v, torch.Tensor):
                        print(f"  deform_info['{k}'] has_nan: {torch.isnan(v).any().item()}")
            else:
                print(f"  deform_info has_nan: {torch.isnan(sample_dict['deform_info']).any().item()}")

        # compute canonical SDF and features
        (
            sdf_output,
            canonical_points,
            feature_vectors,
        ) = volsdf_utils.sdf_func_with_deformer(
            self.deformer,
            self.implicit_network,
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

        num_samples = sample_dict["z_vals"].shape[1]
        color, normal, semantics = self.render(
            sample_dict, num_samples, canonical_points, feature_vectors, time_code
        )
        self.device = color.device

        num_samples = color.shape[1]
        density = self.density(sdf_output).view(-1, num_samples, 1)
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