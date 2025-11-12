import torch
import torch.nn as nn

from ...engine.density import AbsDensity
from ...networks.shape_net import ImplicitNet
from ...networks.texture_net import RenderingNet


class Background(nn.Module):
    def __init__(self, opt, args, num_frames, sdf_bounding_sphere):
        super().__init__()

        from src.model.background.specs import bg_specs

        # Background networks
        # Frame-dependent radiane
        self.bg_implicit_network = ImplicitNet(opt.bg_implicit_network, args, bg_specs)
        # NeRF++ rendering
        self.bg_rendering_network = RenderingNet(
            opt.bg_rendering_network, args, bg_specs
        )
        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.bg_density = AbsDensity()
        self.frame_latent_encoder = nn.Embedding(
            num_frames, opt.bg_rendering_network.dim_frame_encoding
        )

        N_samples_inverse_sphere = 32
        from src.engine.ray_sampler import UniformSampler

        self.inverse_sphere_sampler = UniformSampler(
            1.0, 0.0, N_samples_inverse_sphere, False, far=1.0
        )

    def forward(self, bg_weights, ray_dirs, cam_loc, z_vals_bg, idx):
        # ===== DEBUG START =====
        print("\n" + "="*70)
        print("[BACKGROUND] forward() called")
        print(f"  idx type: {type(idx)}, value: {idx}")
        print(f"  bg_weights shape: {bg_weights.shape}")
        print(f"  bg_weights has_nan: {torch.isnan(bg_weights).any().item()}")
        print(f"  bg_weights has_inf: {torch.isinf(bg_weights).any().item()}")
        if not torch.isnan(bg_weights).any() and not torch.isinf(bg_weights).any():
            print(f"  bg_weights min: {bg_weights.min().item():.6f}, max: {bg_weights.max().item():.6f}")
        print("="*70)
        # ===== DEBUG END =====

        bg_rgb_values_only = self.bg_rendering(
            ray_dirs,
            cam_loc,
            z_vals_bg,
            idx,
        )
        bg_rgb_values = bg_weights.unsqueeze(-1) * bg_rgb_values_only

        MAX_CLASS = 4
        bg_sem = torch.zeros(bg_rgb_values.shape[0], MAX_CLASS).to(bg_rgb_values.device)
        # bg_sem = torch.zeros_like(bg_rgb_values, device=bg_rgb_values.device)
        bg_sem[:, 0] = 1.0  # bg is class-zero
        bg_semantics_values = bg_weights.unsqueeze(-1) * bg_sem

        out = {}
        out["bg_rgb"] = bg_rgb_values
        out["bg_rgb_only"] = bg_rgb_values_only
        out["bg_semantics"] = bg_semantics_values
        return out

    def bg_rendering(self, ray_dirs, cam_loc, z_vals_bg, idx):
        # ===== DEBUG START =====
        print(f"\n[BACKGROUND] bg_rendering() called")
        print(f"  idx: {idx}")
        if idx is None:
            print(f"  ⚠️  WARNING: idx is None! Returning torch.ones")
            print(f"  cam_loc shape: {cam_loc.shape}")
            # ===== DEBUG END =====
        if idx is None:
            bg_rgb_values = torch.ones_like(cam_loc, device=cam_loc.device)
            return bg_rgb_values

        frame_latent_code = self.frame_latent_encoder(idx)
        N_bg_samples = z_vals_bg.shape[1]
        z_vals_bg = torch.flip(
            z_vals_bg,
            dims=[
                -1,
            ],
        )  # 1--->0

        bg_dirs = ray_dirs.unsqueeze(1).repeat(1, N_bg_samples, 1)
        bg_locs = cam_loc.unsqueeze(1).repeat(1, N_bg_samples, 1)

        bg_points = self.depth2pts_outside(
            bg_locs, bg_dirs, z_vals_bg
        )  # [..., N_samples, 4]

        num_images = idx.shape[0]
        bg_points_flat = bg_points.reshape(num_images, -1, 4)
        bg_dirs_flat = bg_dirs.reshape(num_images, -1, 3)

        # ===== DEBUG: Check inputs to bg_implicit_network =====
        print(f"  [DEBUG] Before bg_implicit_network:")
        print(f"    bg_points_flat: shape={bg_points_flat.shape}, has_nan={torch.isnan(bg_points_flat).any().item()}, has_inf={torch.isinf(bg_points_flat).any().item()}")
        if torch.isinf(bg_points_flat).any():
            print(f"      ⚠️ bg_points_flat contains Inf!")
        if not torch.isnan(bg_points_flat).any() and not torch.isinf(bg_points_flat).any():
            print(f"      min={bg_points_flat.min().item():.6f}, max={bg_points_flat.max().item():.6f}")
        print(f"    frame_latent_code: shape={frame_latent_code.shape}, has_nan={torch.isnan(frame_latent_code).any().item()}")
        print(f"    z_vals_bg: min={z_vals_bg.min().item():.6f}, max={z_vals_bg.max().item():.6f}")
        # ===== END DEBUG =====

        bg_output = self.bg_implicit_network(
            bg_points_flat, {"frame": frame_latent_code}
        )

        # ===== DEBUG START =====
        print(f"  bg_implicit_network output has_nan: {torch.isnan(bg_output).any().item()}")
        # ===== DEBUG END =====

        bg_sdf = bg_output[:, :, :1]
        bg_feature_vectors = bg_output[:, :, 1:]
        bg_rendering_output = self.bg_rendering_network(
            None, None, bg_dirs_flat, None, bg_feature_vectors, frame_latent_code
        )

        # ===== DEBUG START =====
        print(f"  bg_rendering_network output has_nan: {torch.isnan(bg_rendering_output).any().item()}")
        # ===== DEBUG END =====

        if bg_rendering_output.shape[-1] == 4:
            bg_rgb_flat = bg_rendering_output[..., :-1]
            shadow_r = bg_rendering_output[..., -1]
            bg_rgb = bg_rgb_flat.reshape(-1, N_bg_samples, 3)
            shadow_r = shadow_r.reshape(-1, N_bg_samples, 1)
            bg_rgb = (1 - shadow_r) * bg_rgb
        else:
            bg_rgb_flat = bg_rendering_output
            bg_rgb = bg_rgb_flat.reshape(-1, N_bg_samples, 3)
        bg_weights = self.bg_volume_rendering(z_vals_bg, bg_sdf)
        bg_rgb_values = torch.sum(bg_weights.unsqueeze(-1) * bg_rgb, 1)

        return bg_rgb_values

    def depth2pts_outside(self, ray_o, ray_d, depth):
        """
        ray_o, ray_d: [..., 3]
        depth: [...]; inverse of distance to sphere origin
        """
        # Debug: Check inputs
        print(f"[depth2pts_outside] Input checks:")
        print(f"  depth: min={depth.min():.6f}, max={depth.max():.6f}, has_nan={torch.isnan(depth).any().item()}")
        print(f"  ray_o: has_nan={torch.isnan(ray_o).any().item()}, max_norm={torch.norm(ray_o, dim=-1).max():.6f}")
        print(f"  ray_d: has_nan={torch.isnan(ray_d).any().item()}")
        print(f"  sdf_bounding_sphere: {self.sdf_bounding_sphere}")

        o_dot_d = torch.sum(ray_d * ray_o, dim=-1)

        under_sqrt = o_dot_d**2 - ((ray_o**2).sum(-1) - self.sdf_bounding_sphere**2)
        print(f"  under_sqrt: min={under_sqrt.min():.6f}, max={under_sqrt.max():.6f}")

        if (under_sqrt < 0).any():
            neg_count = (under_sqrt < 0).sum().item()
            print(f"  ❌ WARNING: under_sqrt has {neg_count} negative values!")
            print(f"     This will produce NaN in sqrt!")
            # Clamp to avoid NaN
            under_sqrt = torch.clamp(under_sqrt, min=1e-6)

        d_sphere = torch.sqrt(under_sqrt) - o_dot_d
        p_sphere = ray_o + d_sphere.unsqueeze(-1) * ray_d
        p_mid = ray_o - o_dot_d.unsqueeze(-1) * ray_d
        p_mid_norm = torch.norm(p_mid, dim=-1)

        print(f"  p_mid_norm: min={p_mid_norm.min():.6f}, max={p_mid_norm.max():.6f}")

        rot_axis = torch.cross(ray_o, p_sphere, dim=-1)
        rot_axis_norm = torch.norm(rot_axis, dim=-1, keepdim=True)

        print(f"  rot_axis_norm: min={rot_axis_norm.min():.6f}, max={rot_axis_norm.max():.6f}")

        if (rot_axis_norm < 1e-6).any():
            zero_count = (rot_axis_norm < 1e-6).sum().item()
            print(f"  ❌ WARNING: rot_axis_norm has {zero_count} near-zero values!")
            print(f"     This will produce Inf in division!")
            rot_axis_norm = torch.clamp(rot_axis_norm, min=1e-6)

        rot_axis = rot_axis / rot_axis_norm

        # Check asin arguments
        asin_arg1 = p_mid_norm / self.sdf_bounding_sphere
        asin_arg2 = p_mid_norm * depth

        print(f"  asin_arg1 (p_mid_norm/sphere): min={asin_arg1.min():.6f}, max={asin_arg1.max():.6f}")
        print(f"  asin_arg2 (p_mid_norm*depth): min={asin_arg2.min():.6f}, max={asin_arg2.max():.6f}")

        if (torch.abs(asin_arg1) > 1.0).any():
            bad_count = (torch.abs(asin_arg1) > 1.0).sum().item()
            print(f"  ❌ WARNING: asin_arg1 has {bad_count} values outside [-1,1]!")
            print(f"     This will produce NaN in asin!")
            asin_arg1 = torch.clamp(asin_arg1, -1.0 + 1e-6, 1.0 - 1e-6)

        if (torch.abs(asin_arg2) > 1.0).any():
            bad_count = (torch.abs(asin_arg2) > 1.0).sum().item()
            print(f"  ❌ WARNING: asin_arg2 has {bad_count} values outside [-1,1]!")
            print(f"     This will produce NaN in asin!")
            asin_arg2 = torch.clamp(asin_arg2, -1.0 + 1e-6, 1.0 - 1e-6)

        phi = torch.asin(asin_arg1)
        theta = torch.asin(asin_arg2)
        rot_angle = (phi - theta).unsqueeze(-1)

        # now rotate p_sphere
        # Rodrigues formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        p_sphere_new = (
            p_sphere * torch.cos(rot_angle)
            + torch.cross(rot_axis, p_sphere, dim=-1) * torch.sin(rot_angle)
            + rot_axis
            * torch.sum(rot_axis * p_sphere, dim=-1, keepdim=True)
            * (1.0 - torch.cos(rot_angle))
        )
        p_sphere_new = p_sphere_new / torch.norm(p_sphere_new, dim=-1, keepdim=True)
        pts = torch.cat((p_sphere_new, depth.unsqueeze(-1)), dim=-1)

        return pts

    def bg_volume_rendering(self, z_vals_bg, bg_sdf):
        bg_density_flat = self.bg_density(bg_sdf)
        bg_density = bg_density_flat.reshape(
            -1, z_vals_bg.shape[1]
        )  # (batch_size * num_pixels) x N_samples

        bg_dists = z_vals_bg[:, :-1] - z_vals_bg[:, 1:]
        bg_dists = torch.cat(
            [
                bg_dists,
                torch.tensor([1e10]).cuda().unsqueeze(0).repeat(bg_dists.shape[0], 1),
            ],
            -1,
        )

        # LOG SPACE
        bg_free_energy = bg_dists * bg_density
        bg_shifted_free_energy = torch.cat(
            [torch.zeros(bg_dists.shape[0], 1).cuda(), bg_free_energy[:, :-1]], dim=-1
        )  # shift one step
        bg_alpha = 1 - torch.exp(-bg_free_energy)  # probability of it is not empty here
        bg_transmittance = torch.exp(
            -torch.cumsum(bg_shifted_free_energy, dim=-1)
        )  # probability of everything is empty up to now
        bg_weights = (
            bg_alpha * bg_transmittance
        )  # probability of the ray hits something here

        return bg_weights

    def step_embedding(self):
        """
        Step embeddings for progressive training.

        ✅ MEMORY OPTIMIZATION COMPATIBLE:
        When multires=0 (no positional encoding), embedder_obj is None.
        Skip the step call safely.
        """
        # ================================================================
        # ✅ CRITICAL: Check if embedders exist before calling .step()
        # When multires=0 or multires_view=-1, embedder_obj is None
        # ================================================================

        # Check bg_implicit_network embedder
        if (hasattr(self, 'bg_implicit_network') and
            hasattr(self.bg_implicit_network, 'embedder_obj') and
            self.bg_implicit_network.embedder_obj is not None):
            self.bg_implicit_network.embedder_obj.step()

        # Check bg_rendering_network embedder
        if (hasattr(self, 'bg_rendering_network') and
            hasattr(self.bg_rendering_network, 'embedder_obj') and
            self.bg_rendering_network.embedder_obj is not None):
            self.bg_rendering_network.embedder_obj.step()
