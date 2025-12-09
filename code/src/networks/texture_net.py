import torch
import torch.nn as nn

from ..engine.embedders import get_embedder

def debug_embedder(multires):
    """Calculate positional encoding dimensions"""
    # From get_embedder() implementation
    embed_fns = []
    d = 3  # input dimension (xyz or view direction)
    out_dim = 0

    if True:  # include_input
        out_dim += d

    max_freq = multires - 1
    N_freqs = multires

    for freq_idx in range(N_freqs):
        for p_fn in [torch.sin, torch.cos]:
            out_dim += d

    print(f"\n[EMBEDDER DEBUG] multires={multires}:")
    print(f"  Input dim: {d}")
    print(f"  Include input: {d}D")
    print(f"  Frequencies: {N_freqs}")
    print(f"  Sin/Cos pairs: {N_freqs} * 2 * {d} = {N_freqs * 2 * d}D")
    print(f"  Total output: {out_dim}D")
    print(f"  Overhead: {out_dim - d}D")
    return out_dim

class RenderingNet(nn.Module):
    def __init__(self, opt, args, body_specs):
        super().__init__()

        self.mode = opt.mode
        # dims = [opt.d_in + opt.feature_vector_size] + list(opt.dims) + [opt.d_out]
        dims = [opt.d_in] + list(opt.dims) + [opt.d_out]

        # ✅ DEBUG 1: Print initial dims
        print(f"\n[RENDER NET INIT] Initial configuration:")
        print(f"  opt.d_in: {opt.d_in}")
        print(f"  dims[0] BEFORE adjustments: {dims[0]}")

        self.body_specs = body_specs

        self.embedder_obj = None
        if opt.multires_view > 0:
            embedder_obj, input_ch = get_embedder(
                opt.multires_view,
                mode=body_specs.embedding,
                barf_s=args.barf_s,
                barf_e=args.barf_e,
                no_barf=args.no_barf,
            )
            # ✅ DEBUG: Verify embedder dimensions
            calculated_input_ch = debug_embedder(opt.multires_view)
            assert calculated_input_ch == input_ch, f"Embedder mismatch: {calculated_input_ch} != {input_ch}"

            self.embedder_obj = embedder_obj

            # ✅ DEBUG 2: Print multires_view adjustment
            print(f"\n[RENDER NET INIT] multires_view adjustment:")
            print(f"  opt.multires_view: {opt.multires_view}")
            print(f"  input_ch from embedder: {input_ch}")
            print(f"  dims[0] BEFORE: {dims[0]}")
            dims[0] += input_ch - 3
            print(f"  dims[0] AFTER: {dims[0]} (added {input_ch - 3})")

        if self.mode == "nerf_frame_encoding":
            dims[0] += opt.dim_frame_encoding
        if self.mode == "pose":
            self.dim_cond_embed = 8
            self.cond_dim = self.body_specs.pose_dim
            self.lin_pose = torch.nn.Linear(self.cond_dim, self.dim_cond_embed)

            # ✅ DEBUG 3: Print pose configuration
            print(f"\n[RENDER NET INIT] Pose mode configuration:")
            print(f"  body_specs.pose_dim: {self.cond_dim}")
            print(f"  dim_cond_embed: {self.dim_cond_embed}")

        # ✅ DEBUG 4: Print final dims
        print(f"\n[RENDER NET INIT] Final network dims:")
        print(f"  dims: {dims}")
        print(f"  lin0 will be: {dims[0]} → {dims[1]}")
        print(f"  Total layers: {len(dims) - 1}")
        print(f"=" * 70 + "\n")

        self.num_layers = len(dims)
        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            if opt.weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        points,
        normals,
        view_dirs,
        body_pose,
        feature_vectors,
        frame_latent_code=None,
    ):
        # ✅ NEW DEBUG: Check all inputs for NaN before processing
        print(f"\n[RENDER NET INPUT NaN CHECK]")
        if points is not None:
            print(f"  points has_nan: {torch.isnan(points).any().item()}")
        if normals is not None:
            print(f"  normals has_nan: {torch.isnan(normals).any().item()}")
        if view_dirs is not None:
            print(f"  view_dirs has_nan: {torch.isnan(view_dirs).any().item()}")
        if body_pose is not None:
            print(f"  body_pose has_nan: {torch.isnan(body_pose).any().item()}")
        if feature_vectors is not None:
            print(f"  feature_vectors has_nan: {torch.isnan(feature_vectors).any().item()}")
        if frame_latent_code is not None:
            print(f"  frame_latent_code has_nan: {torch.isnan(frame_latent_code).any().item()}")

        if self.embedder_obj is not None:
            if self.mode == "nerf_frame_encoding":
                view_dirs = self.embedder_obj.embed(view_dirs)

        if self.mode == "nerf_frame_encoding":
            # ================================================================
            # ✅ FIX: Handle frame_latent_code dimensionality COMPLETELY
            # ================================================================
            if frame_latent_code is not None:
                # Ensure frame_latent_code is EXACTLY 2D [B, D]
                while frame_latent_code.ndim > 2:
                    # Squeeze all extra dimensions
                    frame_latent_code = frame_latent_code.squeeze(1)

                if frame_latent_code.ndim == 1:
                    # 1D: [D] -> [1, D]
                    frame_latent_code = frame_latent_code.unsqueeze(0)

                # Now frame_latent_code is guaranteed to be [B, D]

                # After embedding, view_dirs might be 2D, 3D, or 4D
                # We need to flatten to 2D: [total_points, embed_dim]
                if view_dirs.ndim >= 3:
                    # Flatten all batch/point dimensions: [B*N*..., D]
                    view_dirs_2d = view_dirs.reshape(-1, view_dirs.shape[-1])
                    num_total_points = view_dirs_2d.shape[0]

                    # Expand frame_latent_code to match
                    batch_size = frame_latent_code.shape[0]
                    points_per_batch = num_total_points // batch_size

                    # Now safe: frame_latent_code is [B, D]
                    frame_latent_code_expanded = frame_latent_code.unsqueeze(1).repeat(1, points_per_batch, 1)
                    frame_latent_code_2d = frame_latent_code_expanded.reshape(-1, frame_latent_code.shape[-1])

                    # ✅ FIX: Flatten feature_vectors to match!
                    if feature_vectors.ndim > 2:
                        feature_vectors_2d = feature_vectors.reshape(-1, feature_vectors.shape[-1])
                    else:
                        feature_vectors_2d = feature_vectors

                    rendering_input = torch.cat([view_dirs_2d, frame_latent_code_2d, feature_vectors_2d], dim=-1)
                elif view_dirs.ndim == 2:
                    # Already 2D: [N, D]
                    num_points = view_dirs.shape[0]
                    batch_size = frame_latent_code.shape[0]

                    frame_latent_code_expanded = frame_latent_code.unsqueeze(1).repeat(1, num_points, 1)
                    frame_latent_code_2d = frame_latent_code_expanded.reshape(-1, frame_latent_code.shape[-1])

                    # ✅ FIX: Flatten feature_vectors to match!
                    if feature_vectors.ndim > 2:
                        feature_vectors_2d = feature_vectors.reshape(-1, feature_vectors.shape[-1])
                    else:
                        feature_vectors_2d = feature_vectors

                    rendering_input = torch.cat([view_dirs, frame_latent_code_2d, feature_vectors_2d], dim=-1)
                else:
                    raise ValueError(f"Unexpected view_dirs dimensionality: {view_dirs.shape}")
            else:
                # No frame latent code - just flatten view_dirs if needed
                if view_dirs.ndim > 2:
                    view_dirs = view_dirs.reshape(-1, view_dirs.shape[-1])
                rendering_input = torch.cat([view_dirs, feature_vectors], dim=-1)

            # ✅ FIX: Handle different dimensionalities
            if rendering_input.ndim == 3:
                rendering_input = rendering_input.view(-1, rendering_input.shape[2])
            # else already 2D
        elif self.mode == "pose":
            # ✅ FIX: Only access shapes if not None
            if points is None or normals is None or body_pose is None:
                raise ValueError(
                    f"[RENDER NET] mode='pose' requires points, normals, and body_pose, "
                    f"but got points={points is not None}, normals={normals is not None}, "
                    f"body_pose={body_pose is not None}"
                )

            num_images = body_pose.shape[0]
            points = points.view(num_images, -1, 3)

            num_points = points.shape[1]
            points = points.reshape(num_images * num_points, -1)
            body_pose = (
                body_pose[:, None, :]
                .repeat(1, num_points, 1)
                .reshape(num_images * num_points, -1)
            )
            num_dim = body_pose.shape[1]

            if num_dim > 0 and self.cond_dim > 0:
                # Normal case: MANO has pose (45D), has linear layer
                body_pose = self.lin_pose(body_pose)
                print(f"  body_pose after lin_pose: {body_pose.shape}")
            elif num_dim > 0 and self.cond_dim == 0:
                # Object case: has orient (3D) but NO pose linear layer
                # The network was trained WITHOUT pose conditioning

                # ✅ DEBUG: Check actual sizes
                print(f"\n[RENDER NET] cond_dim=0 path:")
                print(f"  points shape: {points.shape}")
                print(f"  normals shape: {normals.shape if normals is not None else 'None'}")
                print(f"  feature_vectors shape: {feature_vectors.shape}")
                print(f"  lin0 expects input: {self.lin0.weight.shape[1]}")

                expected_input_size = self.lin0.weight.shape[1]  # 302
                current_size_without_pose = 3 + 3 + feature_vectors.shape[-1]
                needed_body_pose_size = expected_input_size - current_size_without_pose

                print(f"  Calculated needed_body_pose_size: {needed_body_pose_size}")

                if needed_body_pose_size < 0:
                    print(f"  ⚠️  ERROR: negative padding size! Feature vectors too large!")
                    print(f"  Expected: {expected_input_size}, Got: {current_size_without_pose}")
                    # Fallback: use zero padding
                    needed_body_pose_size = 0

                body_pose = torch.zeros(points.shape[0], needed_body_pose_size, device=points.device)
                print(f"  Created zero body_pose: {body_pose.shape}")
            else:
                # No pose parameters at all
                expected_input_size = self.lin0.weight.shape[1]
                current_size = 3 + 3 + feature_vectors.shape[-1]
                needed_body_pose_size = expected_input_size - current_size
                body_pose = torch.zeros(points.shape[0], needed_body_pose_size, device=points.device)

            rendering_input = torch.cat(
                [points, normals, body_pose, feature_vectors], dim=-1
            )

            if rendering_input.shape[-1] != self.lin0.weight.shape[1]:
                print(f"  ❌ MISMATCH! Difference: {self.lin0.weight.shape[1] - rendering_input.shape[-1]}")
            else:
                print(f"  ✅ MATCH!")
            print(f"=" * 70 + "\n")

        else:
            raise NotImplementedError

        # After constructing rendering_input, add:
        print(f"\n[RENDER NET] rendering_input constructed:")
        print(f"  shape: {rendering_input.shape}")
        print(f"  Expected by lin0: {self.lin0.weight.shape[1]}")
        if rendering_input.shape[-1] != self.lin0.weight.shape[1]:
            print(f"  ❌ MISMATCH! Difference: {rendering_input.shape[-1] - self.lin0.weight.shape[1]}")
        print(f"  has_nan: {torch.isnan(rendering_input).any().item()}")
        if torch.isnan(rendering_input).any():
            print(f"  ❌ rendering_input contains NaN before first layer!")

        x = rendering_input
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)

            # ✅ NEW DEBUG: Check for NaN after each layer
            if torch.isnan(x).any():
                print(f"  ❌ NaN detected after lin{l}: shape={x.shape}")

            if l < self.num_layers - 2:
                x = self.relu(x)

                # ✅ NEW DEBUG: Check for NaN after ReLU
                if torch.isnan(x).any():
                    print(f"  ❌ NaN detected after relu{l}: shape={x.shape}")

        x = self.sigmoid(x)

        # ✅ NEW DEBUG: Check final output
        print(f"\n[RENDER NET OUTPUT]")
        print(f"  shape: {x.shape}")
        print(f"  has_nan: {torch.isnan(x).any().item()}")
        if not torch.isnan(x).any():
            print(f"  min: {x.min().item():.6f}, max: {x.max().item():.6f}")
        else:
            print(f"  ❌ Final output contains NaN!")

        return x
