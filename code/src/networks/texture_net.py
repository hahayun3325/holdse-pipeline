import torch
import torch.nn as nn

from ..engine.embedders import get_embedder


class RenderingNet(nn.Module):
    def __init__(self, opt, args, body_specs):
        super().__init__()

        self.mode = opt.mode
        dims = [opt.d_in + opt.feature_vector_size] + list(opt.dims) + [opt.d_out]

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
            self.embedder_obj = embedder_obj
            dims[0] += input_ch - 3
        if self.mode == "nerf_frame_encoding":
            dims[0] += opt.dim_frame_encoding
        if self.mode == "pose":
            self.dim_cond_embed = 8
            self.cond_dim = (
                self.body_specs.pose_dim
            )  # dimension of the body pose, global orientation excluded.
            # lower the condition dimension
            self.lin_pose = torch.nn.Linear(self.cond_dim, self.dim_cond_embed)
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

            # ================================================================
            # ✅ FIX: Handle zero or non-zero pose dimensions CORRECTLY
            # ================================================================
            if num_dim > 0 and self.cond_dim > 0:
                # Normal case: MANO has pose (45D), has linear layer
                body_pose = self.lin_pose(body_pose)
            elif num_dim > 0 and self.cond_dim == 0:
                # Object case: has orient (3D) but NO pose linear layer
                # The network was trained WITHOUT pose conditioning
                # So we need to create a ZERO embedding of the EXPECTED size
                # BUT: We need to match what the first linear layer expects!
                # The rendering_input will be: [points(3) + normals(3) + body_pose(?) + features(37)]
                # The lin0 expects 78 input features
                # So: 3 + 3 + body_pose_size + 37 = 78
                # Therefore: body_pose_size = 78 - 3 - 3 - 37 = 35
                expected_input_size = self.lin0.weight.shape[1]  # 78
                current_size = 3 + 3 + feature_vectors.shape[-1]  # 3(points) + 3(normals) + 37(features)
                needed_body_pose_size = expected_input_size - current_size

                body_pose = torch.zeros(points.shape[0], needed_body_pose_size, device=points.device)
            else:
                # No pose parameters at all
                expected_input_size = self.lin0.weight.shape[1]
                current_size = 3 + 3 + feature_vectors.shape[-1]
                needed_body_pose_size = expected_input_size - current_size
                body_pose = torch.zeros(points.shape[0], needed_body_pose_size, device=points.device)

            rendering_input = torch.cat(
                [points, normals, body_pose, feature_vectors], dim=-1
            )
        else:
            raise NotImplementedError

        x = rendering_input
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.relu(x)
        x = self.sigmoid(x)
        return x
