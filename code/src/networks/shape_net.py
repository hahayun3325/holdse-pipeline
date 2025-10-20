import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from ..engine.embedders import get_embedder


class ImplicitNet(nn.Module):
    def __init__(self, opt, args, body_specs):
        super().__init__()

        dims = [opt.d_in] + list(opt.dims) + [opt.d_out + opt.feature_vector_size]
        self.num_layers = len(dims)
        self.skip_in = opt.skip_in
        self.embedder_obj = None
        self.opt = opt
        self.body_specs = body_specs

        if opt.multires > 0:
            embedder_obj, input_ch = get_embedder(
                opt.multires,
                input_dims=opt.d_in,
                mode=body_specs.embedding,
                barf_s=args.barf_s,
                barf_e=args.barf_e,
                no_barf=args.no_barf,
            )
            self.embedder_obj = embedder_obj
            dims[0] = input_ch
        self.cond = opt.cond
        if self.cond == "pose":
            self.cond_layer = [0]
            self.cond_dim = body_specs.pose_dim
        elif self.cond == "frame":
            self.cond_layer = [0]
            self.cond_dim = opt.dim_frame_encoding
        self.dim_pose_embed = 0
        if self.dim_pose_embed > 0:
            self.lin_p0 = nn.Linear(self.cond_dim, self.dim_pose_embed)
            self.cond_dim = self.dim_pose_embed
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            if self.cond != "none" and l in self.cond_layer:
                lin = nn.Linear(dims[l] + self.cond_dim, out_dim)
            else:
                lin = nn.Linear(dims[l], out_dim)
            if opt.init == "geometry":
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(
                        lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001
                    )
                    torch.nn.init.constant_(lin.bias, -opt.bias)
                elif opt.multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(
                        lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim)
                    )
                elif opt.multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(
                        lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim)
                    )
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3) :], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(
                        lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim)
                    )
            if opt.init == "zero":
                init_val = 1e-5
                if l == self.num_layers - 2:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.uniform_(lin.weight, -init_val, init_val)
            if opt.weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        self.softplus = nn.Softplus(beta=100)

    def forward(self, input, cond, current_epoch=None):
        if input.ndim == 2:
            input = input.unsqueeze(0)
        num_batch, num_point, num_dim = input.shape

        if num_batch * num_point == 0:
            return input

        # Save original dimensions
        original_num_batch = num_batch
        original_num_point = num_point

        input = input.reshape(num_batch * num_point, num_dim)

        if self.cond != "none":
            # ================================================================
            # ✅ FIX: Handle 2D or 3D cond tensors
            # ================================================================
            cond_tensor = cond[self.cond]

            # If cond is 3D [B, N, D], reshape to 2D [B*N, D]
            if cond_tensor.ndim == 3:
                cond_batch = cond_tensor.shape[0]
                cond_points = cond_tensor.shape[1]
                num_cond = cond_tensor.shape[2]
                # Flatten to 2D: [B*N, D]
                cond_tensor = cond_tensor.reshape(cond_batch * cond_points, num_cond)
                cond_batch = cond_batch  # Use original batch for logic below
            elif cond_tensor.ndim == 2:
                cond_batch, num_cond = cond_tensor.shape
            else:
                raise ValueError(f"Unexpected cond tensor dimensions: {cond_tensor.shape}")

            # Rest of the function continues with cond_tensor instead of cond[self.cond]
            # ================================================================
            # The cond might have different batch size than input
            # Use the ACTUAL input batch size for expansion

            if cond_batch == original_num_batch:
                # Normal case: cond batch matches input batch
                input_cond = (
                    cond_tensor.unsqueeze(1).expand(cond_batch, original_num_point, num_cond)
                )
            else:
                # Mismatch case: repeat cond to match input batch
                logger.warning(f"Cond batch {cond_batch} != input batch {original_num_batch}")
                # Use input batch size, repeat/slice cond as needed
                if cond_batch > original_num_batch:
                    # Too many cond samples, slice
                    cond_slice = cond[self.cond][:original_num_batch]
                else:
                    # Too few cond samples, repeat
                    repeats = (original_num_batch + cond_batch - 1) // cond_batch
                    cond_slice = cond[self.cond].repeat(repeats, 1)[:original_num_batch]

                input_cond = cond_slice.unsqueeze(1).expand(original_num_batch, original_num_point, num_cond)

            if num_cond == 45:
                # no pose dependent for MANO
                input_cond = input_cond * 0.0

            # ✅ FIX: Reshape using original_num_batch, not cond_batch
            input_cond = input_cond.reshape(original_num_batch * original_num_point, num_cond)

            if self.dim_pose_embed:
                input_cond = self.lin_p0(input_cond)

        if self.embedder_obj is not None:
            input = self.embedder_obj.embed(input)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if self.cond != "none" and l in self.cond_layer:
                # ✅ FIX: Ensure input_cond matches expected dimensions
                # The linear layer after concat expects specific input size
                # x.shape: [N, feat_dim]
                # input_cond.shape: [N, cond_dim]
                # lin expects: [N, feat_dim + expected_cond_dim]

                # Check if dimensions match
                expected_input_dim = lin.weight.shape[1]  # Input features expected by linear layer
                current_feat_dim = x.shape[1]
                expected_cond_dim = expected_input_dim - current_feat_dim

                if input_cond.shape[1] != expected_cond_dim:
                    logger.warning(f"input_cond dim {input_cond.shape[1]} != expected {expected_cond_dim}, padding/slicing")
                    if input_cond.shape[1] < expected_cond_dim:
                        # Pad with zeros
                        padding = torch.zeros(input_cond.shape[0], expected_cond_dim - input_cond.shape[1], device=input_cond.device)
                        input_cond = torch.cat([input_cond, padding], dim=-1)
                    else:
                        # Slice
                        input_cond = input_cond[:, :expected_cond_dim]

                x = torch.cat([x, input_cond], dim=-1)

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)
            if l < self.num_layers - 2:
                x = self.softplus(x)

        x = x.reshape(num_batch, num_point, -1)

        return x

    def gradient(self, x, cond):
        x.requires_grad_(True)
        y = self.forward(x, cond)[:, :1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return gradients.unsqueeze(1)
