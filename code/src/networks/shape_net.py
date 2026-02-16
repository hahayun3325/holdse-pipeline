import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from ..engine.embedders import get_embedder


class ImplicitNet(nn.Module):
    def __init__(self, opt, args, body_specs):
        super().__init__()
        # --- DEBUG START ---
        logger.info(f"c")
        logger.info(f"DEBUG: opt.cond_dim = {getattr(opt, 'cond_dim', 'MISSING')}")
        logger.info(f"DEBUG: opt.use_film = {getattr(opt, 'use_film', 'MISSING')}")
        logger.info(f"DEBUG: self.cond = {opt.cond}")
        # --- DEBUG END ---
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
        elif self.cond == "geo" or self.cond == "geometry":
            self.cond_layer = [0]
            # Store original cond_dim for FiLM before potentially zeroing it
            self.film_cond_dim = opt.cond_dim  # Always 128 for geometry latent
            if getattr(opt, "use_film", False):
                # FiLM mode: no concatenation at input layer (cond_dim=0)
                self.cond_dim = 0
            else:
                # Concatenation mode: add latent to input
                self.cond_dim = opt.cond_dim

        else:
            self.cond_layer = []
            self.cond_dim = 0

        if self.cond_dim == 0:
            logger.warning("c - FiLM disabled!")

        self.dim_pose_embed = 0
        if self.dim_pose_embed > 0:
            self.lin_p0 = nn.Linear(self.cond_dim, self.dim_pose_embed)
            self.cond_dim = self.dim_pose_embed

        # ------------------------------------------------------------------
        # NEW: optional FiLM conditioning for geometry latent (Option B)
        # ------------------------------------------------------------------
        # Only turn this on for the object implicit network via config:
        #   opt.use_film = True, opt.cond = "geo" (or similar)
        self.use_film = getattr(opt, "use_film", False)
        if self.use_film and self.cond != "none":
            # Use film_cond_dim (128), not cond_dim (0)
            self.num_cond = self.film_cond_dim

            # Per-layer FiLM MLPs: gamma_l(cond), beta_l(cond) → [feat_dim_l]
            self.film_gamma = nn.ModuleList()
            self.film_beta = nn.ModuleList()
            for l in range(0, self.num_layers - 1):
                out_dim = dims[l + 1]
                # Use film_cond_dim for FiLM layers
                self.film_gamma.append(nn.Linear(self.film_cond_dim, out_dim))  # 128 input dim
                self.film_beta.append(nn.Linear(self.film_cond_dim, out_dim))  # 128 input dim
        else:
            # Backward-compatible default (no FiLM)
            self.use_film = False
            self.num_cond = getattr(self, "cond_dim", 0)

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

        # ================================================================
        # Conditioning: build cond_tensor [B*N, cond_dim]
        # ================================================================
        input_cond = None
        if self.cond != "none":
            cond_tensor = cond[self.cond]  # e.g., 'pose', 'frame', 'geo'

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
                    cond_slice = cond_tensor[:original_num_batch]
                else:
                    # Too few cond samples, repeat
                    repeats = (original_num_batch + cond_batch - 1) // cond_batch
                    cond_slice = cond_tensor.repeat(repeats, 1)[:original_num_batch]

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

        # ================================================================
        # Main MLP with optional FiLM
        # ================================================================
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            # Concatenative conditioning (old behavior) – ONLY when not using FiLM
            if (self.cond != "none"
                and l in self.cond_layer
                and input_cond is not None
                and not self.use_film):
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

                    if expected_cond_dim == 0:
                        # Model was trained without conditioning - don't concatenate anything
                        logger.warning("Model expects NO conditioning (dim=0), ignoring provided conditioning")
                        # x remains unchanged
                    elif input_cond.shape[1] < expected_cond_dim:
                        # Pad with zeros
                        padding = torch.zeros(input_cond.shape[0], expected_cond_dim - input_cond.shape[1], device=input_cond.device)
                        input_cond = torch.cat([input_cond, padding], dim=-1)
                        x = torch.cat([x, input_cond], dim=-1)
                    else:
                        # Slice
                        input_cond = input_cond[:, :expected_cond_dim]
                        x = torch.cat([x, input_cond], dim=-1)
                else:
                    # Dimensions match
                    if expected_cond_dim > 0:
                        x = torch.cat([x, input_cond], dim=-1)
                    # If expected_cond_dim == 0, don't concatenate

            # Skip connections unchanged
            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            # FiLM compatibility: Handle checkpoint dimension mismatch
            if self.use_film and x.shape[1] != lin.weight.shape[1]:
                expected_dim = lin.weight.shape[1]
                actual_dim = x.shape[1]
                if actual_dim < expected_dim:
                    # Pad with zeros to match old checkpoint dimensions
                    padding = torch.zeros(x.shape[0], expected_dim - actual_dim, device=x.device)
                    x = torch.cat([x, padding], dim=-1)
                else:
                    # Slice if too large
                    x = x[:, :expected_dim]

            # Linear layer
            x = lin(x)

            # Nonlinearity on all but last layer
            if l < self.num_layers - 2:
                x = self.softplus(x)

            # ------------------------------------------------------------
            # NEW: FiLM modulation using geometry latent (Option B)
            # ------------------------------------------------------------
            if self.use_film and self.cond != "none" and input_cond is not None:
                gamma = self.film_gamma[l](input_cond)  # [B*N, feat_dim_l]
                beta = self.film_beta[l](input_cond)    # [B*N, feat_dim_l]

                # Handle dimension mismatch between gamma/beta and x
                if gamma.shape[1] != x.shape[1]:
                    if gamma.shape[1] < x.shape[1]:
                        # Pad gamma/beta to match x
                        pad_size = x.shape[1] - gamma.shape[1]
                        gamma = torch.cat([gamma, torch.zeros(gamma.shape[0], pad_size, device=gamma.device)], dim=1)
                        beta = torch.cat([beta, torch.zeros(beta.shape[0], pad_size, device=beta.device)], dim=1)
                    else:
                        # Slice gamma/beta to match x
                        gamma = gamma[:, :x.shape[1]]
                        beta = beta[:, :x.shape[1]]

                x = gamma * x + beta
                logger.info(f"[FiLM] gamma range: [{gamma.min():.4f}, {gamma.max():.4f}], beta range: [{beta.min():.4f}, {beta.max():.4f}]")

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
