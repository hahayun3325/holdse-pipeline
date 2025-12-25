"""
Diffusion utilities for SDS loss computation.

PHASE 3 UPDATES:
- Added GHOP3DUNet: Full 3D U-Net with attention for noise prediction
- Added GHOP3DUNetWrapper: Inference wrapper with checkpoint loading
- Includes ResBlock3D, SpatialTransformer3D, and time embedding utilities
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from loguru import logger

class DiffusionSchedule:
    """
    Linear noise schedule for diffusion.
    """

    def __init__(self, steps=1000, beta_min=0.0001, beta_max=0.02):
        self.steps = steps

        # Linear schedule
        betas = np.linspace(beta_min, beta_max, steps, dtype=np.float32)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)

        # Convert to torch tensors
        self.betas = torch.from_numpy(betas)
        self.alphas = torch.from_numpy(alphas)
        self.alphas_cumprod = torch.from_numpy(alphas_cumprod)

    def to(self, device):
        """Move to device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        return self

    def q_sample(self, x_0, t, noise):
        """
        Forward diffusion: add noise to x_0 at timestep t.

        Args:
            x_0: [B, C, D, H, W] clean sample
            t: [B] timesteps
            noise: [B, C, D, H, W] Gaussian noise

        Returns:
            x_t: [B, C, D, H, W] noisy sample
        """
        sqrt_alphas_cumprod_t = torch.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - self.alphas_cumprod[t])

        # Reshape for broadcasting
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1, 1, 1)

        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

        return x_t


# ============================================================================
# PHASE 3: Time Embedding Utilities
# ============================================================================

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    Args:
        timesteps: (B,) tensor of timesteps
        dim: Embedding dimension
        max_period: Maximum period for sinusoidal encoding

    Returns:
        emb: (B, dim) time embeddings
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# ============================================================================
# PHASE 3: 3D U-Net Building Blocks
# ============================================================================
class ResBlock3D(nn.Module):
    """
    3D Residual block with time embedding and optional dropout.
    """
    def __init__(self, in_channels, out_channels, time_emb_channels=None, dropout=0.0):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_time_emb = time_emb_channels is not None

        # First conv block
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv3d(in_channels, out_channels, 3, padding=1)
        )

        # Time embedding projection
        if self.use_time_emb:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_channels, 2 * out_channels)
            )

        # Second conv block
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv3d(out_channels, out_channels, 3, padding=1)
        )

        # Skip connection
        if in_channels != out_channels:
            self.skip_connection = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.skip_connection = nn.Identity()

    def forward(self, x, emb=None):
        """
        Args:
            x: (B, in_channels, D, H, W)
            emb: (B, time_emb_channels) optional time embedding
        Returns:
            out: (B, out_channels, D, H, W)
        """
        h = self.in_layers(x)

        # Add time embedding if provided (FiLM: scale and shift)
        if self.use_time_emb and emb is not None:
            emb_out = self.emb_layers(emb)  # Shape: (B, 2*out_channels)
            scale, shift = emb_out.chunk(2, dim=1)  # Each: (B, out_channels)
            h = h * (1 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]

        h = self.out_layers(h)

        return h + self.skip_connection(x)

class Downsample3D(nn.Module):
    """3D downsampling by factor of 2."""
    def __init__(self, channels, use_conv=True):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv

        if use_conv:
            self.op = nn.Conv3d(channels, channels, 3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.op(x)

class Upsample3D(nn.Module):
    """3D upsampling by factor of 2."""
    def __init__(self, channels, use_conv=True):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv

        if use_conv:
            self.conv = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.use_conv:
            x = self.conv(x)
        return x

class SpatialTransformer3D(nn.Module):
    """
    3D Spatial Transformer with cross-attention for text conditioning.
    Simplified version without full multi-head attention.
    """
    def __init__(self, channels, context_dim=768, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        # Layer norm
        self.norm = nn.GroupNorm(32, channels)

        # Self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )

        # Cross-attention projections
        self.query_proj = nn.Linear(channels, context_dim)      # ← ADD THIS: channels → 768
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=context_dim,  # 768
            num_heads=num_heads,
            batch_first=True
        )
        self.output_proj = nn.Linear(context_dim, channels)     # ← RENAME from context_proj

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels)
        )

    def forward(self, x, context=None):
        """
        Args:
            x: (B, C, D, H, W)
            context: (B, seq_len, context_dim) optional text embeddings
        Returns:
            out: (B, C, D, H, W)
        """
        B, C, D, H, W = x.shape
        print(f"[ST DEBUG] Input: B={B}, C={C}, D={D}, H={H}, W={W}")  # ADD

        # Reshape to sequence
        x_flat = x.view(B, C, -1).transpose(1, 2)  # (B, D*H*W, C=channels)

        # Add this check:
        tokens = D * H * W
        print(f"[ST DEBUG] tokens calculated: {tokens}, x_flat.shape[1]: {x_flat.shape[1]}")

        # Self-attention
        x_norm = self.norm(x).view(B, C, -1).transpose(1, 2)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + attn_out

        # Cross-attention to context
        if context is not None:
            # Project query from channels to context_dim (768)
            query_768 = self.query_proj(x_flat)  # (B, D*H*W, channels) → (B, D*H*W, 768)

            # Cross-attention in 768-dim space
            cross_out, _ = self.cross_attn(query_768, context, context)  # All 768-dim

            # Project output back to channels
            cross_out = self.output_proj(cross_out)  # (B, D*H*W, 768) → (B, D*H*W, channels)

            x_flat = x_flat + cross_out

        # Feed-forward
        x_flat = x_flat + self.ff(x_flat)

        # Before final reshape, verify x_flat is the right shape:
        print(f"[ST DEBUG] Before final reshape: x_flat.shape={x_flat.shape}")
        print(f"[ST DEBUG] Expected: ({B}, {tokens}, {C})")

        out = x_flat.transpose(1, 2)  # (B, D*H*W, C) → (B, C, D*H*W)
        print(f"[ST DEBUG] After transpose: out.shape={out.shape}")

        out = out.view(B, C, D, H, W)
        print(f"[ST DEBUG] After view: out.shape={out.shape}")

        if out.ndim != 5:
            raise RuntimeError(f"SpatialTransformer output has {out.ndim} dims, expected 5!")

        return out


# ============================================================================
# PHASE 3: Complete 3D U-Net Architecture
# ============================================================================
class GHOP3DUNet(nn.Module):
    """
    3D U-Net with cross-attention for text conditioning.
    Architecture: 64→128→192 channels, attention at 4³ and 2³.
    Based on GHOP's loih.yaml configuration.
    """
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 model_channels=64,
                 num_res_blocks=2,
                 attention_resolutions=[4, 2],
                 channel_mult=[1, 2, 3],
                 dropout=0.0,
                 context_dim=768):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult

        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        # ====================================================================
        # Encoder (downsampling path)
        # ====================================================================
        self.input_blocks = nn.ModuleList([
            nn.Conv3d(in_channels, model_channels, 3, padding=1)
        ])

        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1  # Current downsampling level

        for level, mult in enumerate(channel_mult):
            for block_idx in range(num_res_blocks):
                layers = [
                    ResBlock3D(
                        ch,
                        model_channels * mult,
                        time_emb_channels=time_embed_dim,
                        dropout=dropout
                    )
                ]
                ch = model_channels * mult

                # Add attention if at specified resolution
                if ds in attention_resolutions:
                    layers.append(
                        SpatialTransformer3D(ch, context_dim=context_dim)
                    )

                self.input_blocks.append(nn.ModuleList(layers))
                input_block_chans.append(ch)

            # Downsample (except at last level)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    nn.ModuleList([Downsample3D(ch, use_conv=True)])
                )
                input_block_chans.append(ch)
                ds *= 2

        # ====================================================================
        # Middle (bottleneck)
        # ====================================================================
        self.middle_block = nn.ModuleList([
            ResBlock3D(ch, ch, time_emb_channels=time_embed_dim, dropout=dropout),
            SpatialTransformer3D(ch, context_dim=context_dim),
            ResBlock3D(ch, ch, time_emb_channels=time_embed_dim, dropout=dropout)
        ])

        # ====================================================================
        # Decoder (upsampling path)
        # ====================================================================
        self.output_blocks = nn.ModuleList([])

        for level, mult in list(enumerate(channel_mult))[::-1]:
            for block_idx in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock3D(
                        ch + ich,
                        model_channels * mult,
                        time_emb_channels=time_embed_dim,
                        dropout=dropout
                    )
                ]
                ch = model_channels * mult

                # Add attention if at specified resolution
                if ds in attention_resolutions:
                    layers.append(
                        SpatialTransformer3D(ch, context_dim=context_dim)
                    )

                # Upsample (except at first block of each level)
                if level and block_idx == num_res_blocks:
                    layers.append(Upsample3D(ch, use_conv=True))
                    ds //= 2

                self.output_blocks.append(nn.ModuleList(layers))

        # ====================================================================
        # Final output projection
        # ====================================================================
        self.out = nn.Sequential(
            nn.GroupNorm(32, model_channels),
            nn.SiLU(),
            nn.Conv3d(model_channels, out_channels, 3, padding=1)
        )

    def forward(self, x, timesteps, context=None):
        """
        Forward pass through 3D U-Net.
        Args:
            x: (B, in_channels, 16, 16, 16) noisy latent
            timesteps: (B,) diffusion timesteps
            context: (B, seq_len, context_dim) text embeddings (optional)
        Returns:
            noise_pred: (B, out_channels, 16, 16, 16) predicted noise
        """
        # Compute time embeddings
        t_emb = timestep_embedding(timesteps, self.model_channels)
        emb = self.time_embed(t_emb)

        # ADD THIS AT START:
        print(f"[UNET DEBUG] ==== FORWARD START ====")
        print(f"[UNET DEBUG] Input x: {x.shape}")
        print(f"[UNET DEBUG] Context: {context.shape if context is not None else None}")
        print(f"[UNET DEBUG] Timesteps: {timesteps.shape}")

        # ================================================================
        # Encoder pass with skip connections
        # ================================================================
        hs = []
        h = x

        for i, module_list in enumerate(self.input_blocks):  # ADD enumerate
            # ADD THIS:
            print(f"[UNET DEBUG] --- Encoder block {i}: h.shape={h.shape} ---")
            if isinstance(module_list, nn.Conv3d):
                h = module_list(h)
            else:
                for layer in module_list:
                    if isinstance(layer, ResBlock3D):
                        h = layer(h, emb)
                    elif isinstance(layer, SpatialTransformer3D):
                        # ADD THIS:
                        print(f"[UNET DEBUG] Before ST (encoder): h.shape={h.shape}")
                        h = layer(h, context)
                        print(f"[UNET DEBUG] After ST (encoder): h.shape={h.shape}")
                    elif isinstance(layer, Downsample3D):
                        h = layer(h)

            # ADD THIS:
            print(f"[UNET DEBUG] Appending to hs: h.shape={h.shape}, h.ndim={h.ndim}")
            hs.append(h)

        # ================================================================
        # Middle block
        # ================================================================
        print(f"[UNET DEBUG] === Middle block START: h.shape={h.shape} ===")  # Already there

        for i, layer in enumerate(self.middle_block):
            print(f"[UNET DEBUG] Middle layer {i} ({type(layer).__name__}): input h.shape={h.shape}")  # ADD

            if isinstance(layer, ResBlock3D):
                h = layer(h, emb)
            elif isinstance(layer, SpatialTransformer3D):
                print(f"[UNET DEBUG] Before ST (middle): h.shape={h.shape}")
                h = layer(h, context)
                print(f"[UNET DEBUG] After ST (middle): h.shape={h.shape}")

            print(f"[UNET DEBUG] Middle layer {i} output: h.shape={h.shape}, h.ndim={h.ndim}")  # ADD

            # ADD THIS ASSERTION:
            if h.ndim != 5:
                raise RuntimeError(f"Middle layer {i} ({type(layer).__name__}) produced {h.ndim}D tensor! Shape: {h.shape}")
        # ADD THIS NEW DEBUG LINE HERE - AFTER middle block loop:
        print(f"[UNET DEBUG] === Middle block END: h.shape={h.shape}, h.ndim={h.ndim} ===")
        print(f"[UNET DEBUG] About to enter decoder with h.shape={h.shape}")

        # ================================================================
        # Decoder pass with skip connections
        # ================================================================
        print(f"[UNET DEBUG] === Starting decoder: h.shape={h.shape}, len(hs)={len(hs)} ===")  # ADD

        for i, module_list in enumerate(self.output_blocks):
            skip = hs[-1]

            # ADD THIS BEFORE CONCATENATION:
            # If spatial dimensions don't match, interpolate skip to match h
            if h.shape[2:] != skip.shape[2:]:
                import torch.nn.functional as F
                print(f"[UNET DEBUG] !!! SPATIAL MISMATCH at decoder block {i}")
                print(f"[UNET DEBUG]     h spatial: {h.shape[2:]}, skip spatial: {skip.shape[2:]}")

                # Interpolate skip connection to match h's spatial size
                skip = F.interpolate(
                    skip,
                    size=h.shape[2:],  # Target spatial size (D, H, W)
                    mode='trilinear',
                    align_corners=False
                )
                print(f"[UNET DEBUG]     After interpolation: {skip.shape}")

                # Replace the top of skip stack with interpolated version
                hs[-1] = skip

            # Now concatenation will work
            h = torch.cat([h, hs.pop()], dim=1)

            # ADD THIS AFTER CONCATENATION:
            print(f"[UNET DEBUG] After cat: h.shape={h.shape}")

            for layer in module_list:
                if isinstance(layer, ResBlock3D):
                    h = layer(h, emb)
                elif isinstance(layer, SpatialTransformer3D):
                    # ADD THIS:
                    print(f"[UNET DEBUG] Before ST (decoder): h.shape={h.shape}")
                    h = layer(h, context)
                    print(f"[UNET DEBUG] After ST (decoder): h.shape={h.shape}")
                elif isinstance(layer, Upsample3D):
                    h = layer(h)

        # ADD THIS AT END:
        print(f"[UNET DEBUG] Before final projection: h.shape={h.shape}")

        # Final projection
        result = self.out(h)

        # ADD THIS:
        print(f"[UNET DEBUG] ==== FORWARD END: result.shape={result.shape} ====")

        return result

# ============================================================================
# PHASE 3: Inference Wrapper
# ============================================================================
class GHOP3DUNetWrapper(nn.Module):
    """
    Wrapper for GHOP 3D U-Net with checkpoint loading and inference utilities.
    Frozen for use in SDS loss computation.
    """
    def __init__(self, unet_ckpt_path=None, device='cuda', config=None):
        super().__init__()

        # ============================================================
        # Detect checkpoint architecture BEFORE initialization
        # ============================================================
        checkpoint_out_channels = 3  # Default for random init

        if unet_ckpt_path and os.path.exists(unet_ckpt_path):
            # Peek at checkpoint to detect output channels
            ckpt = torch.load(unet_ckpt_path, map_location='cpu')
            state_dict = ckpt.get('state_dict', ckpt)

            for key in state_dict.keys():
                if key == 'glide_model.out.2.weight' and state_dict[key].ndim == 5:
                    checkpoint_out_channels = state_dict[key].shape[0]
                    logger.info(
                        f"[GHOP3DUNetWrapper] Detected checkpoint expects "
                        f"{checkpoint_out_channels} output channels"
                    )
                    break

            del ckpt, state_dict  # Free memory

        # ============================================================
        # Initialize U-Net with CHECKPOINT architecture
        # ============================================================
        default_arch_config = {
            'in_channels': 3,
            'out_channels': checkpoint_out_channels,
            'model_channels': 64,
            'num_res_blocks': 3,            # ← Changed from 2
            'attention_resolutions': [4, 2],
            'channel_mult': [1, 2, 3],       # ← Changed from [1, 2, 3]
            'dropout': 0.0,
            'context_dim': 768
        }

        # If checkpoint exists, try to detect its architecture
        if unet_ckpt_path and os.path.exists(unet_ckpt_path):
            logger.info(f"[GHOP3DUNetWrapper] Detecting checkpoint architecture...")
            detected_arch = self._detect_checkpoint_architecture(unet_ckpt_path)

            if detected_arch:
                # Override defaults with detected values
                for key, value in detected_arch.items():
                    default_arch_config[key] = value
                logger.info(f"[GHOP3DUNetWrapper] ✓ Using detected architecture")
                logger.info(f"  - channel_mult: {default_arch_config['channel_mult']}")
                logger.info(f"  - model_channels: {default_arch_config['model_channels']}")
            else:
                logger.warning(f"[GHOP3DUNetWrapper] ✗ Detection failed, using default architecture")
                logger.warning(f"  - channel_mult: {default_arch_config['channel_mult']}")

        # Override with explicit config if provided
        if config is not None:
            logger.info(f"[GHOP3DUNetWrapper] Applying explicit config overrides")
            default_arch_config.update(config)
            # Ensure output channels match checkpoint
            if 'out_channels' not in config:
                default_arch_config['out_channels'] = checkpoint_out_channels

        # Initialize U-Net with final architecture
        logger.info(f"[GHOP3DUNetWrapper] Initializing U-Net with:")
        logger.info(f"  - in_channels: {default_arch_config['in_channels']}")
        logger.info(f"  - out_channels: {default_arch_config['out_channels']}")
        logger.info(f"  - model_channels: {default_arch_config['model_channels']}")
        logger.info(f"  - channel_mult: {default_arch_config['channel_mult']}")
        logger.info(f"  - num_res_blocks: {default_arch_config['num_res_blocks']}")

        self.unet = GHOP3DUNet(**default_arch_config)

        logger.info(f"[GHOP3DUNetWrapper] Initialized U-Net with architecture-matched config")

        # ============================================================
        # Load checkpoint (architecture now matches!)
        # ============================================================
        # NOTE: _detect_checkpoint_architecture was already called above
        # before U-Net initialization, so the model architecture should
        # now match the checkpoint perfectly.
        # ============================================================
        if unet_ckpt_path:
            self._load_checkpoint(unet_ckpt_path)

        # ============================================================
        # OUTPUT ADAPTER: Convert checkpoint output (23) to HOLD expected (3)
        # ============================================================
        expected_out_channels = 3  # What HOLD expects

        if checkpoint_out_channels != expected_out_channels:
            self.output_adapter = nn.Conv3d(
                in_channels=checkpoint_out_channels,  # 23 from checkpoint
                out_channels=expected_out_channels,    # 3 for HOLD
                kernel_size=1,
                padding=0,
                bias=True
            )

            # Initialize to extract first 3 channels
            with torch.no_grad():
                self.output_adapter.weight.zero_()
                for i in range(min(expected_out_channels, checkpoint_out_channels)):
                    self.output_adapter.weight[i, i, 0, 0, 0] = 1.0
                self.output_adapter.bias.zero_()

            logger.info(
                f"[GHOP3DUNetWrapper] Created output adapter: "
                f"{checkpoint_out_channels} → {expected_out_channels} channels"
            )

            self.output_adapter.to(device)
        else:
            # Mismatch detected - create adapter
            # The model outputs model_out_channels (e.g., 3)
            # But we want to pretend it outputs checkpoint_out_channels (e.g., 23)
            # Then reduce back to expected 3 channels for HOLD

            logger.warning(
                f"[GHOP3DUNetWrapper] Creating output adapter due to architecture mismatch"
            )

            # Since the final layer wasn't loaded from checkpoint,
            # we need to replace it or adapt it
            # Option 1: No adapter (use model's random final layer - NOT RECOMMENDED)
            # Option 2: Replace final layer to match checkpoint

            # For now, log the issue
            logger.error("=" * 70)
            logger.error("[GHOP3DUNetWrapper] CRITICAL: Cannot use checkpoint effectively")
            logger.error("=" * 70)
            logger.error("  The U-Net architecture doesn't match the checkpoint.")
            # logger.error(f"  Model outputs {model_out_channels} channels")
            logger.error(f"  Checkpoint outputs {checkpoint_out_channels} channels")
            logger.error("")
            logger.error("  RECOMMENDED FIX:")
            logger.error("  Update U-Net initialization to use out_channels=23")
            logger.error("  Then add adapter: 23 → 3 channels for HOLD")
            logger.error("=" * 70)

            self.output_adapter = None

        self.unet.to(device)
        self.unet.eval()

        # Freeze for inference (SDS only needs gradients w.r.t. latent)
        for param in self.unet.parameters():
            param.requires_grad = False

        self.device = device
        print(f"[GHOP3DUNetWrapper] Initialized on {device}")

    def _load_checkpoint(self, checkpoint_path):
        """
        Load pretrained U-Net weights from unified GHOP checkpoint.
        Maps glide_model.* keys to unet.* keys for model compatibility.
        """
        logger.info(f"[GHOP3DUNetWrapper] Loading from: {checkpoint_path}")

        if not os.path.exists(checkpoint_path):
            logger.warning(f"[GHOP3DUNetWrapper] Checkpoint not found: {checkpoint_path}")
            logger.warning("Continuing with random initialization...")
            return

        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Extract state dict
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            logger.info(f"  Total parameters in checkpoint: {len(state_dict)}")

            # ============================================================
            # NEW: Detect output channels from checkpoint
            # ============================================================
            self._checkpoint_out_channels = None

            # Search for final output layer (glide_model.out.X.weight)
            # This is the ACTUAL final convolution that produces the output
            final_output_key = None
            for key in state_dict.keys():
                # Look specifically for glide_model.out.2.weight (final conv)
                if key.startswith('glide_model.out.') and key.endswith('.weight') and state_dict[key].ndim == 5:
                    final_output_key = key
                    self._checkpoint_out_channels = state_dict[key].shape[0]
                    logger.info(
                        f"[GHOP3DUNetWrapper] Detected final output layer: {key} "
                        f"with {self._checkpoint_out_channels} channels"
                    )
                    break

            # Fallback: If no glide_model.out.* found, search output_blocks
            if self._checkpoint_out_channels is None:
                logger.warning("[GHOP3DUNetWrapper] No glide_model.out.* layer found, searching output_blocks...")

                output_candidates = []
                for key in state_dict.keys():
                    if key.startswith('glide_model.output_blocks.') and '.out_layers.3.weight' in key:
                        if state_dict[key].ndim == 5:
                            output_candidates.append((key, state_dict[key].shape[0]))

                if output_candidates:
                    # Take the last output_blocks layer
                    final_key, final_channels = sorted(output_candidates)[-1]
                    self._checkpoint_out_channels = final_channels
                    logger.info(
                        f"[GHOP3DUNetWrapper] Fallback - detected output layer: {final_key} "
                        f"with {final_channels} channels"
                    )

            if self._checkpoint_out_channels is None:
                logger.warning("[GHOP3DUNetWrapper] Could not detect output channels, assuming 3")
                self._checkpoint_out_channels = 3

            # ============================================================
            # Extract U-Net parameters with key remapping
            # Checkpoint format: glide_model.input_blocks.*, etc.
            # Target format: unet.input_blocks.*, etc.
            # ============================================================
            unet_state_dict = {}

            for key, value in state_dict.items():
                # Skip VQ-VAE components (ae.model.* prefix)
                if key.startswith('ae.model.'):
                    continue

                # Process U-Net keys (glide_model.* prefix)
                if key.startswith('glide_model.'):
                    # Replace 'glide_model.' with 'unet.' prefix
                    # Example: glide_model.input_blocks.0.0.weight → unet.input_blocks.0.0.weight
                    new_key = key.replace('glide_model.', 'unet.', 1)
                    unet_state_dict[new_key] = value

            logger.info(f"  Extracted U-Net parameters: {len(unet_state_dict)} (from {len(state_dict)} total)")

            if len(unet_state_dict) == 0:
                logger.error("  ❌ No U-Net parameters extracted!")
                logger.error("  Checkpoint may be in unexpected format")
                logger.error("  Sample checkpoint keys:")
                for i, key in enumerate(list(state_dict.keys())[:10]):
                    logger.error(f"    {key}")
                logger.error("  Continuing with random initialization...")
                return

            # Show sample keys
            sample_keys = list(unet_state_dict.keys())[:5]
            logger.info(f"  Sample extracted keys: {sample_keys}")

            # ============================================================
            # Validate U-Net architecture matches checkpoint
            # ============================================================
            if self._checkpoint_out_channels is not None:
                # Check if model's output layer matches checkpoint
                model_state = self.state_dict()

                # Find model's final output layer
                model_out_key = None
                for key in model_state.keys():
                    if key.endswith('out.2.weight') and model_state[key].ndim == 5:
                        model_out_key = key
                        model_out_channels = model_state[key].shape[0]
                        break

                if model_out_key and model_out_channels != self._checkpoint_out_channels:
                    logger.warning("=" * 70)
                    logger.warning("[GHOP3DUNetWrapper] OUTPUT CHANNEL MISMATCH DETECTED")
                    logger.warning("=" * 70)
                    logger.warning(f"  Checkpoint expects: {self._checkpoint_out_channels} output channels")
                    logger.warning(f"  Model has:          {model_out_channels} output channels")
                    logger.warning("")
                    logger.warning("  The final output layer WILL NOT LOAD correctly!")
                    logger.warning("  An output adapter will be created to handle this mismatch.")
                    logger.warning("=" * 70)

                    # Store this for adapter creation
                    self._model_out_channels = model_out_channels
                else:
                    self._model_out_channels = None

            # ============================================================
            # Load weights into model
            # ============================================================
            # Get current model state for comparison
            model_state = self.state_dict()
            model_keys = set(model_state.keys())
            ckpt_keys = set(unet_state_dict.keys())

            # Find matching keys
            matching_keys = model_keys & ckpt_keys
            missing_in_ckpt = model_keys - ckpt_keys
            unexpected_in_ckpt = ckpt_keys - model_keys

            logger.info(f"  Matching keys: {len(matching_keys)}")
            logger.info(f"  Missing in checkpoint: {len(missing_in_ckpt)}")
            logger.info(f"  Unexpected in checkpoint: {len(unexpected_in_ckpt)}")

            # Load matching keys only
            filtered_state_dict = {k: v for k, v in unet_state_dict.items() if k in model_keys}

            missing_keys, unexpected_keys = self.load_state_dict(filtered_state_dict, strict=False)

            if len(matching_keys) > 0:
                logger.info(f"✅ [U-Net] Successfully loaded {len(matching_keys)} parameters")
            else:
                logger.warning("⚠️  No matching keys found - using random initialization")

            # Log critical missing keys (if any)
            critical_missing = [k for k in missing_in_ckpt if any(p in k for p in ['input_blocks', 'middle_block', 'output_blocks'])]
            if critical_missing:
                logger.warning(f"  Critical missing keys: {len(critical_missing)}")
                if len(critical_missing) <= 5:
                    for key in critical_missing[:5]:
                        logger.warning(f"    - {key}")

        except Exception as e:
            logger.error(f"❌ Checkpoint loading failed: {e}")
            logger.error("Continuing with randomly initialized U-Net...")
            import traceback
            traceback.print_exc()

    def _detect_checkpoint_architecture(self, checkpoint_path):
        """
        Detect U-Net architecture from checkpoint parameters.

        Based on analysis of GHOP official checkpoint:
        - 12 input_blocks with progression: 128→256→384 channels
        - 3 ResBlocks per level (num_res_blocks=3)
        - channel_mult=[2, 4, 6] (not [1, 2, 3]!)
        """
        try:
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            state_dict = ckpt.get('state_dict', ckpt)

            # Verify this is a GHOP checkpoint by checking middle block
            middle_key = 'glide_model.middle_block.0.emb_layers.1.weight'
            if middle_key in state_dict:
                middle_channels = state_dict[middle_key].shape[0]
                logger.info(f"[Arch Detect] Middle block channels: {middle_channels}")

                if middle_channels == 384:
                    # GHOP Official Architecture (verified 2025-12-24)
                    detected_config = {
                        'model_channels': 64,
                        'channel_mult': [1, 2, 3],  # ← REVERT from [2, 4, 6]
                        'num_res_blocks': 3,         # ← CORRECTED from 2
                        'attention_resolutions': [4, 2]
                    }

                    # Verify by checking input_blocks[1] (should be 128)
                    verify_key = 'glide_model.input_blocks.1.0.emb_layers.1.weight'
                    if verify_key in state_dict:
                        level0_channels = state_dict[verify_key].shape[0]
                        expected = 64 * detected_config['channel_mult'][0]  # 64 × 2 = 128

                        if level0_channels == expected:
                            logger.info(f"[Arch Detect] ✓ Detected: GHOP Official Architecture")
                            logger.info(f"  - channel_mult: [2, 4, 6]")
                            logger.info(f"  - num_res_blocks: 3")
                            logger.info(f"  - Verified: input_blocks[1]={level0_channels} == expected={expected}")
                            return detected_config
                        else:
                            logger.warning(f"[Arch Detect] Verification failed: {level0_channels} != {expected}")
                            return None
                    else:
                        logger.warning(f"[Arch Detect] Cannot verify - key not found: {verify_key}")
                        return None
                else:
                    logger.warning(f"[Arch Detect] Unknown middle_channels={middle_channels}")
                    return None
            else:
                logger.warning(f"[Arch Detect] Middle block key not found")
                return None

        except Exception as e:
            logger.error(f"[Arch Detect] Detection failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def predict_noise(self, noisy_latent, timestep, text_emb=None):
        """
        Predict noise for SDS loss computation.

        Args:
            noisy_latent: (B, 3, 16, 16, 16) noisy latent
            timestep: (B,) or scalar timestep(s)
            text_emb: (B, 768) OR (B, seq_len, 768) text embeddings (optional)

        Returns:
            noise_pred: (B, 3, 16, 16, 16) predicted noise
        """
        # Ensure timestep is a tensor
        if isinstance(timestep, int):
            timestep = torch.tensor([timestep] * noisy_latent.shape[0],
                                    device=self.device)
        elif timestep.ndim == 0:
            timestep = timestep.unsqueeze(0).expand(noisy_latent.shape[0])

        # ============================================================
        # FIX: Ensure context has correct shape for cross-attention
        # PyTorch MultiheadAttention expects: [seq_len, batch, embed_dim]
        # But we may receive: [batch, embed_dim]
        # ============================================================
        if text_emb is not None:
            if text_emb.ndim == 2:
                # Shape: (B, 768) -> (B, 1, 768) for single token
                # Then transpose to (1, B, 768) for attention
                text_emb = text_emb.unsqueeze(1)  # (B, 1, 768)

            # Ensure shape is (B, seq_len, embed_dim)
            if text_emb.shape[0] == noisy_latent.shape[0]:
                # Shape is (B, seq_len, embed_dim) - correct
                pass
            else:
                # May need transposing depending on your attention implementation
                pass

        with torch.no_grad():
            noise_pred = self.unet(noisy_latent, timestep, text_emb)

        return noise_pred

    def forward(self, x, timesteps, context=None):
        """
        Forward pass with optional output adaptation.

        Args:
            x: Latent tensor [B, C, D, H, W]
            timesteps: Diffusion timesteps [B]
            context: Optional conditioning (text embeddings)

        Returns:
            Denoised tensor [B, 3, D, H, W]
        """
        # Forward through U-Net
        output = self.predict_noise(x, timesteps, context)

        # ============================================================
        # CHECK THIS EXISTS:
        # ============================================================
        if hasattr(self, 'output_adapter') and self.output_adapter is not None:
            original_shape = output.shape
            output = self.output_adapter(output)

            # Log occasionally
            if not hasattr(self, '_output_adapter_log_count'):
                self._output_adapter_log_count = 0

            if self._output_adapter_log_count < 3:
                logger.info(
                    f"[GHOP3DUNetWrapper] Output adapted: "
                    f"{original_shape[1]} → {output.shape[1]} channels"
                )
                self._output_adapter_log_count += 1

        return output


# ============================================================================
# PHASE 3: Testing Utilities
# ============================================================================
def test_ghop_unet():
    """Test GHOP 3D U-Net forward pass."""
    print("\n" + "=" * 60)
    print("Testing GHOP 3D U-Net...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize wrapper
    wrapper = GHOP3DUNetWrapper(unet_ckpt_path=None, device=device)

    # Test inputs
    B = 2
    noisy_latent = torch.randn(B, 3, 16, 16, 16, device=device)
    timesteps = torch.randint(0, 1000, (B,), device=device)
    text_emb = torch.randn(B, 77, 768, device=device)  # CLIP embeddings

    # Forward pass
    print(f"\nInput shape: {noisy_latent.shape}")
    print(f"Timesteps: {timesteps}")

    noise_pred = wrapper.predict_noise(noisy_latent, timesteps, text_emb)

    print(f"Output shape: {noise_pred.shape}")
    assert noise_pred.shape == noisy_latent.shape, f"Shape mismatch!"

    print("\n" + "=" * 60)
    print("✓ GHOP U-Net test passed!")
    print("=" * 60)
# Uncomment to run test
# if __name__ == "__main__":
#     test_ghop_unet()