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

# ============================================================================
# GHOP Transformer Components (from original GHOP codebase)
# Source: /home/fredcui/Projects/ghop/ddpm3d/models/attention.py
# ============================================================================

def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module

def Normalize(in_channels, num_groups=32):
    """GroupNorm wrapper matching GHOP convention."""
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

def init_weights(m):
    """Initialize weights for conv/linear layers."""
    if isinstance(m, (nn.Conv3d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class CrossAttention(nn.Module):
    """
    Multi-head cross-attention with separate Q/K/V projections.
    Matches GHOP checkpoint structure with to_q, to_k, to_v parameters.
    """
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim or query_dim

        self.scale = dim_head ** -0.5
        self.heads = heads

        # Separate projections (matches checkpoint)
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = context if context is not None else x
        k = self.to_k(context)
        v = self.to_v(context)

        # Reshape for multi-head attention
        # Use einops if available, otherwise manual reshape
        try:
            from einops import rearrange
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        except ImportError:
            # Manual reshape as fallback
            b, n, _ = q.shape
            q = q.view(b, n, h, -1).permute(0, 2, 1, 3).reshape(b * h, n, -1)
            k = k.view(b, -1, h, k.shape[-1] // h).permute(0, 2, 1, 3).reshape(b * h, -1, k.shape[-1] // h)
            v = v.view(b, -1, h, v.shape[-1] // h).permute(0, 2, 1, 3).reshape(b * h, -1, v.shape[-1] // h)

        # Attention computation
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        if mask is not None:
            mask = mask.reshape(mask.shape[0], -1)
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = mask[:, None, :].repeat(h, 1, 1)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)
        out = torch.einsum('b i j, b j d -> b i d', attn, v)

        # Reshape back
        try:
            from einops import rearrange
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        except ImportError:
            b = x.shape[0]
            n = out.shape[1]
            out = out.view(b, h, n, -1).permute(0, 2, 1, 3).reshape(b, n, -1)

        return self.to_out(out)

class GEGLU(nn.Module):
    """Gated Linear Unit with GELU activation (used in feedforward)."""
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    """
    Feedforward network with optional GLU gating.
    Matches GHOP checkpoint ff.net structure.
    """
    def __init__(self, dim, dim_out=None, mult=4, glu=True, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim

        project_in = GEGLU(dim, inner_dim) if glu else nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        )

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

class BasicTransformerBlock(nn.Module):
    """
    Full transformer block matching GHOP checkpoint structure.
    Contains: self-attention + cross-attention + feedforward, each with pre-LayerNorm.
    """
    def __init__(self, dim, n_heads, d_head, dropout=0.0, context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )  # Self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(
            query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )  # Cross-attention
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        # Use gradient checkpointing if enabled (saves memory during training)
        if self.checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(self._forward, x, context)
        else:
            return self._forward(x, context)

    def _forward(self, x, context=None):
        # Pre-norm transformer architecture
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x

class SpatialTransformer3D(nn.Module):
    """
    3D Spatial Transformer matching GHOP checkpoint architecture.
    Uses full BasicTransformerBlock with separate Q/K/V projections and feedforward.

    Architecture:
    1. GroupNorm normalization
    2. Conv3D projection (in_channels → inner_dim)
    3. Reshape to sequence (b c d h w → b (d h w) c)
    4. Apply transformer blocks
    5. Reshape back to spatial (b (d h w) c → b c d h w)
    6. Conv3D projection (inner_dim → in_channels)
    7. Residual connection
    """
    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0.0, context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        # Project input to inner dimension
        self.proj_in = nn.Conv3d(
            in_channels, inner_dim, kernel_size=1, stride=1, padding=0
        )

        # Stack of transformer blocks
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim
            )
            for d in range(depth)
        ])

        # Project back to original channels (zero-initialized for stability)
        self.proj_out = zero_module(
            nn.Conv3d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
        )

        # Initialize weights
        self.apply(init_weights)

    def forward(self, x, context=None):
        """
        Args:
            x: (B, C, D, H, W) spatial tensor
            context: (B, seq_len, context_dim) optional conditioning
        Returns:
            (B, C, D, H, W) transformed spatial tensor
        """
        b, c, d, h, w = x.shape
        x_in = x

        # Normalize and project to inner dimension
        x = self.norm(x)
        x = self.proj_in(x)

        # Reshape to sequence format for transformer
        try:
            from einops import rearrange
            x = rearrange(x, 'b c d h w -> b (d h w) c')
        except ImportError:
            # Fallback without einops
            x = x.permute(0, 2, 3, 4, 1).reshape(b, d * h * w, -1)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, context=context)

        # Reshape back to spatial format
        try:
            from einops import rearrange
            x = rearrange(x, 'b (d h w) c -> b c d h w', d=d, h=h, w=w)
        except ImportError:
            # Fallback without einops
            x = x.reshape(b, d, h, w, -1).permute(0, 4, 1, 2, 3)

        # Project back to original channels and add residual
        x = self.proj_out(x)
        return x + x_in


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
            nn.ModuleList([nn.Conv3d(in_channels, model_channels, 3, padding=1)])
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
                    # Calculate attention heads based on channel dimensions
                    # Pattern: d_head=16, n_heads varies by stage (8 for 128ch, 12 for 192ch)
                    d_head = 16  # From checkpoint analysis: num_head_channels=16
                    n_heads = ch // d_head

                    layers.append(
                        SpatialTransformer3D(
                            in_channels=ch,
                            n_heads=n_heads,
                            d_head=d_head,
                            depth=1,
                            dropout=dropout,
                            context_dim=context_dim
                        )
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
        # Middle block attention configuration
        d_head = 16
        n_heads = ch // d_head

        self.middle_block = nn.ModuleList([
            ResBlock3D(ch, ch, time_emb_channels=time_embed_dim, dropout=dropout),
            SpatialTransformer3D(
                in_channels=ch,
                n_heads=n_heads,
                d_head=d_head,
                depth=1,
                dropout=dropout,
                context_dim=context_dim
            ),
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
                    # Calculate attention heads based on channel dimensions
                    d_head = 16
                    n_heads = ch // d_head

                    layers.append(
                        SpatialTransformer3D(
                            in_channels=ch,
                            n_heads=n_heads,
                            d_head=d_head,
                            depth=1,
                            dropout=dropout,
                            context_dim=context_dim
                        )
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

        for i, module_list in enumerate(self.input_blocks):
            print(f"[UNET DEBUG] --- Encoder block {i}: h.shape={h.shape} ---")
            if isinstance(module_list, nn.Conv3d):
                h = module_list(h)
            else:
                for layer in module_list:
                    if isinstance(layer, ResBlock3D):
                        h = layer(h, emb)
                    elif isinstance(layer, SpatialTransformer3D):
                        print(f"[UNET DEBUG] Before ST (encoder): h.shape={h.shape}")
                        h = layer(h, context)
                        print(f"[UNET DEBUG] After ST (encoder): h.shape={h.shape}")
                    elif isinstance(layer, Downsample3D):
                        h = layer(h)
                    elif isinstance(layer, nn.Conv3d):  # ✅ ADD THIS!
                        print(f"[UNET DEBUG] Applying Conv3d: {h.shape[1]} → {layer.out_channels} channels")
                        h = layer(h)
                        print(f"[UNET DEBUG] After Conv3d: h.shape={h.shape}")

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
                elif isinstance(layer, nn.Conv3d):
                    print(f"[UNET DEBUG] Applying Conv3d (decoder): {h.shape[1]} → {layer.out_channels} channels")
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
        # Detect checkpoint input/output channels BEFORE initialization
        # ============================================================
        checkpoint_out_channels = 3  # Default
        checkpoint_in_channels = 3   # Default

        if unet_ckpt_path and os.path.exists(unet_ckpt_path):
            logger.info("[GHOP3DUNetWrapper] Detecting checkpoint I/O channels...")
            ckpt = torch.load(unet_ckpt_path, map_location='cpu')
            state_dict = ckpt.get('state_dict', ckpt)

            # Detect output channels
            for key in state_dict.keys():
                if key == 'glide_model.out.2.weight' and state_dict[key].ndim == 5:
                    checkpoint_out_channels = state_dict[key].shape[0]
                    logger.info(
                        f"[GHOP3DUNetWrapper] Detected output: {checkpoint_out_channels} channels"
                    )
                    break

            # Detect input channels
            for key in state_dict.keys():
                if key == 'glide_model.input_blocks.0.0.weight' and state_dict[key].ndim == 5:
                    checkpoint_in_channels = state_dict[key].shape[1]
                    logger.info(
                        f"[GHOP3DUNetWrapper] Detected input: {checkpoint_in_channels} channels"
                    )
                    break

            del ckpt, state_dict  # Single cleanup
        # ============================================================
        # Initialize U-Net with CHECKPOINT architecture
        # ============================================================
        default_arch_config = {
            'in_channels': checkpoint_in_channels,
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

            try:
                detected_arch = self._detect_checkpoint_architecture(unet_ckpt_path)

                # ✅ DEBUG: Log what was detected
                logger.info(f"[GHOP3DUNetWrapper] Detected architecture keys: {list(detected_arch.keys())}")

                # Merge detected config
                for key, value in detected_arch.items():
                    default_arch_config[key] = value

                # ✅ DEBUG: Log what's in config after merge
                logger.info(f"[GHOP3DUNetWrapper] Config after merge: {list(default_arch_config.keys())}")
                logger.info(f"[GHOP3DUNetWrapper] attention_resolutions in config: {'attention_resolutions' in default_arch_config}")
                if 'attention_resolutions' in default_arch_config:
                    logger.info(f"  Value: {default_arch_config['attention_resolutions']}")

                logger.info(f"[GHOP3DUNetWrapper] ✓ Using detected architecture from checkpoint")
                logger.info(f"  - model_channels: {default_arch_config['model_channels']}")
                logger.info(f"  - channel_mult: {default_arch_config['channel_mult']}")
                logger.info(f"  - num_res_blocks: {default_arch_config['num_res_blocks']}")  # ADD THIS LINE
                logger.info(f"  - attention_resolutions: {default_arch_config.get('attention_resolutions', 'NOT SET')}")  # ADD THIS
            except Exception as e:
                # ❌ CRITICAL: Detection completely failed - cannot proceed safely
                logger.error(f"[GHOP3DUNetWrapper] ❌ Architecture detection failed: {e}")
                logger.error(f"Cannot initialize U-Net with incompatible architecture!")
                logger.error(f"Checkpoint path: {unet_ckpt_path}")
                logger.error(f"\nThis will cause parameter loading failures and OOM errors.")
                logger.error(f"Please verify checkpoint is valid GHOP checkpoint.")

                # Show what would be used as fallback
                logger.error(f"\nFallback architecture (likely INCOMPATIBLE):")
                logger.error(f"  - model_channels: {default_arch_config['model_channels']}")
                logger.error(f"  - channel_mult: {default_arch_config['channel_mult']}")

                raise RuntimeError(
                    f"U-Net architecture detection failed. Cannot safely initialize with "
                    f"default architecture as it will cause shape mismatches. "
                    f"Original error: {e}"
                )

        # Override with explicit config if provided
        if config is not None:
            logger.info(f"[GHOP3DUNetWrapper] Applying explicit config overrides")
            default_arch_config.update(config)
            # Ensure output channels match checkpoint
            if 'out_channels' not in config:
                default_arch_config['out_channels'] = checkpoint_out_channels

        # ============================================================
        # PRE-INITIALIZATION: Detect input channels from checkpoint
        # ============================================================
        expected_in_channels = 3  # What HOLD provides

        if checkpoint_in_channels != expected_in_channels:
            logger.info(f"[GHOP3DUNetWrapper] PRE-INIT: Input channel mismatch detected")
            logger.info(f"[GHOP3DUNetWrapper] Checkpoint expects: {checkpoint_in_channels} channels")
            logger.info(f"[GHOP3DUNetWrapper] HOLD provides: {expected_in_channels} channels")

            # ✅ USE CHECKPOINT'S EXACT VALUE - No rounding needed!
            # The first conv (input_blocks.0.0) is a regular Conv3d without GroupNorm,
            # so it can accept ANY number of input channels.
            # GroupNorm operates on the OUTPUT of the first conv (64 channels), not the input.
            target_channels = checkpoint_in_channels  # Use 23 exactly

            # ✅ CRITICAL: Update U-Net architecture to match checkpoint
            default_arch_config['in_channels'] = target_channels
            logger.info(
                f"[GHOP3DUNetWrapper] ✅ U-Net architecture updated: in_channels = {target_channels}"
            )

            logger.info(
                f"[GHOP3DUNetWrapper] Input adapter will convert: "
                f"{expected_in_channels} → {target_channels} channels"
            )
            logger.info(
                f"[GHOP3DUNetWrapper] U-Net first conv will be initialized as: {target_channels} → 64 channels"
            )
            logger.info(
                f"[GHOP3DUNetWrapper] This allows checkpoint's first conv to load correctly"
            )
            logger.info(
                f"[GHOP3DUNetWrapper] Note: GroupNorm operates on first conv OUTPUT "
                f"(64 channels), not input ({target_channels} channels)"
            )
        else:
            target_channels = expected_in_channels
            logger.info(f"[GHOP3DUNetWrapper] No input adapter needed")

        # Store target channels for adapter creation later
        self._target_input_channels = target_channels

        # Initialize U-Net with final architecture
        logger.info(f"[GHOP3DUNetWrapper] Initializing U-Net with:")
        logger.info(f"  - in_channels: {default_arch_config['in_channels']}")
        logger.info(f"  - out_channels: {default_arch_config['out_channels']}")
        logger.info(f"  - model_channels: {default_arch_config['model_channels']}")
        logger.info(f"  - channel_mult: {default_arch_config['channel_mult']}")
        logger.info(f"  - num_res_blocks: {default_arch_config['num_res_blocks']}")
        logger.info(f"  - attention_resolutions: {default_arch_config.get('attention_resolutions', 'NOT SET')}")  # ADD THIS

        # ✅ DEBUG: Print FULL config being passed
        logger.info(f"[GHOP3DUNetWrapper] Full config keys being passed: {list(default_arch_config.keys())}")
        self.unet = GHOP3DUNet(**default_arch_config)

        logger.info(f"[GHOP3DUNetWrapper] Initialized U-Net with architecture-matched config")

        # ============================================================
        # POST-INITIALIZATION VALIDATION
        # ============================================================
        logger.info("\n" + "="*70)
        logger.info("[GHOP3DUNetWrapper] POST-INITIALIZATION VALIDATION")
        logger.info("="*70)

        # Check first layer's GroupNorm
        first_block = self.unet.input_blocks[0]

        # Find GroupNorm in first block
        for name, module in first_block.named_modules():
            if isinstance(module, nn.GroupNorm):
                logger.info(f"Found GroupNorm in first block:")
                logger.info(f"  - num_groups: {module.num_groups}")
                logger.info(f"  - num_channels: {module.num_channels}")

                if module.num_channels % module.num_groups != 0:
                    logger.error(f"  ❌ INCOMPATIBLE: {module.num_channels} % {module.num_groups} = {module.num_channels % module.num_groups}")
                    logger.error(f"  This indicates the fix didn't work - U-Net still has wrong channel count!")
                    raise RuntimeError(
                        f"U-Net GroupNorm incompatibility: {module.num_channels} channels "
                        f"not divisible by {module.num_groups} groups. "
                        f"Input adapter adjustment failed to apply before U-Net initialization."
                    )
                else:
                    logger.info(f"  ✅ COMPATIBLE: {module.num_channels} % {module.num_groups} = 0")
                break

        logger.info("="*70 + "\n")

        # ============================================================
        # VALIDATE: Count created parameters
        # ============================================================
        total_unet_params = sum(1 for _ in self.unet.state_dict().keys())
        logger.info(f"[GHOP3DUNetWrapper] U-Net has {total_unet_params} parameters")

        # Count blocks for validation
        input_blocks_count = len(self.unet.input_blocks)
        output_blocks_count = len(self.unet.output_blocks)
        logger.info(f"[GHOP3DUNetWrapper] Architecture summary:")
        logger.info(f"  - Input blocks: {input_blocks_count}")
        logger.info(f"  - Output blocks: {output_blocks_count}")
        logger.info(f"  - Middle block layers: {len(self.unet.middle_block)}")

        # ============================================================
        # VALIDATE: Ensure adapters are compatible with checkpoint loading
        # ============================================================
        if checkpoint_in_channels != default_arch_config['in_channels']:
            logger.warning("="*70)
            logger.warning("[GHOP3DUNetWrapper] ARCHITECTURE MISMATCH WARNING")
            logger.warning("="*70)
            logger.warning(f"  Checkpoint expects: {checkpoint_in_channels} input channels")
            logger.warning(f"  U-Net initialized with: {default_arch_config['in_channels']} input channels")
            logger.warning("")
            logger.warning("  This means checkpoint's first layer weights CANNOT be loaded!")
            logger.warning("  First layer will use RANDOM initialization.")
            logger.warning("")
            logger.warning("  IMPACT: SDS loss quality will be degraded.")
            logger.warning("  RECOMMENDATION: Fine-tune first layer or use compatible architecture.")
            logger.warning("="*70 + "\n")

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

        # ============================================================
        # INPUT ADAPTER: Create adapter using pre-computed target channels
        # ============================================================
        expected_in_channels = 3  # What HOLD provides

        if hasattr(self, '_target_input_channels') and self._target_input_channels != expected_in_channels:
            self.input_adapter = nn.Conv3d(
                in_channels=expected_in_channels,      # 3 from HOLD
                out_channels=self._target_input_channels,  # 32 (pre-computed)
                kernel_size=1,
                padding=0,
                bias=True
            )

            # Initialize: replicate first 3 channels, zero-pad remaining
            with torch.no_grad():
                self.input_adapter.weight.zero_()
                for i in range(min(expected_in_channels, self._target_input_channels)):
                    self.input_adapter.weight[i, i, 0, 0, 0] = 1.0
                self.input_adapter.bias.zero_()

            logger.info(
                f"[GHOP3DUNetWrapper] Created input adapter: "
                f"{expected_in_channels} → {self._target_input_channels} channels"
            )

            if self._target_input_channels != checkpoint_in_channels:
                logger.warning(
                    f"[GHOP3DUNetWrapper] Note: U-Net has {self._target_input_channels} input channels "
                    f"but checkpoint has {checkpoint_in_channels}"
                )
                logger.warning(f"[GHOP3DUNetWrapper] First layer will use random weights")

            self.input_adapter.to(device)
        else:
            self.input_adapter = None
            logger.info(f"[GHOP3DUNetWrapper] No input adapter created")

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

            # ============================================================
            # DIAGNOSTIC: Analyze what's missing
            # ============================================================
            if len(unexpected_in_ckpt) > 0:
                logger.warning("=" * 70)
                logger.warning("[DIAGNOSTIC] Analyzing unexpected checkpoint parameters")
                logger.warning("=" * 70)

                # Categorize unexpected keys
                transformer_unexpected = [k for k in unexpected_in_ckpt if 'transformer_blocks' in k]
                resblock_unexpected = [k for k in unexpected_in_ckpt if
                                       any(x in k for x in ['in_layers', 'out_layers', 'emb_layers'])]
                other_unexpected = [k for k in unexpected_in_ckpt if
                                    k not in transformer_unexpected and k not in resblock_unexpected]

                logger.warning(f"  Transformer-related: {len(transformer_unexpected)}")
                logger.warning(f"  ResBlock-related: {len(resblock_unexpected)}")
                logger.warning(f"  Other: {len(other_unexpected)}")

                if len(transformer_unexpected) > 0:
                    logger.warning(f"\n  Sample transformer unexpected keys:")
                    for key in sorted(transformer_unexpected)[:5]:
                        logger.warning(f"    {key}")

                if len(resblock_unexpected) > 0:
                    logger.warning(f"\n  Sample ResBlock unexpected keys:")
                    for key in sorted(resblock_unexpected)[:5]:
                        logger.warning(f"    {key}")

                if len(other_unexpected) > 0:
                    logger.warning(f"\n  Sample other unexpected keys:")
                    for key in sorted(other_unexpected)[:5]:
                        logger.warning(f"    {key}")

                # Check if model is missing entire layers
                logger.warning(f"\n  Checking model architecture completeness:")

                # Count layers in checkpoint vs model
                ckpt_input_blocks = len(set([k.split('.')[1] for k in ckpt_keys if k.startswith('unet.input_blocks.')]))
                model_input_blocks = len(
                    set([k.split('.')[1] for k in model_keys if k.startswith('unet.input_blocks.')]))

                ckpt_output_blocks = len(
                    set([k.split('.')[1] for k in ckpt_keys if k.startswith('unet.output_blocks.')]))
                model_output_blocks = len(
                    set([k.split('.')[1] for k in model_keys if k.startswith('unet.output_blocks.')]))

                logger.warning(f"    Input blocks - Checkpoint: {ckpt_input_blocks}, Model: {model_input_blocks}")
                logger.warning(f"    Output blocks - Checkpoint: {ckpt_output_blocks}, Model: {model_output_blocks}")

                if ckpt_input_blocks != model_input_blocks or ckpt_output_blocks != model_output_blocks:
                    logger.error("    ❌ ARCHITECTURE MISMATCH: Model has different number of blocks than checkpoint!")
                    logger.error("    This explains why parameters don't load fully.")

                logger.warning("=" * 70)

            # ============================================================
            # STRATEGY A: Load with size-mismatch tolerance
            # ============================================================
            # Filter out keys with shape mismatches
            compatible_state_dict = {}
            incompatible_keys = []

            for key, value in unet_state_dict.items():
                if key in model_keys:
                    model_param = model_state[key]
                    if model_param.shape == value.shape:
                        compatible_state_dict[key] = value
                    else:
                        incompatible_keys.append(
                            f"{key}: checkpoint {value.shape} vs model {model_param.shape}"
                        )
                        logger.warning(f"  Shape mismatch: {key}")
                        logger.warning(f"    Checkpoint: {value.shape}")
                        logger.warning(f"    Model:      {model_param.shape}")

            if len(incompatible_keys) > 0:
                logger.warning(f"  Found {len(incompatible_keys)} parameters with shape mismatches")
                logger.warning(f"  These will NOT be loaded (keeping random initialization)")

            logger.info(f"  Loading {len(compatible_state_dict)} compatible parameters...")

            try:
                missing_keys, unexpected_keys = self.load_state_dict(compatible_state_dict, strict=False)

                # Calculate actual loaded count
                actually_loaded = len(compatible_state_dict) - len(missing_keys)
                logger.info(f"  ✅ Actually loaded: {actually_loaded}/{len(unet_state_dict)} parameters")

                if actually_loaded < len(unet_state_dict) * 0.9:  # Less than 90% loaded
                    logger.error("=" * 70)
                    logger.error("⚠️  WARNING: Less than 90% of checkpoint parameters loaded!")
                    logger.error(f"  Expected: {len(unet_state_dict)}")
                    logger.error(f"  Loaded:   {actually_loaded}")
                    logger.error(f"  Missing:  {len(unet_state_dict) - actually_loaded}")
                    logger.error("=" * 70)
            except RuntimeError as e:
                logger.error(f"❌ Loading failed even with compatible keys: {e}")
                logger.error("This indicates a deeper architecture mismatch")
                raise

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
        Detect U-Net architecture from checkpoint structure.
        Trust checkpoint architecture over config defaults.
        Based on analysis of GHOP official checkpoint:
        - 12 input_blocks with progression: 128→256→384 channels
        - 3 ResBlocks per level (num_res_blocks=3)
        - channel_mult=[2, 4, 6] (not [1, 2, 3]!)
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)

            logger.info(f"[Arch Detect] Analyzing checkpoint structure...")

            # ============================================================
            # Step 1: Find all input block convolution layers
            # ============================================================
            input_block_info = {}

            for key in state_dict.keys():
                # Match pattern: glide_model.input_blocks.{block_id}.{layer_id}.weight
                if key.startswith('glide_model.input_blocks.') and key.endswith('.weight'):
                    tensor = state_dict[key]

                    # Only consider 3D convolutions (5D tensors: out_ch, in_ch, D, H, W)
                    if tensor.ndim == 5:
                        # Extract block ID
                        parts = key.split('.')
                        block_id = int(parts[2])  # glide_model.input_blocks.{block_id}.layer.weight
                        out_channels = tensor.shape[0]

                        # Store first conv layer for each block
                        if block_id not in input_block_info:
                            input_block_info[block_id] = {
                                'out_channels': out_channels,
                                'key': key
                            }

            logger.info(f"[Arch Detect] Found {len(input_block_info)} input blocks with convolutions")

            if len(input_block_info) == 0:
                raise ValueError("No input block convolutions found in checkpoint")

            # ============================================================
            # Step 2: Extract channel progression from ALL blocks
            # ============================================================
            # Sort by block ID and get ALL blocks (not just first 6!)
            sorted_blocks = sorted(input_block_info.items())

            # Get output channels for ALL blocks
            channel_sequence = [info['out_channels'] for block_id, info in sorted_blocks]

            logger.info(f"[Arch Detect] Channel progression (all blocks): {channel_sequence}")
            logger.info(f"[Arch Detect] Total blocks analyzed: {len(sorted_blocks)}")

            # Warn if suspiciously few blocks found
            if len(sorted_blocks) < 8:
                logger.warning(
                    f"[Arch Detect] Only {len(sorted_blocks)} input blocks found. "
                    f"Typical U-Net has 9-15 blocks. Checkpoint may be incomplete."
                )

            # First block determines base model_channels
            model_channels = channel_sequence[0]

            # ============================================================
            # Step 3: Detect channel_mult from unique channel values
            # ============================================================
            # Find unique channel counts (represent different resolution stages)
            unique_channels = []
            seen = set()
            for ch in channel_sequence:
                if ch not in seen:
                    unique_channels.append(ch)
                    seen.add(ch)

            # Calculate multipliers relative to base
            channel_mult = tuple(ch // model_channels for ch in unique_channels)

            logger.info(f"[Arch Detect] Detected architecture:")
            logger.info(f"  - model_channels: {model_channels}")
            logger.info(f"  - channel_mult: {channel_mult}")
            logger.info(f"  - unique_channels: {unique_channels}")

            # ============================================================
            # Step 3.5: VALIDATE against middle block
            # ============================================================
            middle_block_keys = [k for k in state_dict.keys()
                                 if 'glide_model.middle_block.' in k
                                 and 'in_layers.2.weight' in k
                                 and state_dict[k].ndim == 5]

            if middle_block_keys:
                middle_key = middle_block_keys[0]
                middle_channels = state_dict[middle_key].shape[0]
                expected_middle = model_channels * max(channel_mult)

                logger.info(f"[Arch Detect] Middle block validation:")
                logger.info(f"  Found: {middle_channels} channels")
                logger.info(f"  Expected: {expected_middle} channels")

                if middle_channels != expected_middle:
                    logger.error(
                        f"[Arch Detect] ❌ MISMATCH! Detection found channel_mult={channel_mult} "
                        f"but middle_block has {middle_channels} channels (expected {expected_middle})"
                    )
                    raise ValueError(
                        f"Architecture detection failed validation: "
                        f"middle_block={middle_channels} != expected={expected_middle}"
                    )

                logger.info(f"[Arch Detect] ✓ Validation passed")

            # ============================================================
            # Step 4: Detect num_res_blocks from MIDDLE BLOCK (definitive)
            # ============================================================
            # Middle block always uses same num_res_blocks as input/output stages
            middle_block_keys = [k for k in state_dict.keys()
                                 if k.startswith('glide_model.middle_block.')]

            # Extract ResBlock IDs from middle block
            middle_res_block_ids = set()
            for key in middle_block_keys:
                # Pattern: glide_model.middle_block.{res_block_id}.layer.weight
                parts = key.split('.')
                if len(parts) >= 3:
                    try:
                        block_id = int(parts[2])
                        middle_res_block_ids.add(block_id)
                    except ValueError:
                        continue

            # Number of res blocks = number of unique IDs
            num_res_blocks = len(middle_res_block_ids)

            if num_res_blocks == 0:
                logger.warning(
                    f"[Arch Detect] Could not detect num_res_blocks from middle_block. "
                    f"Using fallback estimation."
                )
                # Fallback: estimate from input_blocks structure
                # Count how many blocks exist at first channel stage
                first_stage_channels = unique_channels[0]
                first_stage_blocks = [bid for bid, ch in sorted_blocks if ch == first_stage_channels]
                # Subtract 1 for initial conv (block 0), rest are res blocks
                num_res_blocks = max(2, len(first_stage_blocks) - 1)
            else:
                logger.info(f"[Arch Detect] ✓ Detected num_res_blocks from middle_block: {num_res_blocks}")
                logger.info(f"  Middle block ResBlock IDs: {sorted(middle_res_block_ids)}")

            logger.info(f"  - num_res_blocks (detected): {num_res_blocks}")

            # ============================================================
            # Step 5: Validate total input blocks should match
            # ============================================================
            # ============================================================
            # Validate: Total input blocks should match
            # ============================================================
            # With num_res_blocks per stage + downsamples + initial conv
            # Expected blocks = 1 (initial) + sum(num_res_blocks per stage) + (num_stages - 1) downsamples
            num_stages = len(channel_mult)

            # Simplified: each stage has (num_res_blocks + 1 downsample), except last stage
            expected_blocks = 1  # initial conv
            for i in range(num_stages):
                expected_blocks += num_res_blocks
                if i < num_stages - 1:  # Add downsample except for last stage
                    expected_blocks += 1

            actual_blocks = len(sorted_blocks)

            logger.info(f"\n[Arch Detect] Input blocks validation:")
            logger.info(f"  Actual input blocks in checkpoint: {actual_blocks}")
            logger.info(f"  Expected with detected architecture: {expected_blocks}")

            if abs(actual_blocks - expected_blocks) > 2:  # Allow small tolerance
                logger.warning(
                    f"[Arch Detect] ⚠️  Block count mismatch! "
                    f"Actual={actual_blocks}, Expected={expected_blocks}. "
                    f"Architecture detection may be incomplete."
                )
            else:
                logger.info(f"[Arch Detect] ✓ Block count validation passed")

            # ============================================================
            # Step 6: Detect attention configuration from checkpoint
            # ============================================================
            logger.info(f"[Arch Detect] Analyzing attention layer configuration...")

            # Find input blocks with transformer attention
            attention_input_blocks = set()
            for key in state_dict.keys():
                if 'glide_model.input_blocks' in key and 'transformer_blocks' in key:
                    block_id = int(key.split('.')[2])
                    attention_input_blocks.add(block_id)

            logger.info(f"[Arch Detect] Found attention in input_blocks: {sorted(attention_input_blocks)}")

            # Map blocks to stages to determine attention_resolutions
            # With num_res_blocks and channel_mult, determine which stages have attention
            attention_stages = set()
            block_to_stage_map = {}
            current_block = 0

            # Initial conv
            current_block += 1

            # For each stage
            for stage_idx, mult in enumerate(channel_mult):
                stage_start = current_block
                # Res blocks in this stage
                for res_idx in range(num_res_blocks):
                    block_to_stage_map[current_block] = stage_idx
                    if current_block in attention_input_blocks:
                        attention_stages.add(stage_idx)
                    current_block += 1

                # Downsample (except last stage)
                if stage_idx < len(channel_mult) - 1:
                    current_block += 1

            logger.info(f"[Arch Detect] Block to stage mapping: {block_to_stage_map}")
            logger.info(f"[Arch Detect] Stages with attention: {sorted(attention_stages)}")

            # Convert stage indices to attention_resolutions
            # attention_resolutions uses downsampling factors
            # Stage 0 = resolution factor 1 (no downsample)
            # Stage 1 = resolution factor 2 (one downsample)
            # Stage 2 = resolution factor 4 (two downsamples)
            # etc.
            if attention_stages:
                attention_resolutions = tuple(2 ** stage_idx for stage_idx in sorted(attention_stages))
                logger.info(f"[Arch Detect] Detected attention_resolutions: {attention_resolutions}")
            else:
                attention_resolutions = tuple()
                logger.info(f"[Arch Detect] No attention layers detected")

            # Check for transformer blocks (vs simple attention)
            has_transformer = any('transformer_blocks' in k for k in state_dict.keys()
                                  if 'glide_model.' in k)

            # Count transformer depth and heads
            if has_transformer:
                # Find a sample transformer block to inspect
                sample_keys = [k for k in state_dict.keys()
                              if 'transformer_blocks.0.attn1.to_q.weight' in k]
                if sample_keys:
                    sample_key = sample_keys[0]
                    to_q_weight = state_dict[sample_key]
                    # Shape is typically [inner_dim, channels]
                    # inner_dim = num_heads * head_dim
                    # Common: 8 heads with head_dim = channels / 8
                    inner_dim = to_q_weight.shape[0]
                    channels = to_q_weight.shape[1]
                    num_head_channels = channels // 8  # Common default
                    transformer_depth = 1  # Count by checking max transformer_blocks.X

                    # Find max transformer depth
                    for key in state_dict.keys():
                        if 'transformer_blocks.' in key:
                            depth_match = key.split('transformer_blocks.')[1].split('.')[0]
                            try:
                                depth = int(depth_match) + 1
                                transformer_depth = max(transformer_depth, depth)
                            except ValueError:
                                pass

                    logger.info(f"[Arch Detect] Transformer configuration:")
                    logger.info(f"  - transformer_depth: {transformer_depth}")
                    logger.info(f"  - num_head_channels: {num_head_channels}")
                    logger.info(f"  - context_dim (estimated): {inner_dim}")
                else:
                    transformer_depth = 1
                    num_head_channels = -1
            else:
                transformer_depth = 1
                num_head_channels = -1

            # ============================================================
            # Return detected configuration
            # ============================================================
            # Only return parameters that GHOP3DUNet.__init__() accepts
            # Supported params: model_channels, channel_mult, num_res_blocks, attention_resolutions
            supported_config = {
                'model_channels': model_channels,
                'channel_mult': list(channel_mult),
                'num_res_blocks': num_res_blocks,
                'attention_resolutions': list(attention_resolutions),
            }

            logger.info(f"\n[Arch Detect] Returning configuration:")
            for key, value in supported_config.items():
                logger.info(f"  - {key}: {value}")

            # Log extra detected info (for debugging) but don't return it
            if has_transformer:
                logger.info(f"\n[Arch Detect] Additional detected settings (informational):")
                logger.info(f"  - transformer_depth: {transformer_depth}")
                logger.info(f"  - num_head_channels: {num_head_channels}")
                logger.info(f"  - use_spatial_transformer: {has_transformer}")

            return supported_config

        except Exception as e:
            logger.error(f"[Arch Detect] Detection failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def predict_noise(self, noisy_latent, timestep, text_emb=None):
        """
        Predict noise for SDS loss computation.

        Args:
            noisy_latent: (B, C, D, H, W) noisy latent - ALREADY ADAPTED by forward()
                          C = 32 after input adapter, or 3 if called directly
            timestep: (B,) or scalar timestep(s)
            text_emb: (B, 768) OR (B, seq_len, 768) text embeddings (optional)

        Returns:
            noise_pred: (B, C, D, H, W) predicted noise
        """
        # Ensure timestep is a tensor
        if isinstance(timestep, int):
            timestep = torch.tensor([timestep] * noisy_latent.shape[0],
                                    device=self.device)
        elif timestep.ndim == 0:
            timestep = timestep.unsqueeze(0).expand(noisy_latent.shape[0])

        # ============================================================
        # ❌ REMOVED: Input adapter application
        # ============================================================
        # The input adapter is now ONLY applied in forward()
        # If predict_noise() is called directly from external code,
        # the caller must ensure the input has the correct channel count

        # ✅ ADD VALIDATION: Check if input has expected channels
        if hasattr(self, 'input_adapter') and self.input_adapter is not None:
            expected_channels = self.input_adapter.out_channels  # Should be 32

            if noisy_latent.shape[1] != expected_channels:
                # Log warning but don't fail - this might be intentional
                if not hasattr(self, '_predict_noise_channel_warning_shown'):
                    logger.warning(
                        f"[predict_noise] Input has {noisy_latent.shape[1]} channels, "
                        f"but U-Net expects {expected_channels} channels. "
                        f"If this is called from forward(), this indicates a bug. "
                        f"If called directly, ensure input is pre-adapted."
                    )
                    self._predict_noise_channel_warning_shown = True

        # ============================================================
        # Prepare text embeddings for cross-attention
        # ============================================================
        if text_emb is not None:
            if text_emb.ndim == 2:
                text_emb = text_emb.unsqueeze(1)  # (B, 768) → (B, 1, 768)

        # ============================================================
        # Forward through U-Net (expects 32 channels after adapter)
        # ============================================================
        with torch.no_grad():
            noise_pred = self.unet(noisy_latent, timestep, text_emb)

        return noise_pred

    def forward(self, x, timesteps, context=None):
        """
        Forward pass with input/output adaptation.

        Args:
            x: Latent tensor [B, 3, D, H, W] from VQ-VAE
            timesteps: Diffusion timesteps [B]
            context: Optional conditioning (text embeddings)

        Returns:
            Denoised tensor [B, 3, D, H, W]
        """
        # ✅ ADD THIS AT ENTRY:
        if not hasattr(self, '_forward_log_count'):
            self._forward_log_count = 0

        if self._forward_log_count < 3:  # Log first 3 calls
            logger.info(f"[UNET-FORWARD] === Entry ===")
            logger.info(f"[UNET-FORWARD] Input x shape: {x.shape}")
            logger.info(f"[UNET-FORWARD] Expected: [B, 3, 6/8, 6/8, 6/8]")

            # ✅ TEXT CONDITIONING DEBUG
            logger.critical(f"[UNET-TEXT] === Checking text conditioning ===")
            logger.critical(f"[UNET-TEXT] Received context: {type(context)}")
            if context is not None:
                logger.critical(f"[UNET-TEXT] Context shape: {context.shape}")
                logger.critical(f"[UNET-TEXT] Context norm: {context.norm().item():.4f}")
                logger.critical(f"[UNET-TEXT] Context mean: {context.mean().item():.6f}")
                logger.critical(f"[UNET-TEXT] Context std: {context.std().item():.6f}")
            else:
                logger.critical(f"[UNET-TEXT] ⚠️ Context is None - no text conditioning!")

        # ============================================================
        # APPLY INPUT ADAPTER (3 → 32 channels)
        # ============================================================
        if hasattr(self, 'input_adapter') and self.input_adapter is not None:
            original_shape = x.shape
            x = self.input_adapter(x)  # [B, 3, ...] → [B, 32, ...]

            if self._forward_log_count < 3:
                logger.info(f"[UNET-FORWARD] Input adapted: {original_shape[1]} → {x.shape[1]} channels")
                logger.info(f"[UNET-FORWARD] Now calling U-Net with {x.shape[1]}-channel input")

        # ============================================================
        # FORWARD THROUGH U-NET
        # ============================================================
        if self._forward_log_count < 3:
            logger.info(f"[UNET-FORWARD] Calling self.predict_noise(x={x.shape}, t={timesteps.shape})")

        try:
            output = self.predict_noise(x, timesteps, context)

            if self._forward_log_count < 3:
                logger.info(f"[UNET-FORWARD] U-Net output shape: {output.shape}")

        except RuntimeError as e:
            if "num_groups" in str(e):
                logger.error(f"[UNET-FORWARD] ❌ GroupNorm error: {e}")
                logger.error(f"[UNET-FORWARD] Input shape to U-Net: {x.shape}")
                logger.error(f"[UNET-FORWARD] This means input_adapter fix didn't work!")
                raise
            else:
                raise

        # ============================================================
        # APPLY OUTPUT ADAPTER (23/32 → 3 channels)
        # ============================================================
        if hasattr(self, 'output_adapter') and self.output_adapter is not None:
            original_shape = output.shape
            output = self.output_adapter(output)

            if self._forward_log_count < 3:
                logger.info(f"[UNET-FORWARD] Output adapted: {original_shape[1]} → {output.shape[1]} channels")

        if self._forward_log_count < 3:
            logger.info(f"[UNET-FORWARD] === Exit: returning {output.shape} ===\n")
            self._forward_log_count += 1

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