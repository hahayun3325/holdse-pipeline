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
                nn.Linear(time_emb_channels, out_channels)
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

        # Add time embedding if provided
        if self.use_time_emb and emb is not None:
            emb_out = self.emb_layers(emb)[:, :, None, None, None]
            h = h + emb_out

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

        # Self-attention (simplified)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )

        # Cross-attention to text context (placeholder)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )

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

        # Reshape to sequence: (B, C, D, H, W) -> (B, D*H*W, C)
        x_flat = x.view(B, C, -1).transpose(1, 2)

        # Self-attention
        x_norm = self.norm(x).view(B, C, -1).transpose(1, 2)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + attn_out

        # Cross-attention to context (if provided)
        if context is not None:
            # Project context to match channels
            # Note: In full implementation, add a learned projection layer
            cross_out, _ = self.cross_attn(x_flat, context, context)
            x_flat = x_flat + cross_out

        # Feed-forward
        x_flat = x_flat + self.ff(x_flat)

        # Reshape back
        out = x_flat.transpose(1, 2).view(B, C, D, H, W)

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

        # ================================================================
        # Encoder pass with skip connections
        # ================================================================
        hs = []
        h = x

        for module_list in self.input_blocks:
            if isinstance(module_list, nn.Conv3d):
                h = module_list(h)
            else:
                for layer in module_list:
                    if isinstance(layer, ResBlock3D):
                        h = layer(h, emb)
                    elif isinstance(layer, SpatialTransformer3D):
                        h = layer(h, context)
                    elif isinstance(layer, Downsample3D):
                        h = layer(h)
            hs.append(h)

        # ================================================================
        # Middle block
        # ================================================================
        for i, layer in enumerate(self.middle_block):
            if isinstance(layer, ResBlock3D):
                h = layer(h, emb)
            elif isinstance(layer, SpatialTransformer3D):
                h = layer(h, context)

        # ================================================================
        # Decoder pass with skip connections
        # ================================================================
        for module_list in self.output_blocks:
            # Concatenate skip connection
            h = torch.cat([h, hs.pop()], dim=1)

            for layer in module_list:
                if isinstance(layer, ResBlock3D):
                    h = layer(h, emb)
                elif isinstance(layer, SpatialTransformer3D):
                    h = layer(h, context)
                elif isinstance(layer, Upsample3D):
                    h = layer(h)

        # Final projection
        return self.out(h)


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

        # Initialize U-Net with default or provided config
        if config is None:
            self.unet = GHOP3DUNet(
                in_channels=3,
                out_channels=3,
                model_channels=64,
                num_res_blocks=2,
                attention_resolutions=[4, 2],
                channel_mult=[1, 2, 3],
                dropout=0.0,
                context_dim=768
            )
        else:
            self.unet = GHOP3DUNet(**config)

        # Load pretrained weights if provided
        if unet_ckpt_path:
            self._load_checkpoint(unet_ckpt_path)

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
        Note: GHOP checkpoint may not contain standard U-Net keys.
        """
        from loguru import logger

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
            # CRITICAL: Extract U-Net parameters (exclude VQ-VAE)
            # ============================================================
            unet_state_dict = {}

            # Define VQ-VAE patterns to EXCLUDE
            vqvae_patterns = [
                'encoder.',
                'decoder.',
                'quantize.',
                'quant_conv',
                'post_quant'
            ]

            for key, value in state_dict.items():
                # Skip VQ-VAE components
                if any(pattern in key for pattern in vqvae_patterns):
                    continue

                # Include everything else (U-Net diffusion model)
                new_key = key

                # Remove common prefixes
                if 'ae.model.' in new_key:
                    new_key = new_key.replace('ae.model.', '')

                # Try to map to unet submodule
                if not new_key.startswith('unet.'):
                    # Check if this looks like a U-Net parameter
                    if any(pattern in new_key for pattern in ['conv', 'norm', 'attn', 'resblock', 'down', 'up', 'middle']):
                        new_key = 'unet.' + new_key

                unet_state_dict[new_key] = value

            logger.info(f"  Extracted non-VQ-VAE parameters: {len(unet_state_dict)}")

            if len(unet_state_dict) == 0:
                logger.warning("  ⚠️  No U-Net parameters extracted!")
                logger.warning("  Sample checkpoint keys:")
                for i, key in enumerate(list(state_dict.keys())[:10]):
                    logger.warning(f"    {key}")
                logger.warning("  Continuing with random initialization...")
                return

            # Show sample keys
            sample_keys = list(unet_state_dict.keys())[:5]
            logger.info(f"  Sample extracted keys: {sample_keys}")

            # Try loading
            missing_keys, unexpected_keys = self.load_state_dict(unet_state_dict, strict=False)

            logger.info(f"  Missing keys: {len(missing_keys)}")
            logger.info(f"  Unexpected keys: {len(unexpected_keys)}")

            # Check if any meaningful keys were loaded
            model_keys = set(self.state_dict().keys())
            loaded_keys = set(unet_state_dict.keys()) & model_keys

            if len(loaded_keys) > 0:
                logger.info(f"✓ U-Net weights loaded: {len(loaded_keys)} parameters matched")
            else:
                logger.warning("⚠️  No matching keys found - using random initialization")
                logger.warning("  This is expected if GHOP checkpoint doesn't contain standard U-Net")

        except Exception as e:
            logger.warning(f"⚠️  Checkpoint loading failed: {e}")
            logger.warning("  Continuing with randomly initialized U-Net...")

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
        """Forward pass (alias for predict_noise)."""
        return self.predict_noise(x, timesteps, context)


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