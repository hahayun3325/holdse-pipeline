"""
VQ-VAE for object SDF compression (64³ → 16³).

Adapted from GHOP's ddpm3d/models/autoencoder.py

PHASE 3 UPDATES:
- Added GHOPVQVAEWrapper for HOLD integration with hand field support
- Supports both 1-channel (object only) and 16-channel (object + 15 hand field) inputs
- Added encode_no_quant() method for SDS gradient flow
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf  # For future config loading if needed
from loguru import logger
import os

class VectorQuantizer(nn.Module):
    """
    Vector quantization layer with exponential moving average updates.
    """

    def __init__(self, n_embed, embed_dim, beta=0.25):
        super().__init__()
        self.n_embed = n_embed  # 8192
        self.embed_dim = embed_dim  # 3
        self.beta = beta

        # Codebook embeddings
        self.embedding = nn.Embedding(n_embed, embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_embed, 1.0 / n_embed)

    def forward(self, z):
        """
        Args:
            z: [B, C, D, H, W] continuous latent
        Returns:
            z_q: [B, C, D, H, W] quantized latent
            loss: VQ loss
            indices: [B, D, H, W] codebook indices
        """
        # Reshape: [B, C, D, H, W] -> [B*D*H*W, C]
        z_flat = rearrange(z, 'b c d h w -> (b d h w) c')

        # Compute distances to codebook vectors
        d = torch.sum(z_flat ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - \
            2 * torch.matmul(z_flat, self.embedding.weight.t())

        # Find nearest codebook entries
        min_encoding_indices = torch.argmin(d, dim=1)  # [B*D*H*W]
        z_q_flat = self.embedding(min_encoding_indices)  # [B*D*H*W, C]

        # Reshape back
        z_q = rearrange(z_q_flat, '(b d h w) c -> b c d h w',
                        b=z.shape[0], d=z.shape[2], h=z.shape[3], w=z.shape[4])

        # Compute VQ loss
        loss = F.mse_loss(z_q.detach(), z) + self.beta * F.mse_loss(z_q, z.detach())

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        indices = rearrange(min_encoding_indices, '(b d h w) -> b d h w',
                            b=z.shape[0], d=z.shape[2], h=z.shape[3], w=z.shape[4])

        return z_q, loss, indices


class Encoder3D(nn.Module):
    """3D encoder for SDF compression."""

    def __init__(self, in_channels=1, z_channels=3, ch=64, ch_mult=[1, 2, 4],
                 num_res_blocks=1):
        super().__init__()
        self.num_resolutions = len(ch_mult)

        # Input conv
        self.conv_in = nn.Conv3d(in_channels, ch, kernel_size=3, padding=1)

        # Downsampling blocks
        self.down = nn.ModuleList()
        in_ch = ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            out_ch = ch * ch_mult[i_level]

            # ResBlocks
            for i_block in range(num_res_blocks):
                block.append(ResBlock3D(in_ch, out_ch))
                in_ch = out_ch

            # Downsample
            if i_level != self.num_resolutions - 1:
                block.append(Downsample3D(in_ch))

            self.down.append(block)

        # Output
        self.norm_out = nn.GroupNorm(32, in_ch)
        self.conv_out = nn.Conv3d(in_ch, z_channels, kernel_size=3, padding=1)

    def forward(self, x):
        """
        Args:
            x: [B, 1, 64, 64, 64] input SDF
        Returns:
            z: [B, 3, 16, 16, 16] continuous latent
        """
        h = self.conv_in(x)

        for i_level, blocks in enumerate(self.down):
            for block in blocks:
                h = block(h)

        h = self.norm_out(h)
        h = F.silu(h)
        z = self.conv_out(h)

        return z


class Decoder3D(nn.Module):
    """3D decoder for SDF reconstruction."""

    def __init__(self, out_channels=1, z_channels=3, ch=64, ch_mult=[1, 2, 4],
                 num_res_blocks=1):
        super().__init__()
        self.num_resolutions = len(ch_mult)

        # Input
        in_ch = ch * ch_mult[-1]
        self.conv_in = nn.Conv3d(z_channels, in_ch, kernel_size=3, padding=1)

        # Upsampling blocks
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            out_ch = ch * ch_mult[i_level]

            # ResBlocks
            for i_block in range(num_res_blocks):
                block.append(ResBlock3D(in_ch, out_ch))
                in_ch = out_ch

            # Upsample
            if i_level != 0:
                block.append(Upsample3D(in_ch))

            self.up.append(block)

        # Output
        self.norm_out = nn.GroupNorm(32, in_ch)
        self.conv_out = nn.Conv3d(in_ch, out_channels, kernel_size=3, padding=1)

    def forward(self, z):
        """
        Args:
            z: [B, 3, 16, 16, 16] quantized latent
        Returns:
            x: [B, 1, 64, 64, 64] reconstructed SDF
        """
        h = self.conv_in(z)

        for i_level, blocks in enumerate(self.up):
            for block in blocks:
                h = block(h)

        h = self.norm_out(h)
        h = F.silu(h)
        x = self.conv_out(h)

        return x


class ResBlock3D(nn.Module):
    """3D residual block."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.skip(x)


class Downsample3D(nn.Module):
    """3D downsampling via strided convolution."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample3D(nn.Module):
    """3D upsampling via interpolation + convolution."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        return self.conv(x)


class VQModel(nn.Module):
    """
    Complete VQ-VAE model.
    """

    def __init__(self, embed_dim=3, n_embed=8192, in_channels=1,
                 out_channels=1, ch=64, ch_mult=[1, 2, 4], num_res_blocks=1):
        super().__init__()
        self.encoder = Encoder3D(in_channels, embed_dim, ch, ch_mult, num_res_blocks)
        self.decoder = Decoder3D(out_channels, embed_dim, ch, ch_mult, num_res_blocks)
        self.quantize = VectorQuantizer(n_embed, embed_dim)

    def encode(self, x):
        """
        Encode input to latent space with optional channel adaptation.

        Args:
            x: Input tensor [B, C, D, H, W] where C may not match checkpoint

        Returns:
            Latent distribution or tensor
        """
        # Apply input adapter if needed
        if self.input_adapter is not None:
            original_shape = x.shape
            x = self.input_adapter(x)  # Reduce to 1 channel

            if self.global_step % 100 == 0:  # Log occasionally (if global_step available)
                logger.debug(
                    f"[GHOPVQVAEWrapper] Input adapted: "
                    f"{original_shape} → {x.shape}"
                )

        # Forward through VQ-VAE encoder
        return self.model.encode(x)

    def encode_to_prequant(self, x):
        """Encode without quantization (for gradient flow in SDS)."""
        z = self.encoder(x)
        return z

    def decode(self, z_q):
        """Decode from quantized latent."""
        return self.decoder(z_q)

    def forward(self, x):
        """Full forward pass."""
        z_q, vq_loss, indices = self.encode(x)
        x_recon = self.decode(z_q)
        return x_recon, vq_loss


def load_vqvae(checkpoint_path, device='cuda'):
    """Load pretrained VQ-VAE from GHOP checkpoint."""
    vqvae = VQModel(
        embed_dim=3,
        n_embed=8192,
        in_channels=1,
        out_channels=1,
        ch=64,
        ch_mult=[1, 2, 4],
        num_res_blocks=1
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    vqvae.load_state_dict(checkpoint['state_dict'], strict=False)
    vqvae.eval()
    vqvae.to(device)

    # Freeze parameters
    for param in vqvae.parameters():
        param.requires_grad = False

    return vqvae


# ================================================================
# PHASE 3: GHOP-HOLD Integration Wrapper Classes
# ================================================================

class GHOPVQVAEWrapper(nn.Module):
    """
    Complete VQ-VAE wrapper for HOLD integration with hand field support.
    Extends VQModel to handle concatenated object SDF + 15-channel hand field.
    """

    def __init__(self, vqvae_ckpt_path=None, device='cuda', use_hand_field=True):
        super().__init__()

        # Determine input channels based on hand field usage
        self.use_hand_field = use_hand_field
        in_channels = 16 if use_hand_field else 1  # 1 (object) + 15 (hand field)

        # Initialize encoder with 16 channels for object + hand field
        self.encoder = Encoder3D(
            in_channels=in_channels,
            z_channels=3,
            ch=64,
            ch_mult=[1, 2, 4],
            num_res_blocks=1
        )

        # Initialize quantizer
        self.quantizer = VectorQuantizer(
            n_embed=8192,
            embed_dim=3,
            beta=0.25
        )

        # Initialize decoder (always outputs 1-channel SDF)
        self.decoder = Decoder3D(
            out_channels=1,
            z_channels=3,
            ch=64,
            ch_mult=[1, 2, 4],
            num_res_blocks=1
        )

        # Load pretrained weights if provided
        if vqvae_ckpt_path:
            self._load_checkpoint(vqvae_ckpt_path)

        # ============================================================
        # NO INPUT ADAPTER NEEDED
        # ============================================================
        # The encoder's weights were adapted during checkpoint loading
        # to accept multi-channel input (16 channels for object SDF + hand field)
        # Only the first channel has non-zero weights from checkpoint,
        # so the encoder naturally focuses on the SDF channel
        self.input_adapter = None

        self.to(device)
        self.eval()

        # Freeze for inference (SDS only needs gradients w.r.t. input)
        for param in self.parameters():
            param.requires_grad = False

    def _load_checkpoint(self, checkpoint_path):
        """
        Load pretrained VQ-VAE weights from unified GHOP checkpoint.
        Handles architecture mismatches by adapting model structure.
        """
        logger.info(f"[GHOPVQVAEWrapper] Loading from: {checkpoint_path}")

        if not os.path.exists(checkpoint_path):
            logger.error(f"[GHOPVQVAEWrapper] Checkpoint not found: {checkpoint_path}")
            logger.error("Continuing with random initialization...")
            return

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        logger.info(f"  Total parameters in checkpoint: {len(state_dict)}")

        # ============================================================
        # STEP 1: Detect checkpoint architecture
        # ============================================================
        self._checkpoint_in_channels = None
        for key in state_dict.keys():
            if key == 'ae.model.encoder.conv_in.weight':
                self._checkpoint_in_channels = state_dict[key].shape[1]
                logger.info(
                    f"[GHOPVQVAEWrapper] Checkpoint expects {self._checkpoint_in_channels} "
                    f"input channels (from {key})"
                )
                break

        if self._checkpoint_in_channels is None:
            logger.warning("[GHOPVQVAEWrapper] Could not detect input channels")
            self._checkpoint_in_channels = 1

        # ============================================================
        # STEP 2: Extract VQ-VAE parameters
        # ============================================================
        vqvae_state_dict = {}

        for key, value in state_dict.items():
            if key.startswith('ae.model.'):
                new_key = key.replace('ae.model.', '', 1)

                if any(pattern in new_key for pattern in [
                    'encoder.',
                    'decoder.',
                    'quantize.',
                    'quant_conv',
                    'post_quant'
                ]):
                    vqvae_state_dict[new_key] = value

        logger.info(f"  Extracted VQ-VAE parameters: {len(vqvae_state_dict)} (from {len(state_dict)} total)")

        # ============================================================
        # STEP 3: Handle input channel mismatch
        # ============================================================
        model_state = self.state_dict()

        # Check encoder input layer
        if 'encoder.conv_in.weight' in vqvae_state_dict and 'encoder.conv_in.weight' in model_state:
            ckpt_shape = vqvae_state_dict['encoder.conv_in.weight'].shape
            model_shape = model_state['encoder.conv_in.weight'].shape

            if ckpt_shape[1] != model_shape[1]:  # Input channels mismatch
                logger.warning(
                    f"[GHOPVQVAEWrapper] Input channel mismatch detected:"
                )
                logger.warning(f"  Checkpoint: {ckpt_shape[1]} channels")
                logger.warning(f"  Model:      {model_shape[1]} channels")
                logger.warning(f"  Adapting encoder.conv_in.weight...")

                # Create adapted weight: replicate checkpoint's first channel
                adapted_weight = torch.zeros(model_shape)

                # Strategy: Copy checkpoint weight to first channel,
                # leave rest as zero (will be handled by input adapter)
                adapted_weight[:, 0:ckpt_shape[1], :, :, :] = vqvae_state_dict['encoder.conv_in.weight']

                vqvae_state_dict['encoder.conv_in.weight'] = adapted_weight
                logger.info(f"  ✅ Adapted encoder.conv_in.weight: {ckpt_shape} → {model_shape}")

        # ============================================================
        # STEP 4: Load adapted weights
        # ============================================================
        sample_keys = list(vqvae_state_dict.keys())[:5]
        logger.info(f"  Sample extracted keys: {sample_keys}")

        model_keys = set(model_state.keys())
        ckpt_keys = set(vqvae_state_dict.keys())

        matching_keys = model_keys & ckpt_keys
        missing_in_ckpt = model_keys - ckpt_keys
        unexpected_in_ckpt = ckpt_keys - model_keys

        logger.info(f"  Matching keys: {len(matching_keys)}")
        logger.info(f"  Missing in checkpoint: {len(missing_in_ckpt)}")
        logger.info(f"  Unexpected in checkpoint: {len(unexpected_in_ckpt)}")

        try:
            missing_keys, unexpected_keys = self.load_state_dict(vqvae_state_dict, strict=False)

            if len(matching_keys) > 0:
                logger.info(f"✅ [VQ-VAE] Successfully loaded {len(matching_keys)} parameters")
            else:
                logger.warning("⚠️  No matching keys - using random initialization")

        except Exception as e:
            logger.error(f"Failed to load VQ-VAE weights: {e}")
            logger.error("Continuing with random initialization...")

    def encode(self, object_sdf, hand_field=None):
        """
        Encode object SDF (optionally with hand field) to quantized latent.

        Args:
            object_sdf: (B, 1, 64, 64, 64) - object SDF grid
            hand_field: (B, 15, 64, 64, 64) - hand skeletal distance field (optional)

        Returns:
            z_q: (B, 3, 16, 16, 16) - quantized latent
            indices: (B, 16, 16, 16) - codebook indices
            vq_loss: scalar - VQ loss (for logging, not used in SDS)
        """
        # ============================================================
        # STEP 1: Prepare input
        # ============================================================
        if self.use_hand_field and hand_field is not None:
            # Concatenate object SDF + hand field: (B, 16, 64, 64, 64)
            x = torch.cat([object_sdf, hand_field], dim=1)
        else:
            # Use only object SDF: (B, 1, 64, 64, 64)
            x = object_sdf

        # ============================================================
        # STEP 2: No adaptation needed
        # ============================================================
        # The encoder weights were adapted to accept multi-channel input
        # Log input shape for debugging
        if not hasattr(self, '_encode_log_count'):
            self._encode_log_count = 0

        if self._encode_log_count < 3:  # Log first 3 times only
            logger.info(
                f"[GHOPVQVAEWrapper] Encoding input with shape: {x.shape} "
                f"(encoder expects 16 channels, will use adapted weights)"
            )
            self._encode_log_count += 1

        # ============================================================
        # STEP 3: Encode through VQ-VAE
        # ============================================================
        with torch.no_grad():
            # Encode to continuous latent
            z = self.encoder(x)  # (B, 3, 16, 16, 16)

            # Quantize
            z_q, vq_loss, indices = self.quantizer(z)

        return z_q, indices, vq_loss

    def encode_no_quant(self, object_sdf, hand_field=None):
        """
        Encode without quantization - needed for SDS gradient flow.

        Args:
            object_sdf: (B, 1, 64, 64, 64)
            hand_field: (B, 15, 64, 64, 64) (optional)

        Returns:
            z: (B, 3, 16, 16, 16) - continuous latent (with gradients)
        """
        if self.use_hand_field and hand_field is not None:
            x = torch.cat([object_sdf, hand_field], dim=1)
        else:
            x = object_sdf

        # NO torch.no_grad() here - we need gradients for SDS!
        z = self.encoder(x)
        return z

    def decode(self, z_q):
        """
        Decode from quantized latent to SDF.

        Args:
            z_q: (B, 3, 16, 16, 16) - quantized latent

        Returns:
            sdf_recon: (B, 1, 64, 64, 64) - reconstructed SDF
        """
        with torch.no_grad():
            sdf_recon = self.decoder(z_q)
        return sdf_recon

    def forward(self, object_sdf, hand_field=None):
        """
        Full forward pass: encode → quantize → decode.

        Args:
            object_sdf: (B, 1, 64, 64, 64)
            hand_field: (B, 15, 64, 64, 64) (optional)

        Returns:
            sdf_recon: (B, 1, 64, 64, 64) - reconstructed SDF
            vq_loss: scalar - VQ loss
            indices: (B, 16, 16, 16) - codebook indices
        """
        z_q, indices, vq_loss = self.encode(object_sdf, hand_field)
        sdf_recon = self.decode(z_q)
        return sdf_recon, vq_loss, indices


class VQVAEEncoder(nn.Module):
    """
    Simplified encoder wrapper for explicit 16-channel input.
    Alias for Encoder3D with in_channels=16.
    """

    def __init__(self, in_channels=16, z_channels=3, ch=64, ch_mult=[1, 2, 4]):
        super().__init__()
        self.encoder = Encoder3D(
            in_channels=in_channels,
            z_channels=z_channels,
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=1
        )

    def forward(self, x):
        """
        Args:
            x: (B, 16, 64, 64, 64) - object SDF + hand field
        Returns:
            z: (B, 3, 16, 16, 16) - continuous latent
        """
        return self.encoder(x)


class VQVAEDecoder(nn.Module):
    """
    Simplified decoder wrapper.
    Alias for Decoder3D with out_channels=1.
    """

    def __init__(self, z_channels=3, out_channels=1, ch=64, ch_mult=[1, 2, 4]):
        super().__init__()
        self.decoder = Decoder3D(
            out_channels=out_channels,
            z_channels=z_channels,
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=1
        )

    def forward(self, z_q):
        """
        Args:
            z_q: (B, 3, 16, 16, 16)
        Returns:
            sdf_recon: (B, 1, 64, 64, 64)
        """
        return self.decoder(z_q)