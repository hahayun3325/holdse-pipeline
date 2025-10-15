# File: ~/Projects/holdse/code/test_checkpoint_loading.py
# Purpose: Verify VQ-VAE and U-Net can load from unified checkpoint

import torch
import sys

sys.path.insert(0, '.')

from src.model.ghop.autoencoder import GHOPVQVAEWrapper
from src.model.ghop.diffusion import GHOP3DUNetWrapper
from loguru import logger


def test_vqvae_loading():
    """Test VQ-VAE checkpoint loading"""
    logger.info("=" * 70)
    logger.info("Testing VQ-VAE Checkpoint Loading")
    logger.info("=" * 70)

    try:
        vqvae = GHOPVQVAEWrapper(
            vqvae_ckpt_path='checkpoints/ghop/last.ckpt',
            device='cpu',
            use_hand_field=True
        )

        logger.info("✓ VQ-VAE loaded successfully")

        # Test forward pass
        dummy_input = torch.randn(1, 16, 16, 16, 16)  # [B, 16, D, H, W]
        with torch.no_grad():
            z_q, loss, info = vqvae.encode(dummy_input)
            logger.info(f"  Encoded shape: {z_q.shape}")

            sdf_recon = vqvae.decode(z_q)
            logger.info(f"  Decoded shape: {sdf_recon.shape}")

        logger.info("✓ VQ-VAE forward pass successful")
        return True

    except Exception as e:
        logger.error(f"✗ VQ-VAE loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_unet_loading():
    """Test U-Net checkpoint loading"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("Testing U-Net Checkpoint Loading")
    logger.info("=" * 70)

    try:
        unet = GHOP3DUNetWrapper(
            unet_ckpt_path='checkpoints/ghop/last.ckpt',
            device='cpu'
        )

        logger.info("✓ U-Net loaded successfully")

        # ============================================================
        # FIX: Test forward pass with correct input shapes
        # ============================================================
        dummy_latent = torch.randn(2, 3, 16, 16, 16)  # [B, 3, D, H, W]
        dummy_t = torch.tensor([500, 600])  # [B] timesteps

        # Test 1: Without context (unconditional)
        logger.info("\nTest 1: Unconditional prediction (no context)")
        with torch.no_grad():
            noise_pred = unet(dummy_latent, dummy_t, context=None)
            logger.info(f"  ✓ Noise prediction shape: {noise_pred.shape}")

        # Test 2: With 2D context [B, 768]
        logger.info("\nTest 2: Conditional prediction (2D context)")
        dummy_context_2d = torch.randn(2, 768)  # [B, embed_dim]
        with torch.no_grad():
            noise_pred = unet(dummy_latent, dummy_t, context=dummy_context_2d)
            logger.info(f"  ✓ Noise prediction shape: {noise_pred.shape}")

        # Test 3: With 3D context [B, seq_len, 768]
        logger.info("\nTest 3: Conditional prediction (3D context)")
        dummy_context_3d = torch.randn(2, 77, 768)  # [B, seq_len, embed_dim]
        with torch.no_grad():
            noise_pred = unet(dummy_latent, dummy_t, context=dummy_context_3d)
            logger.info(f"  ✓ Noise prediction shape: {noise_pred.shape}")

        logger.info("\n✓ U-Net forward pass successful (all tests)")
        return True

    except Exception as e:
        logger.error(f"✗ U-Net loading/testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    vqvae_ok = test_vqvae_loading()
    unet_ok = test_unet_loading()

    logger.info("")
    logger.info("=" * 70)
    logger.info("Test Summary")
    logger.info("=" * 70)
    logger.info(f"  VQ-VAE: {'✓ PASS' if vqvae_ok else '✗ FAIL'}")
    logger.info(f"  U-Net: {'✓ PASS' if unet_ok else '✗ FAIL'}")

    if vqvae_ok and unet_ok:
        logger.info("")
        logger.info("✅ All tests passed - ready for training!")
        sys.exit(0)
    else:
        logger.error("")
        logger.error("❌ Some tests failed - fix before training")
        sys.exit(1)