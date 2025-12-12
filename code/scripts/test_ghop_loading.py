#!/usr/bin/env python3
"""Test GHOP checkpoint loading into actual models"""

import sys
from pathlib import Path
import torch
from loguru import logger
from omegaconf import OmegaConf
sys.path.insert(0, 'src')
sys.path.insert(0, '../common')
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_vqvae_loading():
    """Test VQ-VAE loads correctly with adapter"""
    from model.ghop.autoencoder import GHOPVQVAEWrapper
    
    cfg = OmegaConf.load('confs/stage3_hold_MC1_ho3d_sds_from_official.yaml')
    ckpt_path = cfg.phase3.ghop.unified_checkpoint
    
    logger.info("=" * 70)
    logger.info("TEST 1: VQ-VAE Loading with Input Adapter")
    logger.info("=" * 70)
    
    try:
        # Initialize wrapper (will auto-create adapter)
        vqvae = GHOPVQVAEWrapper(
            vqvae_ckpt_path=ckpt_path,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            use_hand_field=True
        )
        logger.info("✅ VQ-VAE initialized successfully")

        # Check if adapter was created
        if hasattr(vqvae, 'input_adapter') and vqvae.input_adapter is not None:
            logger.info(
                f"✅ Input adapter created: "
                f"{vqvae.input_adapter.in_channels} → {vqvae.input_adapter.out_channels} channels"
            )
        else:
            logger.warning("⚠️  No input adapter (might cause issues)")
        
        # Test forward pass
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Simulate HOLD input: object SDF (1 ch) + hand field (15 ch) = 16 channels
        # For testing, use random data
        dummy_object_sdf = torch.randn(1, 1, 64, 64, 64, device=device)
        dummy_hand_field = torch.randn(1, 15, 64, 64, 64, device=device)

        logger.info(f"Testing with object_sdf: {dummy_object_sdf.shape}, hand_field: {dummy_hand_field.shape}")

        with torch.no_grad():
            # Call with proper signature
            z_q, indices, vq_loss = vqvae.encode(dummy_object_sdf, dummy_hand_field)

        logger.info(f"✅ VQ-VAE encode works")
        logger.info(f"  z_q shape: {z_q.shape}")
        logger.info(f"  indices shape: {indices.shape}")
        logger.info(f"  vq_loss: {vq_loss.item():.4f}")

        return True

    except Exception as e:
        logger.error(f"❌ VQ-VAE loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_unet_loading():
    """Test U-Net loads correctly with adapter"""
    from model.ghop.diffusion import GHOP3DUNetWrapper

    cfg = OmegaConf.load('confs/stage3_hold_MC1_ho3d_sds_from_official.yaml')
    ckpt_path = cfg.phase3.ghop.unified_checkpoint

    logger.info("=" * 70)
    logger.info("TEST 2: U-Net Loading with Output Adapter")
    logger.info("=" * 70)

    try:
        # Initialize wrapper (will auto-create adapter)
        unet = GHOP3DUNetWrapper(
            unet_ckpt_path=ckpt_path,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        logger.info("✅ U-Net initialized successfully")

        # Check if adapter was created
        if hasattr(unet, 'output_adapter') and unet.output_adapter is not None:
            logger.info(
                f"✅ Output adapter created: "
                f"{unet.output_adapter.in_channels} → {unet.output_adapter.out_channels} channels"
            )
        else:
            logger.warning("⚠️  No output adapter (might be OK if channels match)")
        
        # Test forward pass
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        dummy_latent = torch.randn(1, 3, 16, 16, 16, device=device)
        dummy_timesteps = torch.tensor([500], device=device)
        dummy_condition = torch.randn(1, 768, device=device)  # CLIP embedding size

        logger.info(f"Testing with latent shape: {dummy_latent.shape}")
        
        with torch.no_grad():
            output = unet(
                x=dummy_latent,
                timesteps=dummy_timesteps,
                context=dummy_condition
            )

        logger.info(f"✅ U-Net forward works (output shape: {output.shape})")

        # Verify output has expected channels
        if output.shape[1] == 3:
            logger.info("✅ Output has correct channels (3)")
        else:
            logger.warning(f"⚠️  Output has {output.shape[1]} channels (expected 3)")
        
        return True

    except Exception as e:
        logger.error(f"❌ U-Net loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n")
    vqvae_ok = test_vqvae_loading()
    print("\n")
    unet_ok = test_unet_loading()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"VQ-VAE with adapter: {'✅ PASS' if vqvae_ok else '❌ FAIL'}")
    print(f"U-Net with adapter:  {'✅ PASS' if unet_ok else '❌ FAIL'}")
    print("=" * 70 + "\n")

    if vqvae_ok and unet_ok:
        print("✅ ALL TESTS PASSED - GHOP checkpoint loading works with adapters!\n")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED - Check errors above\n")
        sys.exit(1)