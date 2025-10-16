# File: scripts/validate_ghop_checkpoint.py
"""
Pre-training validation: Verify GHOP checkpoint integrity and loadability.
"""
import torch
import os
from pathlib import Path
from loguru import logger
import time

def validate_ghop_checkpoint_fast():
    """Fast checkpoint validation (~10 seconds)."""
    start_time = time.time()

    # ================================================================
    # STEP 1: Check file existence and size
    # ================================================================
    checkpoint_path = Path("../checkpoints/ghop/last.ckpt")
    config_path = Path("../checkpoints/ghop/config.yaml")

    logger.info("="*70)
    logger.info("GHOP Checkpoint Validation for Test Training")
    logger.info("="*70)

    # Check checkpoint exists
    if not checkpoint_path.exists():
        logger.error(f"❌ Checkpoint not found: {checkpoint_path}")

        # Provide helpful diagnostics
        logger.error("   Searching for checkpoint...")
        import subprocess
        result = subprocess.run(
            ["find", str(Path.cwd()), "-name", "*.ckpt", "-size", "+1000M"],
            capture_output=True, text=True
        )

        if result.stdout:
            logger.error("   Found large checkpoints:")
            for line in result.stdout.strip().split('\n'):
                logger.error(f"     {line}")
            logger.error("   Update checkpoint_path in scripts to match actual location")
        else:
            logger.error("   No large checkpoints found (>1GB)")
            logger.error("   Please download GHOP checkpoint first")

        return False

    # Verify it's a symlink or file
    if checkpoint_path.is_symlink():
        target = checkpoint_path.resolve()
        logger.info(f"✓ Checkpoint is symlink")
        logger.info(f"  Source: {checkpoint_path}")
        logger.info(f"  Target: {target}")

        # Check if target exists
        if not target.exists():
            logger.error(f"❌ Symlink target doesn't exist: {target}")
            return False

    # Check size
    size_mb = checkpoint_path.stat().st_size / (1024**2)
    if size_mb < 1000:  # Should be ~1100 MB
        logger.warning(f"⚠️  Checkpoint size unusual: {size_mb:.1f} MB (expected ~1100 MB)")
    else:
        logger.info(f"✓ Checkpoint size valid: {size_mb:.1f} MB")

    # Check config (optional)
    if config_path.exists():
        logger.info(f"✓ Config found: {config_path}")
    else:
        logger.warning(f"⚠️  Config not found: {config_path}")
        logger.warning("   Training may still work if checkpoint contains config")

    # ================================================================
    # STEP 2: Load checkpoint (verify PyTorch compatibility)
    # ================================================================
    logger.info("\nLoading checkpoint...")
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        logger.info("✓ Checkpoint loadable by PyTorch")
    except Exception as e:
        logger.error(f"❌ Failed to load checkpoint: {e}")
        return False

    # ================================================================
    # STEP 3: Validate structure
    # ================================================================
    logger.info("\nValidating checkpoint structure...")

    # Check top-level keys
    expected_top_keys = ['state_dict', 'epoch', 'global_step']
    for key in expected_top_keys:
        if key in ckpt:
            if key == 'state_dict':
                logger.info(f"  ✓ {key}: {len(ckpt[key])} parameters")
            else:
                logger.info(f"  ✓ {key}: {ckpt[key]}")
        else:
            logger.warning(f"  ⚠️  Missing key: {key}")

    # Check state_dict
    state_dict = ckpt.get('state_dict', {})
    if not state_dict:
        logger.error("❌ Empty state_dict")
        return False

    # Count VQ-VAE parameters
    vqvae_keys = [k for k in state_dict.keys() if any(
        x in k.lower() for x in ['encoder', 'decoder', 'quant', 'codebook']
    )]
    logger.info(f"\n  VQ-VAE Components:")
    logger.info(f"    ✓ Found {len(vqvae_keys)} VQ-VAE parameters")

    # Show sample VQ-VAE keys
    if vqvae_keys:
        logger.info(f"    Sample keys:")
        for key in vqvae_keys[:3]:
            if key in state_dict and isinstance(state_dict[key], torch.Tensor):
                logger.info(f"      - {key}: {state_dict[key].shape}")

    # Count U-Net parameters
    unet_keys = [k for k in state_dict.keys() if any(
        x in k.lower() for x in ['unet', 'model', 'time_embed', 'input_blocks', 'output_blocks']
    )]
    logger.info(f"\n  U-Net Components:")
    logger.info(f"    ✓ Found {len(unet_keys)} U-Net parameters")

    # Show sample U-Net keys
    if unet_keys:
        logger.info(f"    Sample keys:")
        for key in unet_keys[:3]:
            if key in state_dict and isinstance(state_dict[key], torch.Tensor):
                logger.info(f"      - {key}: {state_dict[key].shape}")

    # ================================================================
    # STEP 4: Verify parameter counts
    # ================================================================
    total_params = sum(
        p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor)
    )
    logger.info(f"\n  Total parameters: {total_params:,}")

    if total_params < 1_000_000:
        logger.warning(f"⚠️  Parameter count seems low: {total_params:,}")

    # ================================================================
    # STEP 5: Memory test (ensure loadable on GPU)
    # ================================================================
    if torch.cuda.is_available():
        logger.info(f"\nGPU Memory Test...")
        gpu_mem_free = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"  GPU memory available: {gpu_mem_free:.2f} GB")

        required_gb = (size_mb / 1024) * 1.5
        logger.info(f"  Estimated GPU requirement: {required_gb:.2f} GB")

        if required_gb > gpu_mem_free * 0.8:
            logger.warning(f"⚠️  GPU memory may be tight")
        else:
            logger.info(f"  ✓ Sufficient GPU memory")

    # ================================================================
    # FINAL REPORT
    # ================================================================
    elapsed = time.time() - start_time

    logger.info("\n" + "=" * 70)
    logger.info("✅ CHECKPOINT VALIDATION PASSED")
    logger.info("=" * 70)
    logger.info(f"  Checkpoint: {checkpoint_path}")
    logger.info(f"  Size: {size_mb:.1f} MB")
    logger.info(f"  VQ-VAE params: {len(vqvae_keys)}")
    logger.info(f"  U-Net params: {len(unet_keys)}")
    logger.info(f"  Total params: {total_params:,}")
    logger.info(f"  Validation time: {elapsed:.2f}s")
    logger.info("=" * 70)
    logger.info("✓ Ready for test training")
    logger.info("")

    return True

if __name__ == "__main__":
    import sys
    success = validate_ghop_checkpoint_fast()
    sys.exit(0 if success else 1)