#!/usr/bin/env python3
"""
FAST Sanity check script for HOISE pipeline.
Optimized for speed (~30 seconds vs 10 minutes).

Key optimizations:
- Validation completely disabled
- Minimal mesh resolution (32³)
- No mesh/image logging
- Fast dataloader (4 workers)

Usage:
    python sanity_train_fast.py --config confs/sanity_check_fast.yaml \\
                                --case hold_mug1_itw \\
                                --shape_init 75268d864 \\
                                --gpu_id 0
"""

import sys
import os
import os.path as op
from pathlib import Path

import torch
import numpy as np
import pytorch_lightning as pl
from loguru import logger
from easydict import EasyDict as edict

# Add project paths
sys.path.insert(0, op.join(op.dirname(__file__), ".."))

from src.hold.hold import HOLD
from src.datasets.utils import create_dataset
from src.utils.parser import parser_args
from common.torch_utils import reset_all_seeds


class FastSanityConfig:
    """Fast sanity check configuration (30 seconds target)."""

    # Training
    NUM_EPOCHS = 1
    TEMPO_LEN = 10
    BATCH_SIZE = 2
    NUM_WORKERS = 4  # Parallel data loading
    LOG_EVERY = 999999  # Disable logging

    # Phase 3
    SDS_ITERS = 5
    W_SDS = 100.0
    GRID_RESOLUTION = 32

    # Phase 4
    CONTACT_ITERS = 3
    W_CONTACT = 5.0
    MESH_RESOLUTION = 32  # Coarse mesh (was 64)

    # Ray sampling
    MAX_RAYS = 64  # Minimal
    MAX_SAMPLES = 16  # Minimal

    # Validation
    VALIDATION_ENABLED = False  # ← KEY: Disable validation


def override_config_for_fast_sanity(opt, args):
    """
    Apply FAST sanity check overrides.

    Speedup breakdown:
    - Skip validation: -9m 58s (99%)
    - Reduce mesh res: -20s
    - Disable logging: -10s

    Total: ~10 minutes → ~30 seconds (20x)
    """
    cfg = FastSanityConfig()

    logger.info("=" * 70)
    logger.info("FAST SANITY CHECK MODE")
    logger.info("=" * 70)
    logger.info("Target runtime: ~30 seconds (vs 10 minutes)")
    logger.info("Validation: DISABLED")
    logger.info("Logging: MINIMAL")
    logger.info("=" * 70)

    # Training settings
    args.num_epoch = cfg.NUM_EPOCHS
    args.tempo_len = cfg.TEMPO_LEN
    args.num_workers = cfg.NUM_WORKERS
    args.log_every = cfg.LOG_EVERY
    args.eval_every_epoch = 999999  # Never validate
    args.fast_dev_run = True
    args.debug = True
    args.num_sample = cfg.MAX_RAYS

    # ============================================================
    # CRITICAL: Disable validation (99% speedup)
    # ============================================================
    if hasattr(opt, 'validation'):
        opt.validation.enabled = False
        logger.info("✓ Validation DISABLED (saves ~10 minutes)")

    opt.dataset.valid.batch_size = 1
    opt.dataset.valid.pixel_per_batch = cfg.MAX_RAYS

    # Dataset settings
    opt.dataset.train.batch_size = cfg.BATCH_SIZE
    opt.dataset.train.num_workers = cfg.NUM_WORKERS
    opt.dataset.train.prefetch_factor = 2

    # Model settings - minimal networks
    opt.model.implicit_network.feature_vector_size = 64
    opt.model.implicit_network.dims = [64, 64, 64]
    opt.model.rendering_network.feature_vector_size = 64
    opt.model.rendering_network.dims = [64]

    # Ray sampler - minimal samples
    opt.model.ray_sampler.N_samples = cfg.MAX_SAMPLES
    opt.model.ray_sampler.N_samples_eval = cfg.MAX_SAMPLES * 2
    opt.model.ray_sampler.N_samples_extra = cfg.MAX_SAMPLES // 2
    opt.model.ray_sampler.max_total_iters = 1
    opt.model.ray_sampler.beta_iters = 3

    # Phase 3: Fast SDS
    if hasattr(opt, 'phase3') and opt.phase3.get('enabled', False):
        opt.phase3.sds.num_sds_iterations = cfg.SDS_ITERS
        opt.phase3.sds.grid_resolution = cfg.GRID_RESOLUTION
        opt.phase3.sds.prediction_respacing = 10  # Very fast
        opt.phase3.sds.weight = cfg.W_SDS
        logger.info(f"✓ Phase 3: {cfg.SDS_ITERS} SDS iters, {cfg.GRID_RESOLUTION}³ grid")

    # Phase 4: Fast contact
    if hasattr(opt, 'phase4') and opt.phase4.get('enabled', False):
        opt.phase4.contact_start_iter = cfg.SDS_ITERS
        opt.phase4.mesh_resolution = cfg.MESH_RESOLUTION  # Coarse
        opt.phase4.contact_warmup_iters = 0  # No warmup
        opt.phase4.log_contact_every = 999999  # No logging
        opt.phase4.visualize_contact = False
        opt.phase4.use_mesh_cache = False
        logger.info(f"✓ Phase 4: {cfg.MESH_RESOLUTION}³ mesh (fast)")

    # Phase 5: Disabled
    if hasattr(opt, 'phase5'):
        opt.phase5.enabled = False

    # Logging: Disabled
    if hasattr(opt, 'logging'):
        opt.logging.log_images_every = 999999
        opt.logging.log_mesh_every = 999999
        logger.info("✓ Image/mesh logging DISABLED")

    # Callbacks: Disabled
    if hasattr(opt, 'callbacks'):
        opt.callbacks.render_validation = False
        opt.callbacks.mesh_extraction = False
        opt.callbacks.mesh_cano = False
        logger.info("✓ Validation callbacks DISABLED")

    logger.info("=" * 70)
    logger.info(f"Expected steps: {cfg.SDS_ITERS + cfg.CONTACT_ITERS}")
    logger.info(f"Expected time: ~30 seconds")
    logger.info("=" * 70 + "\n")

    return opt, args


def main():
    """Fast sanity check main entry point."""
    logger.info("=" * 70)
    logger.info("FAST SANITY TRAINING MODE")
    logger.info("Environment: ghop_hold_integrated")
    logger.info("=" * 70 + "\n")

    # Parse arguments
    args, opt = parser_args()

    # Apply FAST overrides
    opt, args = override_config_for_fast_sanity(opt, args)

    # Check data directory
    if not op.exists(f"./data/{args.case}/build/data.npy"):
        logger.error(f"✗ Data directory not found: ./data/{args.case}")
        logger.error("  Run regular sanity_train.py first to create test data")
        sys.exit(1)

    logger.info(f"✓ Using data directory: ./data/{args.case}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Case: {args.case}")
    logger.info(f"GPU ID: {args.gpu_id}\n")

    # ============================================================
    # Minimal trainer configuration
    # ============================================================
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=op.join(args.log_dir, "checkpoints/"),
        filename="fast-sanity-{epoch:04d}",
        save_last=False,  # Don't save
        save_top_k=0,  # Don't save
        verbose=False,
    )

    trainer = pl.Trainer(
        gpus=1,
        accelerator="gpu",
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback],
        max_epochs=args.num_epoch,
        check_val_every_n_epoch=999999,  # Never validate
        log_every_n_steps=999999,  # Never log
        num_sanity_val_steps=0,  # Skip sanity validation
        logger=False,  # Disable logger
        enable_progress_bar=True,
        enable_model_summary=False,  # Skip summary
    )

    # Create datasets
    logger.info("--- Dataset Creation ---")
    trainset = create_dataset(opt.dataset.train, args)
    # Don't create validset since validation is disabled
    logger.info(f"✓ Training dataset: {len(trainset)} samples")
    logger.info(f"✓ Validation: DISABLED\n")

    # Initialize model
    logger.info("--- Model Initialization ---")
    reset_all_seeds(1)
    model = HOLD(opt, args)
    model.trainset = trainset
    logger.info(f"✓ HOLD model initialized\n")

    # Phase configuration
    use_ghop = (hasattr(opt, 'phase3') and opt.phase3.get('enabled', False)) or \
               (hasattr(args, 'use_ghop') and args.use_ghop)

    if use_ghop:
        logger.info("=" * 70)
        logger.info("PHASE 3: GHOP (Random Init)")
        logger.info("=" * 70)
        logger.info("  VQ-VAE: Random weights")
        logger.info("  U-Net: Random weights")
        logger.info("  (Pretrained weights skipped for speed)")
        model.phase3_enabled = True
        model.ghop_enabled = True
        logger.info("=" * 70 + "\n")
    else:
        model.phase3_enabled = False
        model.ghop_enabled = False

    # Phase 4 configuration
    if model.phase3_enabled and hasattr(opt, 'phase4') and opt.phase4.get('enabled', False):
        logger.info("=" * 70)
        logger.info("PHASE 4: Contact (Minimal)")
        logger.info("=" * 70)
        model.phase4_enabled = True
        model.contact_start_iter = opt.phase4.get('contact_start_iter', 5)
        model.mesh_resolution = opt.phase4.get('mesh_resolution', 32)
        logger.info(f"  Mesh resolution: {model.mesh_resolution}³")
        logger.info("=" * 70 + "\n")
    else:
        model.phase4_enabled = False

    # Start training (no pre-checks for speed)
    logger.info("=" * 70)
    logger.info("STARTING FAST SANITY TRAINING")
    logger.info("=" * 70 + "\n")

    import time
    start_time = time.time()

    try:
        # Train without validation
        trainer.fit(model, trainset)

        elapsed = time.time() - start_time

        logger.info("\n" + "=" * 70)
        logger.info("✅ FAST SANITY TRAINING COMPLETED")
        logger.info("=" * 70)
        logger.info(f"Runtime: {elapsed:.1f} seconds")
        logger.info(f"Target was: ~30 seconds")

        if elapsed < 60:
            logger.info(f"✓ FAST MODE SUCCESSFUL ({elapsed:.0f}s < 60s)")
        else:
            logger.warning(f"⚠️  Slower than expected ({elapsed:.0f}s)")

        logger.info("=" * 70 + "\n")

        return 0

    except Exception as e:
        logger.error(f"\n✗ FAST SANITY TRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)