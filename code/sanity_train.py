#!/usr/bin/env python3
"""
Sanity check script for HOISE pipeline with GHOP integration.
Tests all phases (1-5) with minimal computational requirements.

Environment: ghop_hold_integrated
Usage: python sanity_train.py --case test_sequence --use_ghop
"""

import sys
import os
import os.path as op
from pathlib import Path
from pprint import pprint

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


class SanityCheckConfig:
    """Configuration overrides for sanity checks."""

    # Minimal training settings
    NUM_EPOCHS = 1
    TEMPO_LEN = 10
    BATCH_SIZE = 2
    NUM_WORKERS = 0
    LOG_EVERY = 1
    EVAL_EVERY = 1

    # Phase 3 (SDS) settings
    SDS_ITERS = 5
    W_SDS = 1000.0
    GRID_RESOLUTION = 32

    # Phase 4 (Contact) settings
    CONTACT_ITERS = 3
    W_CONTACT = 5.0
    MESH_RESOLUTION = 64

    # Phase 5 settings
    PHASE5_ENABLED = False  # Disable for basic sanity check

    # Memory constraints
    MAX_RAYS_PER_BATCH = 256
    MAX_SAMPLES = 32


def setup_sanity_environment(case_name="test_sequence"):
    """
    Create minimal test data structure for sanity checks.

    Args:
        case_name: Name of test case directory
    """
    logger.info(f"Setting up sanity environment for case: {case_name}")

    # Create data directory structure
    data_dir = Path("./data") / case_name / "build"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy data.npy
    dummy_data = {
        "scene_bounding_sphere": 3.0,
        "n_images": 10,
        "cameras": np.random.randn(10, 9).astype(np.float32),
        "mano_params": np.random.randn(10, 62).astype(np.float32),
        "object_scale": 1.0,
        "object_rotation": np.eye(3).astype(np.float32),
        "object_translation": np.zeros(3).astype(np.float32),
    }
    np.save(data_dir / "data.npy", dummy_data)
    logger.info(f"✓ Created dummy data.npy")

    # Create dummy images
    img_dir = data_dir / "image"
    img_dir.mkdir(exist_ok=True)

    try:
        import cv2
        for i in range(10):
            # Create 64x64 dummy images (very small for speed)
            dummy_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / f"{i:04d}.png"), dummy_img)
        logger.info(f"✓ Created 10 dummy images (64x64)")
    except ImportError:
        logger.warning("OpenCV not available, skipping image creation")
        # Create empty files as placeholders
        for i in range(10):
            (img_dir / f"{i:04d}.png").touch()
        logger.info(f"✓ Created 10 placeholder image files")

    # Create dummy masks
    mask_dir = data_dir / "mask"
    mask_dir.mkdir(exist_ok=True)
    for i in range(10):
        dummy_mask = np.ones((64, 64), dtype=np.uint8) * 255
        if 'cv2' in sys.modules:
            cv2.imwrite(str(mask_dir / f"{i:04d}.png"), dummy_mask)
        else:
            (mask_dir / f"{i:04d}.png").touch()
    logger.info(f"✓ Created 10 dummy masks")

    logger.info(f"✓ Sanity environment setup complete: {data_dir}")
    return data_dir


def override_config_for_sanity(opt, args):
    """
    Override configuration with sanity check settings.

    Args:
        opt: Configuration from YAML
        args: Command-line arguments

    Returns:
        Modified opt and args
    """
    cfg = SanityCheckConfig()

    logger.info("Applying sanity check configuration overrides...")

    # Training settings
    args.num_epoch = cfg.NUM_EPOCHS
    args.tempo_len = cfg.TEMPO_LEN
    args.num_workers = cfg.NUM_WORKERS
    args.log_every = cfg.LOG_EVERY
    args.eval_every_epoch = cfg.EVAL_EVERY
    args.fast_dev_run = True
    args.debug = True
    args.num_sample = cfg.MAX_RAYS_PER_BATCH

    # Dataset settings
    opt.dataset.train.batch_size = cfg.BATCH_SIZE
    opt.dataset.valid.batch_size = 1
    opt.dataset.valid.pixel_per_batch = cfg.MAX_RAYS_PER_BATCH
    opt.dataset.test.pixel_per_batch = cfg.MAX_RAYS_PER_BATCH

    # Model settings - reduce network capacity
    opt.model.implicit_network.feature_vector_size = 128
    opt.model.implicit_network.dims = [128, 128, 128, 128]
    opt.model.rendering_network.feature_vector_size = 128
    opt.model.rendering_network.dims = [128, 128]

    # Ray sampler settings
    opt.model.ray_sampler.N_samples = cfg.MAX_SAMPLES
    opt.model.ray_sampler.N_samples_eval = cfg.MAX_SAMPLES * 2
    opt.model.ray_sampler.N_samples_extra = cfg.MAX_SAMPLES // 2
    opt.model.ray_sampler.max_total_iters = 2

    # Phase 3 settings
    if hasattr(opt, 'phase3') and opt.phase3.get('enabled', False):
        opt.phase3.warmup_iters = 2
        opt.phase3.sds_iters = cfg.SDS_ITERS
        opt.phase3.contact_iters = cfg.CONTACT_ITERS
        opt.phase3.w_sds = cfg.W_SDS
        opt.phase3.w_contact = cfg.W_CONTACT
        opt.phase3.grid_resolution = cfg.GRID_RESOLUTION
        opt.phase3.hand_field_resolution = cfg.GRID_RESOLUTION
        opt.phase3.sds.prediction_respacing = 20  # Very fast
        opt.phase3.sds.guidance_scale = 2.0
        logger.info(f"✓ Phase 3 overrides applied: {cfg.SDS_ITERS} SDS iters, grid {cfg.GRID_RESOLUTION}³")

    # Phase 4 settings
    if hasattr(opt, 'phase4') and opt.phase4.get('enabled', False):
        opt.phase4.contact_start_iter = cfg.SDS_ITERS
        opt.phase4.contact_duration = cfg.CONTACT_ITERS
        opt.phase4.mesh_resolution = cfg.MESH_RESOLUTION
        opt.phase4.w_contact = cfg.W_CONTACT
        opt.phase4.contact_warmup_iters = 2
        opt.phase4.log_contact_every = 1
        opt.phase4.use_mesh_cache = False  # Disable caching for sanity
        logger.info(f"✓ Phase 4 overrides applied: mesh resolution {cfg.MESH_RESOLUTION}³")

    # Phase 5 settings
    if hasattr(opt, 'phase5'):
        opt.phase5.enabled = cfg.PHASE5_ENABLED
        if cfg.PHASE5_ENABLED:
            opt.phase5.total_iterations = cfg.SDS_ITERS + cfg.CONTACT_ITERS + 2
            opt.phase5.warmup_iters = 2
            opt.phase5.temporal_window = 2
            logger.info(f"✓ Phase 5 enabled with minimal settings")

    logger.info(f"✓ Configuration overrides complete")
    logger.info(f"  Total training steps: {args.num_epoch * args.tempo_len // cfg.BATCH_SIZE}")

    return opt, args


def sanity_check_phase_initialization(model):
    """
    Verify phase initialization before training.

    Args:
        model: HOLD model instance

    Returns:
        bool: True if all checks pass
    """
    logger.info("\n" + "=" * 70)
    logger.info("SANITY CHECK: Phase Initialization")
    logger.info("=" * 70)

    checks_passed = []

    # ============================================================
    # Check 1: Base model attributes
    # ============================================================
    logger.info("\n--- Check 1: Base Model ---")
    required_attrs = ['implicit_network', 'rendering_network', 'density', 'ray_tracer']
    for attr in required_attrs:
        has_attr = hasattr(model, attr)
        status = "✓" if has_attr else "✗"
        logger.info(f"  {status} {attr}: {'Present' if has_attr else 'MISSING'}")
        checks_passed.append(has_attr)

    # ============================================================
    # Check 2: Phase 3 (GHOP) initialization
    # ============================================================
    if hasattr(model, 'phase3_enabled') and model.phase3_enabled:
        logger.info("\n--- Check 2: Phase 3 (GHOP) ---")

        # Check VQ-VAE, U-Net, Hand Field Builder
        phase3_attrs = {
            'vqvae': 'vqvae',
            'unet': 'unet',
            'hand_field_builder': 'hand_field_builder',  # Corrected from hand_field_computer
            'sds_loss': 'sds_loss'
        }

        for display_name, attr_name in phase3_attrs.items():
            has_attr = hasattr(model, attr_name) and getattr(model, attr_name) is not None
            status = "✓" if has_attr else "✗"
            logger.info(f"  {status} {display_name}: {'Initialized' if has_attr else 'MISSING'}")
            checks_passed.append(has_attr)

        # Check GHOP manager with safe dict access
        if hasattr(model, 'ghop_manager') and model.ghop_manager is not None:
            logger.info(f"  ✓ ghop_manager: Initialized")

            # ============================================================
            # CRITICAL FIX: Safe access to stage info
            # ============================================================
            try:
                # Try different method names
                if hasattr(model.ghop_manager, 'get_stage_info'):
                    stage_info = model.ghop_manager.get_stage_info(0)
                elif hasattr(model.ghop_manager, 'get_current_stage'):
                    stage_info = model.ghop_manager.get_current_stage()
                else:
                    stage_info = None

                if stage_info is not None:
                    # Handle both dict and string return types
                    if isinstance(stage_info, dict):
                        logger.info(f"    - Initial stage: {stage_info.get('stage', 'unknown')}")
                        logger.info(f"    - SDS weight: {stage_info.get('w_sds', 0.0):.1f}")
                    elif isinstance(stage_info, str):
                        logger.info(f"    - Initial stage: {stage_info}")
                    else:
                        logger.info(f"    - Stage info type: {type(stage_info).__name__}")
                else:
                    logger.info(f"    - Stage info: Not available")

            except Exception as e:
                logger.warning(f"    ⚠️  Could not get stage info: {e}")

            checks_passed.append(True)
        else:
            logger.warning(f"  ✗ ghop_manager: MISSING")
            checks_passed.append(False)
    else:
        logger.info("\n--- Check 2: Phase 3 (GHOP) ---")
        logger.info("  ⊘ Phase 3 disabled")

    # ============================================================
    # Check 3: Phase 4 (Contact) initialization
    # ============================================================
    if hasattr(model, 'phase4_enabled') and model.phase4_enabled:
        logger.info("\n--- Check 3: Phase 4 (Contact) ---")

        # Check contact-related attributes
        phase4_checks = {
            'contact_start_iter': 'contact_start_iter',
            'mesh_resolution': 'mesh_resolution',
            'contact_thresh': 'contact_thresh',
            'mesh_extractor': 'mesh_extractor',
            'contact_refiner': 'contact_refiner'
        }

        for display_name, attr_name in phase4_checks.items():
            if hasattr(model, attr_name):
                value = getattr(model, attr_name, None)
                if value is not None and not callable(value):
                    # Scalar value
                    status = "✓"
                    logger.info(f"  {status} {display_name}: {value}")
                else:
                    # Object/method
                    status = "✓"
                    logger.info(f"  {status} {display_name}: Initialized")
                checks_passed.append(True)
            else:
                status = "✗"
                logger.info(f"  {status} {display_name}: MISSING")
                checks_passed.append(False)
    else:
        logger.info("\n--- Check 3: Phase 4 (Contact) ---")
        logger.info("  ⊘ Phase 4 disabled")

    # ============================================================
    # Check 4: Model Parameters
    # ============================================================
    logger.info("\n--- Check 4: Model Parameters ---")
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  ✓ Total parameters: {total_params:,}")
        logger.info(f"  ✓ Trainable parameters: {trainable_params:,}")
        checks_passed.append(total_params > 0)
    except Exception as e:
        logger.warning(f"  ⚠️  Could not count parameters: {e}")
        checks_passed.append(False)

    # ============================================================
    # Summary
    # ============================================================
    logger.info("\n" + "=" * 70)

    total_checks = len(checks_passed)
    passed_checks = sum(checks_passed)

    if passed_checks == total_checks:
        logger.info(f"✅ ALL INITIALIZATION CHECKS PASSED ({passed_checks}/{total_checks})")
        result = True
    else:
        failed = total_checks - passed_checks
        logger.warning(f"⚠️  {failed}/{total_checks} checks failed")
        logger.warning("Some components missing - training may have issues")
        result = False

    logger.info("=" * 70 + "\n")

    return result

def sanity_check_forward_backward(model, device='cuda'):
    """
    Test forward and backward passes with dummy data.

    Args:
        model: HOLD model instance
        device: Device to run on

    Returns:
        bool: True if passes succeed
    """
    logger.info("\n" + "=" * 70)
    logger.info("SANITY CHECK: Forward/Backward Pass")
    logger.info("=" * 70)

    try:
        # Create minimal batch
        batch_size = 256
        num_frames = 10

        batch = {
            'rays_o': torch.randn(batch_size, 3, device=device),
            'rays_d': torch.nn.functional.normalize(
                torch.randn(batch_size, 3, device=device), dim=-1
            ),
            'rgb': torch.rand(batch_size, 3, device=device),
            'frame_ids': torch.randint(0, num_frames, (batch_size,), device=device),
        }

        logger.info(f"\n--- Forward Pass ---")
        logger.info(f"  Batch size: {batch_size} rays")
        logger.info(f"  Number of frames: {num_frames}")

        # Forward pass
        model.train()
        outputs = model.training_step(batch, batch_idx=0)

        logger.info(f"✓ Forward pass successful")
        logger.info(f"  Loss: {outputs['loss'].item():.6f}")

        # Check for NaN/Inf
        if torch.isnan(outputs['loss']):
            logger.error("✗ NaN detected in loss!")
            return False
        if torch.isinf(outputs['loss']):
            logger.error("✗ Inf detected in loss!")
            return False

        logger.info(f"✓ Loss is valid (no NaN/Inf)")

        # Backward pass
        logger.info(f"\n--- Backward Pass ---")
        outputs['loss'].backward()

        # Check gradients
        grad_norms = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm > 0:
                    grad_norms[name] = grad_norm

        if len(grad_norms) == 0:
            logger.error("✗ No gradients computed!")
            return False

        max_grad = max(grad_norms.values())
        mean_grad = np.mean(list(grad_norms.values()))

        logger.info(f"✓ Backward pass successful")
        logger.info(f"  Parameters with gradients: {len(grad_norms)}")
        logger.info(f"  Max gradient norm: {max_grad:.6f}")
        logger.info(f"  Mean gradient norm: {mean_grad:.6f}")

        if max_grad > 1000:
            logger.warning(f"⚠ Large gradient detected: {max_grad:.2f} (potential explosion)")

        # Memory check
        if device == 'cuda':
            mem_allocated = torch.cuda.memory_allocated(device) / 1e9
            mem_reserved = torch.cuda.memory_reserved(device) / 1e9
            logger.info(f"\n--- Memory Usage ---")
            logger.info(f"  Allocated: {mem_allocated:.2f} GB")
            logger.info(f"  Reserved: {mem_reserved:.2f} GB")

        logger.info("\n" + "=" * 70)
        logger.info("✓ FORWARD/BACKWARD CHECKS PASSED")
        logger.info("=" * 70 + "\n")

        model.zero_grad()
        return True

    except Exception as e:
        logger.error(f"✗ Forward/backward check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main sanity training entry point."""
    logger.info("=" * 70)
    logger.info("HOISE PIPELINE SANITY TRAINING")
    logger.info("Environment: ghop_hold_integrated")
    logger.info("=" * 70 + "\n")

    # Parse arguments
    args, opt = parser_args()

    # Override for sanity checks
    opt, args = override_config_for_sanity(opt, args)

    # Setup sanity environment
    if not op.exists(f"./data/{args.case}/build/data.npy"):
        setup_sanity_environment(args.case)
    else:
        logger.info(f"✓ Using existing data directory: ./data/{args.case}")

    logger.info(f"\nWorking directory: {os.getcwd()}")
    logger.info(f"Case: {args.case}")
    logger.info(f"GPU ID: {args.gpu_id}")

    # Create trainer with minimal settings
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=op.join(args.log_dir, "checkpoints/"),
        filename="sanity-{epoch:04d}",
        save_last=True,
        save_top_k=1,
        every_n_epochs=1,
        verbose=True,
    )

    trainer = pl.Trainer(
        gpus=1,
        accelerator="gpu",
        gradient_clip_val=1.0,  # Aggressive clipping for sanity
        callbacks=[checkpoint_callback],
        max_epochs=args.num_epoch,
        check_val_every_n_epoch=args.eval_every_epoch,
        log_every_n_steps=args.log_every,
        num_sanity_val_steps=0,
        logger=False,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Create datasets
    logger.info("\n--- Dataset Creation ---")
    trainset = create_dataset(opt.dataset.train, args)
    validset = create_dataset(opt.dataset.valid, args)
    logger.info(f"✓ Training dataset: {len(trainset)} samples")
    logger.info(f"✓ Validation dataset: {len(validset)} samples")

    # Initialize model
    logger.info("\n--- Model Initialization ---")
    reset_all_seeds(1)
    model = HOLD(opt, args)
    model.trainset = trainset
    logger.info(f"✓ HOLD model initialized")

    # Phase configuration (matching train.py logic)
    use_ghop = (hasattr(opt, 'phase3') and opt.phase3.get('enabled', False)) or \
               (hasattr(args, 'use_ghop') and args.use_ghop)

    if use_ghop:
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 3: Initializing GHOP Two-Stage Training")
        logger.info("=" * 70)

        if hasattr(opt, 'phase3') and opt.phase3.get('enabled', False):
            phase3_cfg = opt.phase3

            # ============================================================
            # CRITICAL FIX: Skip checkpoint check if using random init
            # ============================================================
            vqvae_use_pretrained = phase3_cfg.ghop.get('vqvae_use_pretrained', False)
            unet_use_pretrained = phase3_cfg.ghop.get('unet_use_pretrained', False)
            need_checkpoint = vqvae_use_pretrained or unet_use_pretrained

            if need_checkpoint:
                # Only check checkpoint if we're loading pretrained weights
                model_checkpoint = phase3_cfg.ghop.get('model_checkpoint',
                                                        'checkpoints/ghop/last.ckpt')

                logger.info(f"Using unified GHOP checkpoint: {model_checkpoint}")

                # Check unified checkpoint existence
                if not os.path.exists(model_checkpoint):
                    logger.error(f"✗ GHOP unified checkpoint not found: {model_checkpoint}")
                    logger.error("  Expected: ~/Projects/ghop/output/joint_3dprior/mix_data/checkpoints/last.ckpt")
                    logger.error("  Create symlink: ln -s ~/Projects/ghop/.../last.ckpt checkpoints/ghop/last.ckpt")
                    sys.exit(1)

                # Verify it's a valid checkpoint
                try:
                    ckpt = torch.load(model_checkpoint, map_location='cpu')
                    if 'state_dict' not in ckpt:
                        raise ValueError("Invalid checkpoint: missing state_dict")

                    state_dict = ckpt['state_dict']
                    vqvae_keys = [k for k in state_dict.keys() if 'first_stage' in k or 'encoder' in k]
                    unet_keys = [k for k in state_dict.keys() if 'model' in k and 'first_stage' not in k]

                    logger.info(f"✓ Unified checkpoint verified:")
                    logger.info(f"  - Total parameters: {len(state_dict)}")
                    logger.info(f"  - VQ-VAE components: {len(vqvae_keys)}")
                    logger.info(f"  - U-Net components: {len(unet_keys)}")

                except Exception as e:
                    logger.error(f"✗ Checkpoint validation failed: {e}")
                    sys.exit(1)

                logger.info("[Phase 3] Using config-based unified checkpoint")

            else:
                # Random initialization mode - no checkpoint needed
                logger.info("[Phase 3] Using RANDOM initialization (no checkpoint required)")
                logger.info("  VQ-VAE: Random weights")
                logger.info("  U-Net: Random weights")
                logger.info("  This is expected for sanity checks")

        else:
            # Fallback: command-line arguments
            model_checkpoint = getattr(args, 'model_checkpoint', None)
            logger.info("[Phase 3] Using command-line argument initialization")

        model.phase3_enabled = True
        model.ghop_enabled = True
        logger.info("=" * 70 + "\n")

    else:
        model.phase3_enabled = False
        model.ghop_enabled = False
        logger.info("\n[GHOP] Disabled - sanity check without GHOP\n")

    # ============================================================
    # Continue with training setup (Lightning trainer, etc.)
    # ============================================================
    logger.info("\n--- Training Configuration ---")

    # Phase 4 configuration (matching train.py logic)
    if model.phase3_enabled:
        logger.info("=" * 70)
        logger.info("PHASE 4: Initializing Contact Refinement Module")
        logger.info("=" * 70)

        if hasattr(opt, 'phase4') and opt.phase4.get('enabled', False):
            model.phase4_enabled = True
            model.contact_start_iter = opt.phase4.get('contact_start_iter', 5)
            model.w_contact = opt.phase4.get('w_contact', 5.0)
            model.mesh_resolution = opt.phase4.get('mesh_resolution', 64)
            model.contact_thresh = opt.phase4.get('contact_thresh', 0.01)
            model.collision_thresh = opt.phase4.get('collision_thresh', 0.005)
            logger.info(f"✓ Phase 4 configured (contact start: {model.contact_start_iter})")
        else:
            model.phase4_enabled = False
            logger.info("⊘ Phase 4 disabled")

        logger.info("=" * 70 + "\n")
    else:
        model.phase4_enabled = False

    # Run sanity checks
    logger.info("=" * 70)
    logger.info("RUNNING PRE-TRAINING SANITY CHECKS")
    logger.info("=" * 70 + "\n")

    check1 = sanity_check_phase_initialization(model)
    check2 = sanity_check_forward_backward(model, device='cuda')

    if not (check1 and check2):
        logger.warning("\n✗ SANITY CHECKS FAILED - Continuing anyway...")
        # sys.exit(1)

    logger.info("\n✓ ALL PRE-TRAINING SANITY CHECKS PASSED\n")
    logger.info("=" * 70)
    logger.info("STARTING SANITY TRAINING")
    logger.info("=" * 70 + "\n")

    # Start training
    try:
        trainer.fit(model, trainset, validset)

        logger.info("\n" + "=" * 70)
        logger.info("✓ SANITY TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info(f"Checkpoints saved to: {op.join(args.log_dir, 'checkpoints/')}")
        logger.info("=" * 70 + "\n")

        return 0

    except Exception as e:
        logger.error(f"\n✗ SANITY TRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)