from pprint import pprint
import os
import os.path as op
import sys
import numpy as np
import pytorch_lightning as pl
from loguru import logger

from src.hold.hold import HOLD
from src.datasets.utils import create_dataset
from src.utils.parser import parser_args
from common.torch_utils import reset_all_seeds
import torch
from src.hold.loss import HOLDLoss
from src.datasets.ghop_hoi_dataset import GHOPHOIDataset  # NEW IMPORT

def validate_phase4_config(opt, model):
    """
    Validate Phase 4 GHOP configuration and dependencies.

    Args:
        opt: Configuration object with phase4 section
        model: HOLD model instance with Phase 4 attributes set

    Returns:
        bool: True if validation passes, False otherwise
    """
    # Check if Phase 4 is enabled
    if not hasattr(model, 'phase4_enabled') or not model.phase4_enabled:
        return True

    try:
        # Rule 1: Phase 4 requires Phase 3
        if not model.phase3_enabled:
            logger.error(
                "[Phase 4 Error] Phase 4 requires Phase 3 to be enabled.\n"
                "Please set phase3.enabled=true in your configuration."
            )
            return False

        # Rule 2: Phase 4 requires modular Phase 3 initialization (VQ-VAE accessible)
        if not hasattr(model, 'vqvae') or model.vqvae is None:
            logger.error(
                "[Phase 4 Error] Phase 4 requires modular Phase 3 initialization.\n"
                "Please set phase3.use_modular_init=true in your configuration."
            )
            return False

        # Rule 3: Contact start iteration must be after SDS stage
        if hasattr(opt, 'phase3'):
            sds_iters = opt.phase3.get('sds_iters', 500)
            contact_start = model.contact_start_iter

            if contact_start < sds_iters:
                logger.warning(
                    f"[Phase 4 Warning] contact_start_iter ({contact_start}) < "
                    f"phase3.sds_iters ({sds_iters}).\n"
                    f"Contact refinement will overlap with SDS stage."
                )

        # Rule 4: Validate mesh resolution
        mesh_resolution = model.mesh_resolution
        if mesh_resolution < 64 or mesh_resolution > 256:
            logger.warning(
                f"[Phase 4 Warning] Unusual mesh_resolution: {mesh_resolution}.\n"
                f"Recommended range: 64-256 (128 for production)."
            )

        # Rule 5: Validate thresholds
        contact_thresh = model.contact_thresh
        collision_thresh = model.collision_thresh

        if collision_thresh >= contact_thresh:
            logger.error(
                f"[Phase 4 Error] collision_thresh ({collision_thresh}) must be < "
                f"contact_thresh ({contact_thresh})."
            )
            return False

        logger.info(
            f"[Phase 4] Configuration validated:\n"
            f"  - Contact start iteration: {contact_start}\n"
            f"  - Contact loss weight: {model.w_contact}\n"
            f"  - Mesh resolution: {mesh_resolution}³\n"
            f"  - Contact threshold: {contact_thresh}m\n"
            f"  - Collision threshold: {collision_thresh}m"
        )

        return True

    except Exception as e:
        logger.error(f"[Phase 4] Validation failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_dataset_with_ghop_support(dataset_config, args):
    """
    Create dataset with GHOP HOI4D/HO3D support.

    Dataset selection logic:
    - If --use_ghop flag: Load from data/{case}/ghop_data/
    - If no flag: Load from data/{case}/build/

    This allows the same case to support both HOLD and GHOP modes:
    - hold_bottle1_itw (no flag) → HOLD single-image
    - hold_bottle1_itw --use_ghop → GHOP video (if ghop_data exists)

    Args:
        dataset_config: Dataset configuration from YAML
        args: Command-line arguments

    Returns:
        Dataset instance (ImageDataset, TempoDataset, or GHOPHOIDataset)
    """
    from src.datasets.utils import create_dataset as create_image_dataset

    # ================================================================
    # STEP 1: Check dataset selection via command-line flag
    # ================================================================
    dataset_type = dataset_config.get('type', 'train')

    # ✅ PRIMARY: Use --use_ghop flag to determine dataset type
    use_ghop_flag = hasattr(args, 'use_ghop') and args.use_ghop

    # Also check config for explicit override (optional)
    explicit_ghop_config = (
        dataset_config.get('dataset_type', '') == 'ghop_hoi' or
        dataset_config.get('dataset_type', '') == 'ghop_ho3d' or
        dataset_config.get('is_video', False)
    )

    # Final decision: flag takes precedence, then config
    use_ghop = use_ghop_flag or explicit_ghop_config

    logger.info(f"[Dataset Selection] --use_ghop flag: {use_ghop_flag}")
    logger.info(f"[Dataset Selection] Config override: {explicit_ghop_config}")
    logger.info(f"[Dataset Selection] Final decision: {'GHOP' if use_ghop else 'HOLD'}")

    # ================================================================
    # STEP 2: Determine data path based on dataset type
    # ================================================================
    case_path = os.path.join("./data", args.case)

    if use_ghop:
        # ✅ GHOP: Use ghop_data symlink
        data_path = os.path.join(case_path, "ghop_data")

        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"GHOP dataset requested (--use_ghop) but ghop_data not found:\n"
                f"  Expected: {data_path}\n"
                f"  Case: {args.case}\n\n"
                f"To use GHOP dataset:\n"
                f"  1. Create symlink: ln -s /path/to/HOI4D_clip/Object_1 {data_path}\n"
                f"  2. Or remove --use_ghop flag to use HOLD dataset\n"
            )

        # Resolve symlink to actual location
        if os.path.islink(data_path):
            data_path = os.path.realpath(data_path)
            logger.info(f"[Dataset Path] Symlink resolved: {data_path}")
        else:
            logger.info(f"[Dataset Path] Direct path: {data_path}")

    else:
        # ✅ HOLD: Use build directory
        data_path = os.path.join(case_path, "build")

        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"HOLD dataset requested (no --use_ghop) but build/ not found:\n"
                f"  Expected: {data_path}\n"
                f"  Case: {args.case}\n\n"
                f"To use HOLD dataset:\n"
                f"  1. Ensure data/{args.case}/build/ exists with processed data\n"
                f"  2. Or add --use_ghop flag to use GHOP dataset (if available)\n"
            )

        logger.info(f"[Dataset Path] HOLD build: {data_path}")

    # ================================================================
    # STEP 3: Create GHOP dataset (HOI4D/HO3D video sequences)
    # ================================================================
    if use_ghop:
        logger.info("=" * 70)
        logger.info("CREATING GHOP VIDEO DATASET (HOI4D/HO3D)")
        logger.info("=" * 70)

        # Detect source dataset from path
        if 'HOI4D' in data_path or 'hoi4d' in data_path.lower():
            source_dataset = 'HOI4D'
        elif 'HO3D' in data_path or 'ho3d' in data_path.lower():
            source_dataset = 'HO3D'
        else:
            source_dataset = 'Unknown'

        logger.info(f"GHOP Configuration:")
        logger.info(f"  Case name: {args.case}")
        logger.info(f"  Source dataset: {source_dataset}")
        logger.info(f"  Data directory: {data_path}")
        logger.info(f"  Split: {dataset_type}")

        # Verify required files
        required_files = ['cameras_hoi.npz', 'hands.npz']
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(data_path, f))]

        if missing_files:
            raise FileNotFoundError(
                f"GHOP dataset incomplete: {data_path}\n"
                f"Missing files: {missing_files}\n"
                f"Expected structure:\n"
                f"  - cameras_hoi.npz (camera poses and intrinsics)\n"
                f"  - hands.npz (hand articulation parameters)\n"
                f"Please verify dataset extraction."
            )

        # Create GHOP dataset
        dataset = GHOPHOIDataset(
            data_dir=data_path,
            split=dataset_type,  # 'train' or 'val'
            args=args
        )

        logger.info(f"✓ GHOP dataset created:")
        logger.info(f"  Total frames: {dataset.n_frames}")
        logger.info(f"  Frame pairs: {len(dataset)}")
        logger.info(f"  Category: {dataset.category}")
        logger.info(f"  Source: {source_dataset}")

        # Verify temporal fields
        sample = dataset[0]
        if 'hA_n' in sample and 'c2w_n' in sample:
            logger.info(f"  ✅ Temporal fields present (hA_n, c2w_n)")
            logger.info(f"  ✅ Phase 5 temporal consistency will ACTIVATE")
        else:
            logger.warning(f"  ⚠️  Temporal fields missing!")
            logger.warning(f"  ⚠️  Phase 5 will SKIP")

        logger.info("=" * 70)
        return dataset

    # ================================================================
    # STEP 4: Create HOLD dataset (single-image reconstruction)
    # ================================================================
    else:
        logger.info("=" * 70)
        logger.info("CREATING HOLD SINGLE-IMAGE DATASET")
        logger.info("=" * 70)

        logger.info(f"  Case: {args.case}")
        logger.info(f"  Build path: {data_path}")

        # Determine dataset class based on split type
        if dataset_type == "train":
            from src.datasets.tempo_dataset import TempoDataset
            dataset = TempoDataset(args)
            logger.info(f"  Dataset class: TempoDataset (wraps ImageDataset)")
        elif dataset_type == "val":
            from src.datasets.eval_datasets import ValDataset
            dataset = ValDataset(args)
            logger.info(f"  Dataset class: ValDataset")
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        logger.info(f"✓ HOLD dataset created:")
        logger.info(f"  Total samples: {len(dataset)}")

        # Verify RGB ground truth
        sample = dataset[0]
        has_rgb = 'gt.rgb' in sample

        if has_rgb:
            logger.info(f"  ✅ Has gt.rgb: {sample['gt.rgb'].shape}")
        else:
            logger.warning(f"  ⚠️  Missing gt.rgb - RGB loss will fail!")

        # Check for text metadata (ImageDataset feature)
        if hasattr(dataset, 'text_metadata'):
            logger.info(f"  ✅ Text metadata: {dataset.category}")
            prompt = dataset.get_text_prompt('detailed')
            logger.info(f"     Detailed prompt: {prompt[:60]}...")
        elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'text_metadata'):
            # TempoDataset wraps ImageDataset
            logger.info(f"  ✅ Text metadata: {dataset.dataset.category}")
            prompt = dataset.dataset.get_text_prompt('detailed')
            logger.info(f"     Detailed prompt: {prompt[:60]}...")
        else:
            logger.warning(f"  ⚠️  No text metadata found")

        logger.info(f"  ⚠️  Phase 5 temporal will SKIP (single images, no temporal pairs)")
        logger.info("=" * 70)

        return dataset

def main():
    # ================================================================
    # CRITICAL: Disable Comet BEFORE parser_args() is called
    # ================================================================
    # Check for --no-comet flag in sys.argv BEFORE parsing
    if '--no-comet' in sys.argv:
        print("\n" + "=" * 70)
        print("⚠️  COMET LOGGING DISABLED")
        print("=" * 70)
        print("Running in debug mode without Comet ML metric uploads.")
        print("Training will be MUCH faster but metrics won't be logged.")
        print("To re-enable: Remove --no-comet flag")
        print("=" * 70 + "\n")

        # Set environment variable BEFORE any comet_ml imports
        os.environ['COMET_MODE'] = 'disabled'

    # Now parse arguments (comet_ml will see COMET_MODE=disabled)
    args, opt = parser_args()
    print("Working dir:", os.getcwd())

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=op.join(args.log_dir, "checkpoints/"),
        filename="{epoch:04d}-{loss}",
        save_last=True,
        save_top_k=-1,
        every_n_epochs=args.eval_every_epoch,
        verbose=True,
    )

    # ================================================================
    # ✅ CRITICAL: Mixed Precision Trainer with Memory Optimization
    #
    # Issue: Memory fragmentation causes OOM at epoch 20
    # Root cause: FP32 activations + gradients consume too much memory
    #
    # Fix: FP16 mixed precision training
    #   - Reduces activation memory by 50%
    #   - Reduces gradient memory by 50%
    #   - Less memory pressure → less fragmentation
    #   - Native PyTorch AMP with automatic loss scaling
    #
    # Expected result: Training completes past epoch 20
    # ================================================================

    trainer = pl.Trainer(
        gpus=1,
        accelerator="gpu",
        callbacks=[checkpoint_callback],
        max_epochs=args.num_epoch,
        check_val_every_n_epoch=args.eval_every_epoch,
        log_every_n_steps=args.log_every,
        num_sanity_val_steps=0,
        limit_val_batches=0,  # Disable validation (GHOP dataset incompatible)
        enable_checkpointing=True,  # Keep checkpointing enabled
        enable_progress_bar=False,  # ✅ CRITICAL: Disable to prevent metric accumulation memory leak
        logger=False,
        # precision=16,  # ✅ CRITICAL: Use FP16 mixed precision (50% memory reduction)
        # amp_backend='native',  # Use PyTorch native AMP
    )

    # ================================================================
    # Additional Comet suppression for --no-comet mode
    # ================================================================
    if args.no_comet:
        # Suppress any remaining Comet warnings
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='comet_ml')

    pprint(args)

    trainset = create_dataset_with_ghop_support(opt.dataset.train, args)
    validset = create_dataset_with_ghop_support(opt.dataset.valid, args)

    # ========================================================================
    # Model Initialization
    # ========================================================================
    model = HOLD(opt, args)
    model.trainset = trainset

    # ========================================================================
    # Phase 3: GHOP Two-Stage Training Integration
    # ========================================================================
    use_ghop = (hasattr(opt, 'phase3') and opt.phase3.get('enabled', False)) or \
               (hasattr(args, 'use_ghop') and args.use_ghop)

    if use_ghop:
        print("\n" + "="*70)
        print("PHASE 3: Initializing GHOP Two-Stage Training")
        print("="*70)

        # Determine if using config or command-line arguments
        if hasattr(opt, 'phase3') and opt.phase3.get('enabled', False):
            phase3_cfg = opt.phase3
            vqvae_ckpt = phase3_cfg.ghop.vqvae_checkpoint
            unet_ckpt = phase3_cfg.ghop.unet_checkpoint
            sds_iters = phase3_cfg.get('sds_iters', 500)
            contact_iters = phase3_cfg.get('contact_iters', 100)
            grid_resolution = phase3_cfg.get('grid_resolution', 64)
            w_sds = phase3_cfg.get('w_sds', 5000.0)
            w_contact = phase3_cfg.get('w_contact', 10.0)
            use_modular = phase3_cfg.get('use_modular_init', False)
            print("[Phase 3] Using config-based initialization")
        else:
            vqvae_ckpt = args.vqvae_ckpt
            unet_ckpt = args.unet_ckpt
            sds_iters = args.sds_iters
            contact_iters = args.contact_iters
            grid_resolution = getattr(args, 'grid_resolution', 64)
            w_sds = getattr(args, 'w_sds', 5000.0)
            w_contact = getattr(args, 'w_contact', 10.0)
            use_modular = True
            print("[Phase 3] Using command-line argument initialization")

        # Verify GHOP checkpoint paths
        if not os.path.exists(vqvae_ckpt):
            raise FileNotFoundError(
                f"[Phase 3 Error] VQ-VAE checkpoint not found: {vqvae_ckpt}\n"
                f"Please download GHOP checkpoints and place in checkpoints/ghop/ directory"
            )

        if not os.path.exists(unet_ckpt):
            raise FileNotFoundError(
                f"[Phase 3 Error] U-Net checkpoint not found: {unet_ckpt}\n"
                f"Please download GHOP checkpoints and place in checkpoints/ghop/ directory"
            )

        print(f"✓ VQ-VAE checkpoint verified: {vqvae_ckpt}")
        print(f"✓ U-Net checkpoint verified: {unet_ckpt}")

        # Set flags for training_step (components initialized in HOLD.__init__)
        model.phase3_enabled = True
        model.ghop_enabled = True

        print(f"\n✓ Phase 3 two-stage training initialized:")
        print(f"  - Stage 1 (SDS): {sds_iters} iterations")
        print(f"  - Stage 2 (Contact): {contact_iters} iterations")
        print(f"  - SDS loss weight: {w_sds}")
        print(f"  - Contact loss weight: {w_contact}")
        print(f"  - Grid resolution: {grid_resolution}³ voxels")
        print(f"  - Modular init: {use_modular}")

        if hasattr(model, 'ghop_manager'):
            print(f"  - Two-Stage Manager: Initialized")
            stage_info = model.ghop_manager.get_stage_info(0)
            print(f"  - Initial stage: {stage_info}")

        print("="*70 + "\n")

    elif hasattr(opt, 'phase2') and opt.phase2.get('enabled', False):
        # Backward compatibility: Phase 2 legacy mode
        print("\n" + "="*70)
        print("PHASE 2: Initializing GHOP Prior (Legacy Mode)")
        print("="*70 + "\n")
        print("[Warning] Phase 2 detected. Consider upgrading to Phase 3 config.")
        print("          Phase 2 will be supported but Phase 3 offers better features.\n")

        vqvae_ckpt = opt.phase2.ghop.vqvae_checkpoint
        unet_ckpt = opt.phase2.ghop.unet_checkpoint

        if not os.path.exists(vqvae_ckpt):
            raise FileNotFoundError(f"VQ-VAE checkpoint not found: {vqvae_ckpt}")
        if not os.path.exists(unet_ckpt):
            raise FileNotFoundError(f"U-Net checkpoint not found: {unet_ckpt}")

        hold_loss_module = HOLDLoss(opt)
        model.hold_loss_module = hold_loss_module
        model.phase2_enabled = True
        model.phase3_enabled = False
        model.ghop_enabled = False

        print(f"✓ GHOP prior initialized (Phase 2 legacy mode)")
        print(f"✓ SDS loss weight: {opt.phase2.get('w_sds', 5000.0)}")
        print(f"✓ Grid resolution: {opt.phase2.get('grid_resolution', 16)}³")
        print("="*70 + "\n")

    else:
        # No GHOP integration
        model.hold_loss_module = None
        model.phase2_enabled = False
        model.phase3_enabled = False
        model.ghop_enabled = False
        print("\n[GHOP] Disabled - training with original HOLD losses only\n")

    # ========================================================================
    # Phase 4: Contact Refinement Configuration
    # ========================================================================
    if model.phase3_enabled:  # Phase 4 requires Phase 3
        print("\n" + "=" * 70)
        print("PHASE 4: Initializing Contact Refinement Module")
        print("=" * 70)

        try:
            # Extract contact parameters from config or args
            if hasattr(opt, 'phase4') and opt.phase4.get('enabled', False):
                contact_thresh = opt.phase4.get('contact_thresh', 0.01)
                collision_thresh = opt.phase4.get('collision_thresh', 0.005)
                contact_start_iter = opt.phase4.get('contact_start_iter', 500)
                w_contact = opt.phase4.get('w_contact', 10.0)
                mesh_resolution = opt.phase4.get('mesh_resolution', 128)
                print("[Phase 4] Using config-based initialization")
            else:
                # Fallback to default Phase 4 parameters
                contact_thresh = 0.01
                collision_thresh = 0.005
                contact_start_iter = 500
                w_contact = 10.0
                mesh_resolution = 128
                print("[Phase 4] Using default parameters")

            # Store Phase 4 config in model
            model.phase4_enabled = True
            model.contact_start_iter = contact_start_iter
            model.w_contact = w_contact
            model.mesh_resolution = mesh_resolution
            model.contact_thresh = contact_thresh
            model.collision_thresh = collision_thresh

            # Validate Phase 4 configuration
            validation_passed = validate_phase4_config(opt, model)

            if validation_passed:
                print(f"\n✓ Phase 4 contact refinement configured:")
                print(f"   - Contact start iteration: {contact_start_iter}")
                print(f"   - Contact loss weight: {w_contact}")
                print(f"   - Mesh resolution: {mesh_resolution}³ voxels")
                print(f"   - Contact threshold: {contact_thresh}m")
                print(f"   - Collision threshold: {collision_thresh}m")
                print("=" * 70 + "\n")
            else:
                logger.error("[Phase 4] Validation failed. Disabling Phase 4.")
                model.phase4_enabled = False
                print("=" * 70 + "\n")

        except Exception as e:
            logger.error(f"[Phase 4] Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            model.phase4_enabled = False
            print("\n[Phase 4] Disabled due to initialization error\n")
            print("=" * 70 + "\n")
    else:
        model.phase4_enabled = False
        print("\n[Phase 4] Disabled - Phase 3 must be enabled for Phase 4\n")
    # ========================================================================

    # ========================================================================
    # Dataset Summary Logging (FIXED VERSION)
    # ========================================================================
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)

    # Detect dataset type
    is_ghop_dataset = isinstance(trainset, GHOPHOIDataset)

    if is_ghop_dataset:
        # ================================================================
        # GHOP HOI4D Video Dataset
        # ================================================================
        print(f"Dataset type: GHOP HOI4D Video")
        print(f"  Sequence: {trainset.data_dir.name}")
        print(f"  Total frames: {trainset.n_frames}")
        print(f"  Training frame pairs: {len(trainset)}")
        print(f"  Validation frame pairs: {len(validset)}")
        print(f"  Category: {trainset.category}")
        print(f"  Split ratio: 80% train / 20% val")

        # Show frame pair indices
        if len(trainset.frame_indices) > 0:
            print(f"\n  Training frame pairs:")
            print(f"    First 3: {trainset.frame_indices[:3]}")
            if len(trainset.frame_indices) > 6:
                print(f"    ...")
            print(f"    Last 3: {trainset.frame_indices[-3:]}")

        # Check temporal fields in first sample
        sample = trainset[0]
        has_temporal = 'hA_n' in sample and 'c2w_n' in sample
        print(f"\n  Temporal fields present: {has_temporal}")
        if has_temporal:
            print(f"    ✅ Phase 5 will activate")
        else:
            print(f"    ⚠️  Phase 5 will skip")

    else:
        # ================================================================
        # HOLD Single-Image Dataset
        # ================================================================
        print(f"Dataset type: HOLD Single-Image")
        print(f"  Training images: {len(trainset)}")
        print(f"  Validation images: {len(validset)}")

        # Try to access img_paths if available
        if hasattr(trainset, 'dataset') and hasattr(trainset.dataset, 'img_paths'):
            img_paths = np.array(trainset.dataset.img_paths)
            print(f"\n  Image paths:")
            print(f"    First 3: {img_paths[:3]}")
            if len(img_paths) > 6:
                print(f"    ...")
            print(f"    Last 3: {img_paths[-3:]}")
        elif hasattr(trainset, 'img_paths'):
            # Direct access
            img_paths = np.array(trainset.img_paths)
            print(f"\n  Image paths:")
            print(f"    First 3: {img_paths[:3]}")
            if len(img_paths) > 6:
                print(f"    ...")
            print(f"    Last 3: {img_paths[-3:]}")
        else:
            print(f"  (Image path details not available)")

        print(f"\n  ⚠️  Phase 5 temporal will skip (single images)")

    print("="*70 + "\n")

    # ========================================================================
    # Continue with training setup
    # ========================================================================
    reset_all_seeds(1)

    # Checkpoint loading
    ckpt_path = None if args.ckpt_p == "" else args.ckpt_p
    # ============================================================================
    # Resume training: Use --infer_ckpt to restore full training state
    # ============================================================================
    if args.infer_ckpt != "":
        print(f"\n{'=' * 70}")
        print("RESUME TRAINING: Restoring full training state")
        print(f"{'=' * 70}")
        print(f"  Checkpoint: {args.infer_ckpt}")

        # Verify checkpoint exists
        if not os.path.exists(args.infer_ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {args.infer_ckpt}")

        # Load and display info
        ckpt_info = torch.load(args.infer_ckpt, map_location='cpu')
        current_epoch = ckpt_info.get('epoch', -1)
        global_step = ckpt_info.get('global_step', 0)

        print(f"  Checkpoint epoch: {current_epoch}")
        print(f"  Global step: {global_step}")
        print(f"  Contains optimizer: {'optimizer_states' in ckpt_info}")
        print(f"  Contains LR scheduler: {'lr_schedulers' in ckpt_info}")

        # Validate epoch range
        if current_epoch >= args.num_epoch:
            print(f"\n  ⚠️  WARNING: Checkpoint at epoch {current_epoch} >= target {args.num_epoch}")
            print(f"      Training will complete immediately.")
            print(f"      Increase --num_epoch to continue (e.g., --num_epoch {current_epoch + 10})")
        else:
            print(f"\n  ✓ Will resume from epoch {current_epoch + 1} to {args.num_epoch}")

        # ✅ KEY: Set ckpt_path for trainer.fit() WITHOUT loading weights
        ckpt_path = args.infer_ckpt
        print(f"{'=' * 70}\n")

    if args.load_ckpt != "":
        # ================================================================
        # TRANSFER LEARNING: Load HOLD weights only (exclude GHOP)
        # ================================================================
        print(f"\n{'='*70}")
        print("TRANSFER LEARNING: Loading HOLD model weights")
        print(f"{'='*70}")
        print(f"Checkpoint: {args.load_ckpt}")

        # Step 1: Load checkpoint to CPU
        sd = torch.load(args.load_ckpt, map_location='cpu')["state_dict"]
        print(f"  Loaded {len(sd)} parameters from checkpoint")

        # Step 2: Aggressive GPU memory cleanup
        if torch.cuda.is_available():
            print("  Clearing GPU memory...")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Force garbage collection
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            mem_before = torch.cuda.memory_allocated() / 1024**2
            print(f"  GPU memory before load: {mem_before:.1f} MB")

        # ================================================================
        # Step 3: FILTER OUT GHOP COMPONENTS (if Phase 3 enabled)
        # ================================================================
        if hasattr(opt, 'phase3') and opt.phase3.get('enabled', False):
            print("\n[FILTER] Phase 3 enabled - excluding GHOP components from checkpoint")

            # Keys to exclude (GHOP components already initialized from GHOP checkpoint)
            ghop_component_prefixes = [
                'vqvae.',               # VQ-VAE wrapper
                'unet.',                # U-Net wrapper (THIS IS THE KEY ONE!)
                'hand_field_builder.',  # Hand field builder
                'sds_loss.',            # SDS loss module
                'ghop_manager.',        # GHOP manager
            ]

            # Get current model structure for shape validation
            model_sd = model.state_dict()

            # Filter checkpoint
            original_count = len(sd)
            filtered_sd = {}
            excluded_by_prefix = []
            excluded_by_shape = []

            for key, value in sd.items():
                # Check if key belongs to GHOP components (by prefix)
                is_ghop_component = any(key.startswith(prefix) for prefix in ghop_component_prefixes)

                if is_ghop_component:
                    excluded_by_prefix.append(key)
                    continue

                # Check shape mismatch (catches remaining GHOP conflicts)
                if key in model_sd and value.shape != model_sd[key].shape:
                    excluded_by_shape.append(key)
                    print(f"[FILTER] Shape mismatch: {key}")
                    print(f"         Checkpoint: {value.shape} → Model: {model_sd[key].shape}")
                    continue

                # Safe to load
                filtered_sd[key] = value

            print(f"[FILTER] Results:")
            print(f"  Original checkpoint keys: {original_count}")
            print(f"  Excluded (GHOP prefix): {len(excluded_by_prefix)}")
            print(f"  Excluded (shape mismatch): {len(excluded_by_shape)}")
            print(f"  Remaining HOLD keys: {len(filtered_sd)}")

            if len(excluded_by_prefix) > 0:
                print(f"\n[FILTER] Sample excluded GHOP keys:")
                for key in excluded_by_prefix[:5]:
                    print(f"  - {key}")

            if len(excluded_by_shape) > 0:
                print(f"\n[FILTER] Keys with shape mismatch:")
                for key in excluded_by_shape[:10]:
                    print(f"  - {key}")

            sd = filtered_sd
            print(f"\n✓ Filtered checkpoint ready: {len(sd)} parameters")
        else:
            print("\n[FILTER] Phase 3 disabled - loading all checkpoint weights")

        # ================================================================
        # Step 4: Load filtered state dict
        # ================================================================
        print(f"\nLoading {len(sd)} parameters into model...")
        missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)

        print(f"\n✓ Checkpoint loaded:")
        print(f"  - Parameters loaded: {len(sd)}")
        print(f"  - Missing in checkpoint: {len(missing_keys)}")
        print(f"  - Unexpected in checkpoint: {len(unexpected_keys)}")

        # Log missing keys breakdown (if Phase 3 enabled)
        if hasattr(opt, 'phase3') and opt.phase3.get('enabled', False):
            ghop_component_prefixes = ['vqvae.', 'unet.', 'hand_field_builder.', 'sds_loss.', 'ghop_manager.']
            ghop_missing = [k for k in missing_keys if any(k.startswith(p) for p in ghop_component_prefixes)]
            hold_missing = [k for k in missing_keys if k not in ghop_missing]

            if len(ghop_missing) > 0:
                print(f"\n✅ GHOP components not loaded from checkpoint (expected): {len(ghop_missing)}")
                print(f"   These were initialized from GHOP checkpoint instead")

            if len(hold_missing) > 0:
                print(f"\n⚠️  HOLD components missing from checkpoint: {len(hold_missing)}")
                if len(hold_missing) <= 10:
                    for key in hold_missing:
                        print(f"   - {key}")

        # Step 5: Post-load cleanup
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            mem_after = torch.cuda.memory_allocated() / 1024**2
            print(f"\n  GPU memory after load: {mem_after:.1f} MB")
            print(f"  Memory increase: {mem_after - mem_before:.1f} MB")

        print(f"\n✅ Transfer learning setup complete")
        print(f"✅ Training will start from epoch 0")
        print(f"{'='*70}\n")
        ckpt_path = None

    if args.load_pose != "":
        sd = torch.load(args.load_pose)["state_dict"]
        mysd = model.state_dict()
        print("Loading pose from: ", args.load_pose)
        print("Keys in loaded state dict:")
        for k, v in sd.items():
            if ".params." in k or "object_model.obj_scale" in k:
                assert k in mysd, f"{k} not in mysd"
                print("\t" + k)
                mysd[k] = v
        print("End of keys")
        model.load_state_dict(mysd, strict=True)
        ckpt_path = None

    # ================================================================
    # CRITICAL FIX: Create DataLoaders for PyTorch Lightning
    # ================================================================
    # PyTorch Lightning trainer.fit() requires DataLoader objects,
    # not raw Dataset objects. This is especially important for
    # GHOPHOIDataset which doesn't have a built-in DataLoader wrapper.

    print("\n" + "="*70)
    print("DATALOADER CREATION")
    print("="*70)

    from torch.utils.data import DataLoader

    # Get batch size from config or args
    if hasattr(opt.dataset, 'train') and 'batch_size' in opt.dataset.train:
        train_batch_size = opt.dataset.train.batch_size
    else:
        train_batch_size = 2  # Default

    if hasattr(opt.dataset, 'valid') and 'batch_size' in opt.dataset.valid:
        val_batch_size = opt.dataset.valid.batch_size
    else:
        val_batch_size = 1  # Default

    # Get other DataLoader parameters
    num_workers = opt.dataset.train.get('num_workers', 0)
    shuffle_train = opt.dataset.train.get('shuffle', True)

    # ✅ FIXED CODE:
    # ================================================================
    # CRITICAL FIX: Respect --no-pin-memory flag
    # ================================================================
    use_pin_memory = not args.no_pin_memory if hasattr(args, 'no_pin_memory') else False

    # Create training DataLoader
    train_loader = DataLoader(
        trainset,
        batch_size=train_batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=use_pin_memory,  # ✅ FIX: Respect command-line flag
        drop_last=opt.dataset.train.get('drop_last', False)
    )

    print(f"Training DataLoader:")
    print(f"  Dataset type: {type(trainset).__name__}")
    print(f"  Dataset size: {len(trainset)}")
    print(f"  Batch size: {train_batch_size}")
    print(f"  Num workers: {num_workers}")
    print(f"  Pin memory: {use_pin_memory}  # ✅ {'DISABLED' if not use_pin_memory else 'ENABLED'}")
    print(f"  Shuffle: {shuffle_train}")
    print(f"  Total batches: {len(train_loader)}")

    # Create validation DataLoader
    val_loader = DataLoader(
        validset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,  # ✅ FIX: Respect command-line flag
        drop_last=False
    )

    print(f"\nValidation DataLoader:")
    print(f"  Dataset type: {type(validset).__name__}")
    print(f"  Dataset size: {len(validset)}")
    print(f"  Batch size: {val_batch_size}")
    print(f"  Total batches: {len(val_loader)}")

    print("="*70 + "\n")

    # ================================================================
    # Start training with DataLoaders (not raw datasets)
    # ================================================================
    print("Starting training...")
    print(f"  Max epochs: {args.num_epoch}")
    print(f"  Checkpoint: {ckpt_path if ckpt_path else 'Training from scratch'}")
    print("")

    # ✅ FIX: Pass DataLoaders instead of raw datasets
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()


'''
training command
# Test with environment variable
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train.py \
    --config confs/ghop_production_chunked_20251022_172403.yaml \
    --case ghop_bottle_1 \
    --gpu_id 0 \
    --num_epoch 60 \
    --load_ckpt logs/6c533c888/checkpoints/last.ckpt \
    --no-pin-memory \
    --no-comet 2>&1 | tee test_defrag_fix_$(date +%Y%m%d_%H%M%S).log
'''