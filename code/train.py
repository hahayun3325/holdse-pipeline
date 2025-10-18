from pprint import pprint
import os
import os.path as op

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
    Create dataset with GHOP HOI4D support.

    This function replaces the hardcoded ImageDataset creation
    to support both HOLD (single images) and GHOP HOI4D (video sequences).

    Args:
        dataset_config: Dataset configuration from YAML
        args: Command-line arguments

    Returns:
        Dataset instance (ImageDataset or GHOPHOIDataset)
    """
    from src.datasets.utils import create_dataset as create_image_dataset

    # ================================================================
    # STEP 1: Check dataset type from config
    # ================================================================
    dataset_type = dataset_config.get('type', 'train')

    # Check for GHOP-specific markers in config
    use_ghop = (
            hasattr(args, 'use_ghop') and args.use_ghop or
            dataset_config.get('dataset_type', '') == 'ghop_hoi' or
            dataset_config.get('is_video', False) or
            'ghop' in args.case.lower()  # Case name contains 'ghop'
    )

    # ================================================================
    # STEP 2: Create appropriate dataset
    # ================================================================
    if use_ghop:
        logger.info("=" * 70)
        logger.info("CREATING GHOP HOI4D VIDEO DATASET")
        logger.info("=" * 70)

        # Extract GHOP data directory from args.case
        # Expected: args.case = 'ghop_bottle_1'
        # Maps to: ~/Projects/ghop/data/HOI4D_clip/Bottle_1
        case_name = args.case.lower()

        # Parse object name from case (e.g., 'ghop_bottle_1' -> 'Bottle_1')
        if 'ghop_' in case_name:
            obj_name = case_name.replace('ghop_', '')
            # Convert back to title case (bottle_1 -> Bottle_1)
            obj_name = '_'.join(word.capitalize() for word in obj_name.split('_'))
        else:
            obj_name = 'Bottle_1'  # Default fallback

        # Build GHOP data path
        ghop_root = os.path.expanduser('~/Projects/ghop/data/HOI4D_clip')
        data_dir = os.path.join(ghop_root, obj_name)

        logger.info(f"GHOP Configuration:")
        logger.info(f"  Case name: {args.case}")
        logger.info(f"  Object name: {obj_name}")
        logger.info(f"  Data directory: {data_dir}")
        logger.info(f"  Split: {dataset_type}")

        # Verify directory exists
        if not os.path.exists(data_dir):
            raise FileNotFoundError(
                f"GHOP data directory not found: {data_dir}\n"
                f"Expected structure: ~/Projects/ghop/data/HOI4D_clip/{obj_name}/\n"
                f"Please ensure GHOP HOI4D dataset is extracted correctly."
            )

        # Create GHOP HOI4D dataset
        dataset = GHOPHOIDataset(
            data_dir=data_dir,
            split=dataset_type,  # 'train' or 'val'
            args=args
        )

        logger.info(f"✓ GHOP dataset created:")
        logger.info(f"  Total frames: {dataset.n_frames}")
        logger.info(f"  Frame pairs: {len(dataset)}")
        logger.info(f"  Category: {dataset.category}")

        # ✅ CRITICAL: Verify temporal fields present
        sample = dataset[0]
        if 'hA_n' in sample and 'c2w_n' in sample:
            logger.info(f"  ✅ Temporal fields: hA_n, c2w_n present")
            logger.info(f"  ✅ Phase 5 will ACTIVATE")
        else:
            logger.warning(f"  ⚠️  Temporal fields missing!")
            logger.warning(f"  ⚠️  Phase 5 will SKIP")

        logger.info("=" * 70)

    else:
        # ================================================================
        # HOLD Single-Image Dataset (Original)
        # ================================================================
        logger.info("=" * 70)
        logger.info("CREATING HOLD SINGLE-IMAGE DATASET")
        logger.info("=" * 70)

        dataset = create_image_dataset(dataset_config, args)

        logger.info(f"✓ HOLD dataset created:")
        logger.info(f"  Images: {len(dataset)}")
        logger.info(f"  Type: Single-image")
        logger.info(f"  ⚠️  Phase 5 temporal will SKIP (no video data)")
        logger.info("=" * 70)

    return dataset

def main():
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

    trainer = pl.Trainer(
        gpus=1,
        accelerator="gpu",
        # gradient_clip_val=0.5,removed - handled manually in HOLD.training_step for GHOP compatibility
        callbacks=[checkpoint_callback],
        max_epochs=args.num_epoch,
        check_val_every_n_epoch=args.eval_every_epoch,
        log_every_n_steps=args.log_every,
        num_sanity_val_steps=0,
        logger=False,
    )

    pprint(args)

    trainset = create_dataset(opt.dataset.train, args)
    validset = create_dataset(opt.dataset.valid, args)

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
    # Dataset and Training Setup
    # ========================================================================
    print("img_paths: ")
    img_paths = np.array(trainset.dataset.img_paths)
    print(img_paths[:3])
    print("...")
    print(img_paths[-3:])

    reset_all_seeds(1)

    # Checkpoint loading
    ckpt_path = None if args.ckpt_p == "" else args.ckpt_p
    if args.load_ckpt != "":
        sd = torch.load(args.load_ckpt)["state_dict"]
        model.load_state_dict(sd)
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

    # Start training
    trainer.fit(model, trainset, validset, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
