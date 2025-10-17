import os
import os.path as op
import sys

import pytorch_lightning as pl
import torch
import torch.optim as optim
from tqdm import tqdm
from typing import Dict, Optional, List, Tuple, Any  # ✅ ADD THIS LINE

import src.hold.hold_utils as hold_utils
from src.hold.loss import Loss
from src.hold.hold_net import HOLDNet
from src.utils.metrics import Metrics

sys.path = [".."] + sys.path
from common.xdict import xdict

import src.utils.debug as debug
import src.utils.vis_utils as vis_utils
import common.comet_utils as comet_utils
import numpy as np
from loguru import logger
from src.datasets.utils import split_input, merge_output
from src.model.ghop.ghop_loss import GHOPSDSLoss
from src.model.ghop.text_template import create_text_template
from src.hold.loss import HOLDLoss
from src.model.ghop.ghop_prior import TwoStageTrainingManager
from src.model.ghop.diffusion_prior import GHOPDiffusionPrior
from src.model.ghop.temporal_consistency import TemporalConsistencyModule
from src.model.ghop.adaptive_contact_zones import AdaptiveContactZones
from src.training.phase5_scheduler import Phase5TrainingScheduler


# ========================================================================
# PHASE 2: GHOP VALIDATION FUNCTION
# ========================================================================
def validate_phase2_config(opt):
    """
    Validate Phase 2 GHOP configuration and check required files exist.

    Args:
        opt: Configuration object with phase2 section

    Returns:
        opt: Validated configuration object
    """
    # Check if Phase 2 is enabled
    if not hasattr(opt, 'phase2') or not opt.phase2.get('enabled', False):
        return opt

    phase2_cfg = opt.phase2

    # Validate VQ-VAE checkpoint
    vqvae_path = phase2_cfg.ghop.vqvae_checkpoint
    if not os.path.exists(vqvae_path):
        raise FileNotFoundError(
            f"[Phase 2 Error] VQ-VAE checkpoint not found: {vqvae_path}\n"
            f"Please download GHOP checkpoints from: https://judyye.github.io/g-hop-www/\n"
            f"Expected location: {vqvae_path}"
        )
    logger.info(f"✓ Found VQ-VAE checkpoint: {vqvae_path}")

    # Validate U-Net checkpoint
    unet_path = phase2_cfg.ghop.unet_checkpoint
    if not os.path.exists(unet_path):
        raise FileNotFoundError(
            f"[Phase 2 Error] U-Net checkpoint not found: {unet_path}\n"
            f"Please download GHOP checkpoints from: https://judyye.github.io/g-hop-www/\n"
            f"Expected location: {unet_path}"
        )
    logger.info(f"✓ Found U-Net checkpoint: {unet_path}")

    # Validate text template library (optional)
    text_lib_path = phase2_cfg.ghop.get('text_lib', None)
    if text_lib_path and not os.path.exists(text_lib_path):
        logger.warning(
            f"Text template library not found: {text_lib_path}\n"
            f"Using default category text prompts."
        )

    # Validate warmup iterations
    warmup_iters = phase2_cfg.get('warmup_iters', 1000)
    if warmup_iters < 0:
        logger.warning(f"Invalid warmup_iters {warmup_iters}, setting to 0")
        phase2_cfg['warmup_iters'] = 0

    # Validate SDS weight
    w_sds = phase2_cfg.get('w_sds', 5000.0)
    if w_sds <= 0:
        logger.warning(f"Invalid w_sds {w_sds}, setting to 5000.0")
        phase2_cfg['w_sds'] = 5000.0

    logger.info(
        f"Phase 2 configuration validated:\n"
        f"  - SDS weight: {phase2_cfg.get('w_sds', 5000.0)}\n"
        f"  - Warmup iterations: {phase2_cfg.get('warmup_iters', 1000)}\n"
        f"  - Grid resolution: {phase2_cfg.get('grid_resolution', 16)}³\n"
        f"  - Guidance scale: {phase2_cfg.sds.get('guidance_scale', 4.0)}"
    )

    return opt
# ========================================================================


class HOLD(pl.LightningModule):
    def __init__(self, opt, args) -> None:
        super().__init__()

        self.opt = opt
        self.args = args
        num_frames = args.n_images

        data_path = os.path.join("./data", args.case, f"build/data.npy")
        entities = np.load(data_path, allow_pickle=True).item()["entities"]

        betas_r = entities["right"]["mean_shape"] if "right" in entities else None
        betas_l = entities["left"]["mean_shape"] if "left" in entities else None

        self.model = HOLDNet(
            opt.model,
            betas_r,
            betas_l,
            num_frames,
            args,
        )

        for node in self.model.nodes.values():
            if self.args.freeze_pose:
                node.params.freeze()
            else:
                node.params.defrost()

        # Initialize metrics early
        self.metrics = Metrics(args.experiment)

        # Initialize phase flags
        self.phase2_enabled = False
        self.phase3_enabled = False
        self.phase4_enabled = False
        self.phase5_enabled = False
        self.ghop_enabled = False

        # ====================================================================
        # PHASE 3: GHOP Two-Stage Training Integration
        # ====================================================================
        if hasattr(opt, 'phase3') and opt.phase3.get('enabled', False):
            logger.info("\n" + "=" * 70)
            logger.info("PHASE 3: Initializing GHOP Two-Stage Training")
            logger.info("=" * 70)

            # Validate Phase 3 configuration
            opt = validate_phase2_config(opt)
            phase3_cfg = opt.phase3

            # Determine initialization mode
            use_modular_init = phase3_cfg.get('use_modular_init', False)

            if use_modular_init:
                # ============================================================
                # MODE A: Direct Modular Initialization (Recommended)
                # ============================================================
                logger.info("Using modular GHOP component initialization...")

                from src.model.ghop.autoencoder import GHOPVQVAEWrapper
                from src.model.ghop.diffusion import GHOP3DUNetWrapper
                from src.model.ghop.hand_field import HandFieldBuilder
                from src.model.ghop.ghop_loss import SDSLoss
                from src.model.ghop.ghop_prior import TwoStageTrainingManager

                # Get device
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                # ============================================================
                # CRITICAL FIX: Check pretrained flags BEFORE file access
                # ============================================================
                vqvae_use_pretrained = phase3_cfg.ghop.get('vqvae_use_pretrained', False)
                unet_use_pretrained = phase3_cfg.ghop.get('unet_use_pretrained', False)

                model_checkpoint = phase3_cfg.ghop.get('model_checkpoint', 'checkpoints/ghop/last.ckpt')

                # Determine if we need the checkpoint file
                need_checkpoint = vqvae_use_pretrained or unet_use_pretrained

                if need_checkpoint:
                    # Only check file existence if we're loading pretrained weights
                    logger.info(f"Loading unified GHOP checkpoint from: {model_checkpoint}")
                    logger.info("  This checkpoint contains BOTH VQ-VAE and U-Net components")

                    # Verify checkpoint exists
                    if not os.path.exists(model_checkpoint):
                        raise FileNotFoundError(
                            f"GHOP checkpoint not found: {model_checkpoint}\n"
                            f"Expected: Unified checkpoint (~1.1 GB) from GHOP project\n"
                            f"Run: ln -s ~/Projects/ghop/output/joint_3dprior/mix_data/checkpoints/last.ckpt {model_checkpoint}"
                        )

                    # Load checkpoint once
                    logger.info("Loading checkpoint...")
                    unified_checkpoint = torch.load(model_checkpoint, map_location='cpu')

                    if 'state_dict' not in unified_checkpoint:
                        raise ValueError(f"Invalid checkpoint structure: missing 'state_dict'")

                    state_dict = unified_checkpoint['state_dict']
                    logger.info(f"✓ Unified checkpoint loaded: {len(state_dict)} parameters")

                    # Analyze checkpoint structure
                    vqvae_keys = [k for k in state_dict.keys() if 'first_stage_model' in k or 'encoder' in k or 'decoder' in k or 'quantize' in k]
                    unet_keys = [k for k in state_dict.keys() if 'model' in k and 'first_stage' not in k]

                    logger.info(f"  VQ-VAE components: {len(vqvae_keys)} parameters")
                    logger.info(f"  U-Net components: {len(unet_keys)} parameters")

                else:
                    # Random initialization - no checkpoint needed
                    logger.warning("")
                    logger.warning("=" * 70)
                    logger.warning("RUNNING WITH RANDOM INITIALIZATION")
                    logger.warning("=" * 70)
                    logger.warning("  VQ-VAE: Random weights (vqvae_use_pretrained=False)")
                    logger.warning("  U-Net: Random weights (unet_use_pretrained=False)")
                    logger.warning("")
                    logger.warning("This is acceptable for:")
                    logger.warning("  ✓ Sanity checks (verify pipeline works)")
                    logger.warning("  ✓ Architecture debugging")
                    logger.warning("  ✓ Testing data flow")
                    logger.warning("")
                    logger.warning("Limitations:")
                    logger.warning("  ✗ SDS loss will not provide effective guidance")
                    logger.warning("  ✗ GHOP prior cannot guide reconstruction")
                    logger.warning("")
                    logger.warning("For production training, set:")
                    logger.warning("  vqvae_use_pretrained: true")
                    logger.warning("  unet_use_pretrained: true")
                    logger.warning("=" * 70)
                    logger.warning("")

                # ============================================================
                # Initialize VQ-VAE
                # ============================================================
                logger.info("Initializing VQ-VAE...")
                self.vqvae = GHOPVQVAEWrapper(
                    vqvae_ckpt_path=model_checkpoint if vqvae_use_pretrained else None,
                    device=device,
                    use_hand_field=phase3_cfg.get('use_hand_field', True)
                )

                if vqvae_use_pretrained:
                    logger.info("✓ VQ-VAE initialized with pretrained weights")
                else:
                    logger.info("✓ VQ-VAE initialized with RANDOM weights")

                # ============================================================
                # Initialize 3D U-Net
                # ============================================================
                logger.info("Initializing U-Net...")
                self.unet = GHOP3DUNetWrapper(
                    unet_ckpt_path=model_checkpoint if unet_use_pretrained else None,
                    device=device
                )

                if unet_use_pretrained:
                    logger.info("✓ U-Net initialized with pretrained weights")
                else:
                    logger.info("✓ U-Net initialized with RANDOM weights")

                # ============================================================
                # Initialize Hand Field Builder
                # ============================================================
                mano_server = None
                for node in self.model.nodes.values():
                    if 'right' in node.node_id.lower() or 'left' in node.node_id.lower():
                        mano_server = node.server
                        break

                if mano_server is None:
                    raise ValueError("No hand node found in model. Cannot initialize HandFieldBuilder.")

                logger.info("Initializing Hand Field Builder...")
                self.hand_field_builder = HandFieldBuilder(
                    mano_server=mano_server,
                    resolution=phase3_cfg.get('grid_resolution', 64),
                    spatial_limit=phase3_cfg.get('spatial_limit', 1.5)
                )
                logger.info("✓ Hand Field Builder initialized")

                # ============================================================
                # Initialize SDS Loss Module
                # ============================================================
                logger.info("Initializing SDS Loss Module...")
                self.sds_loss = SDSLoss(
                    vqvae_wrapper=self.vqvae,
                    unet_wrapper=self.unet,
                    hand_field_builder=self.hand_field_builder,
                    guidance_scale=phase3_cfg.sds.get('guidance_scale', 4.0),
                    min_step_ratio=phase3_cfg.sds.get('min_step_ratio', 0.02),
                    max_step_ratio=phase3_cfg.sds.get('max_step_ratio', 0.98),
                    diffusion_steps=phase3_cfg.sds.get('diffusion_steps', 1000)
                )
                logger.info("✓ SDS Loss Module initialized")

                # ============================================================
                # Initialize Two-Stage Training Manager
                # ============================================================
                logger.info("Initializing Two-Stage Training Manager...")
                self.ghop_manager = TwoStageTrainingManager(
                    sds_loss_module=self.sds_loss,
                    sds_iters=phase3_cfg.get('sds_iters', 500),
                    contact_iters=phase3_cfg.get('contact_iters', 100),
                    max_sds_weight=phase3_cfg.get('w_sds', 5000.0),
                    max_contact_weight=phase3_cfg.get('w_contact', 10.0)
                )
                logger.info("✓ Two-Stage Training Manager initialized")

                # Not using HOLDLoss wrapper in modular mode
                self.hold_loss_module = None

                # Final summary
                if need_checkpoint:
                    logger.info(
                        f"✓ Phase 3 initialized with PRETRAINED weights:\n"
                        f"   - Checkpoint: {model_checkpoint}\n"
                        f"   - VQ-VAE: {'Pretrained' if vqvae_use_pretrained else 'Random'}\n"
                        f"   - U-Net: {'Pretrained' if unet_use_pretrained else 'Random'}"
                    )
                else:
                    logger.info(
                        f"✓ Phase 3 initialized with RANDOM weights (sanity mode):\n"
                        f"   - VQ-VAE: Random initialization\n"
                        f"   - U-Net: Random initialization\n"
                        f"   - SDS weight: {phase3_cfg.get('w_sds', 5000.0)}\n"
                        f"   - Note: Reduced effectiveness expected"
                    )
            else:
                # ============================================================
                # MODE B: Legacy Wrapper
                # ============================================================
                logger.info("Using legacy Phase 3 HOLDLoss wrapper initialization...")

                from src.model.ghop.ghop_prior import TwoStageTrainingManager  # FIX: Added import

                # Initialize Phase 3 HOLDLoss module
                self.hold_loss_module = HOLDLoss(opt)

                # Initialize Two-Stage Training Manager
                self.ghop_manager = TwoStageTrainingManager(
                    sds_loss_module=self.hold_loss_module,
                    sds_iters=phase3_cfg.get('sds_iters', 500),
                    contact_iters=phase3_cfg.get('contact_iters', 100),
                    max_sds_weight=phase3_cfg.get('w_sds', 5000.0),
                    max_contact_weight=phase3_cfg.get('w_contact', 10.0)
                )

                # Set component references to None
                self.vqvae = None
                self.unet = None
                self.hand_field_builder = None
                self.sds_loss = None

                logger.info(
                    f"✓ Phase 3 (Legacy HOLDLoss) initialized successfully:\n"
                    f"   - VQ-VAE: {phase3_cfg.ghop.vqvae_checkpoint}\n"
                    f"   - U-Net: {phase3_cfg.ghop.unet_checkpoint}\n"
                    f"   - Stage 1 (SDS): {phase3_cfg.get('sds_iters', 500)} iterations\n"
                    f"   - Stage 2 (Contact): {phase3_cfg.get('contact_iters', 100)} iterations\n"
                    f"   - Max SDS weight: {phase3_cfg.get('w_sds', 5000.0)}\n"
                    f"   - Max Contact weight: {phase3_cfg.get('w_contact', 10.0)}"
                )

            self.phase3_enabled = True
            self.ghop_enabled = True
            logger.info("=" * 70 + "\n")

        else:
            # No Phase 3 - initialize to None
            self.hold_loss_module = None
            self.ghop_manager = None
            self.vqvae = None
            self.unet = None
            self.hand_field_builder = None
            self.sds_loss = None

        # ====================================================================
        # PHASE 2: Backward Compatibility
        # ====================================================================
        if hasattr(opt, 'phase2') and opt.phase2.get('enabled', False):
            if not self.phase3_enabled:  # Only enable if Phase 3 is not active
                logger.info("\n" + "=" * 70)
                logger.info("PHASE 2: Initializing GHOP Prior (Legacy Mode)")
                logger.info("=" * 70)

                opt = validate_phase2_config(opt)

                # Initialize Phase 2 HOLDLoss
                self.hold_loss_module = HOLDLoss(opt)

                self.phase2_enabled = True
                self.phase3_enabled = False
                self.ghop_enabled = False
                self.ghop_manager = None

                logger.info(
                    f"✓ Phase 2 (legacy) initialized:\n"
                    f"   - VQ-VAE: {opt.phase2.ghop.vqvae_checkpoint}\n"
                    f"   - U-Net: {opt.phase2.ghop.unet_checkpoint}\n"
                    f"   - SDS weight: {opt.phase2.get('w_sds', 5000.0)}"
                )
                logger.info("=" * 70 + "\n")

        # ====================================================================
        # PHASE 4: Initialize Contact Refinement Module
        # ====================================================================
        if hasattr(opt, 'phase4') and opt.phase4.get('enabled', False):
            # Phase 4 requires Phase 3 with modular initialization
            if not self.phase3_enabled:
                logger.error("[Phase 4] Cannot enable Phase 4 without Phase 3. Skipping...")
                self.phase4_enabled = False
            elif self.vqvae is None or self.hand_field_builder is None:  # FIX: Consistent naming
                logger.error(
                    "[Phase 4] Cannot enable Phase 4 without modular Phase 3 initialization.\n"
                    "Set phase3.use_modular_init=true in config."
                )
                self.phase4_enabled = False
            else:
                logger.info("\n" + "=" * 70)
                logger.info("PHASE 4: Initializing Contact Refinement Module")
                logger.info("=" * 70)

                phase4_cfg = opt.phase4

                # Import Phase 4 modules
                from src.model.ghop.mesh_extraction import GHOPMeshExtractor
                from src.model.ghop.contact_refinement import GHOPContactRefinement

                # Get device
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                # Initialize mesh extractor
                logger.info("Initializing GHOP Mesh Extractor...")
                self.mesh_extractor = GHOPMeshExtractor(
                    vqvae_wrapper=self.vqvae,
                    resolution=phase4_cfg.get('mesh_resolution', 128)
                )
                logger.info(f"✓ Mesh extractor initialized (resolution: {phase4_cfg.get('mesh_resolution', 128)}³)")

                # Initialize contact refiner
                logger.info("Initializing GHOP Contact Refinement...")
                self.contact_refiner = GHOPContactRefinement(
                    contact_thresh=phase4_cfg.get('contact_thresh', 0.01),
                    collision_thresh=phase4_cfg.get('collision_thresh', 0.005),
                    contact_zones='zones'
                )
                logger.info("✓ Contact refiner initialized")

                # Store Phase 4 hyperparameters
                self.contact_start_iter = phase4_cfg.get('contact_start_iter', 500)
                self.w_contact = phase4_cfg.get('w_contact', 10.0)
                self.mesh_resolution = phase4_cfg.get('mesh_resolution', 128)
                self.contact_warmup_iters = phase4_cfg.get('contact_warmup_iters', 100)
                self.log_contact_every = phase4_cfg.get('log_contact_every', 50)

                # Enable Phase 4 flag
                self.phase4_enabled = True

                logger.info(
                    f"\n✓ Phase 4 initialized successfully:\n"
                    f"   - Contact start iteration: {self.contact_start_iter}\n"
                    f"   - Contact loss weight: {self.w_contact}\n"
                    f"   - Mesh resolution: {self.mesh_resolution}³ voxels\n"
                    f"   - Contact threshold: {phase4_cfg.get('contact_thresh', 0.01)}m\n"
                    f"   - Collision threshold: {phase4_cfg.get('collision_thresh', 0.005)}m\n"
                    f"   - Warmup iterations: {self.contact_warmup_iters}"
                )
                logger.info("=" * 70 + "\n")
        else:
            self.phase4_enabled = False
            logger.info("[Phase 4] Disabled - configure phase4.enabled=true in config to enable\n")

        # ====================================================================
        # LOSS MODULE INITIALIZATION (Final Step)
        # ====================================================================
        logger.info("\n" + "=" * 70)
        logger.info("LOSS MODULE INITIALIZATION")
        logger.info("=" * 70)

        # Always use standard Loss class - Phase 3/4 losses handled separately in training_step
        logger.info("[Loss] Initializing standard Loss module...")
        self.loss = Loss(args)

        if self.phase4_enabled:
            logger.info("✓ Standard Loss initialized (Phase 4 contact handled separately)")
        elif self.phase3_enabled or self.phase2_enabled:
            logger.info("✓ Standard Loss initialized (GHOP SDS handled separately)")
        else:
            logger.info("✓ Standard Loss initialized (baseline configuration)")

        logger.info("=" * 70 + "\n")

        # ====================================================================
        # PHASE 5: ADVANCED GHOP INTEGRATION
        # ====================================================================
        if hasattr(opt, 'phase5') and opt.phase5.get('enabled', False):
            if not self.phase3_enabled:
                logger.error("Phase 5: Cannot enable Phase 5 without Phase 3. Skipping...")
                self.phase5_enabled = False
            elif self.vqvae is None or self.hand_field_builder is None:  # FIX: Consistent naming
                logger.error(
                    "Phase 5: Cannot enable Phase 5 without modular Phase 3 initialization. "
                    "Set phase3.use_modular_init=true in config."
                )
                self.phase5_enabled = False
            else:
                logger.info("=" * 70)
                logger.info("PHASE 5: Advanced GHOP Integration")
                logger.info("=" * 70)

                phase5_cfg = opt.phase5
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                # FIX: Add all Phase 5 imports at the beginning
                from src.model.ghop.diffusion_prior import GHOPDiffusionPrior
                from src.model.ghop.temporal_consistency import TemporalConsistencyModule
                from src.model.ghop.adaptive_contact_zones import AdaptiveContactZones
                from src.training.phase5_scheduler import Phase5TrainingScheduler

                # ============================================================
                # Component 1: Diffusion Prior for Geometry Guidance
                # ============================================================
                logger.info("Initializing GHOP Diffusion Prior...")
                self.diffusion_prior = GHOPDiffusionPrior(
                    vqvae_wrapper=self.vqvae,
                    unet_wrapper=self.unet,
                    handfield_builder=self.hand_field_builder,  # FIX: Consistent naming
                    guidance_scale=phase5_cfg.get('guidance_scale', 4.0),
                    min_step_ratio=phase5_cfg.get('min_step', 0.02),
                    max_step_ratio=phase5_cfg.get('max_step', 0.98),
                    prediction_respacing=phase5_cfg.get('prediction_respacing', 100),
                    w_schedule=phase5_cfg.get('w_schedule', 'dream'),
                    device=device
                )
                logger.info(f"  ✓ Diffusion prior initialized (guidance_scale={phase5_cfg.get('guidance_scale', 4.0)})")

                # ============================================================
                # Component 2: Temporal Consistency Module
                # ============================================================
                logger.info("Initializing Temporal Consistency Module...")
                self.temporal_module = TemporalConsistencyModule(
                    window_size=phase5_cfg.get('temporal_window', 5),
                    w_velocity=phase5_cfg.get('w_velocity', 0.5),
                    w_acceleration=phase5_cfg.get('w_acceleration', 0.1),
                    w_camera_motion=phase5_cfg.get('w_camera_motion', 0.3),
                    adaptive_weight=phase5_cfg.get('adaptive_weight', True)
                )
                logger.info(f"  ✓ Temporal module initialized (window={phase5_cfg.get('temporal_window', 5)})")

                # ============================================================
                # Component 3: Adaptive Contact Zones
                # ============================================================
                logger.info("Initializing Adaptive Contact Zones...")
                self.adaptive_contacts = AdaptiveContactZones(
                    proximity_threshold=phase5_cfg.get('proximity_threshold', 0.015),
                    min_contact_verts=phase5_cfg.get('min_contact_verts', 5),
                    max_contact_verts=phase5_cfg.get('max_contact_verts', 50),
                    update_frequency=phase5_cfg.get('contact_update_freq', 10),
                    penalize_palm=phase5_cfg.get('penalize_palm', True)
                )
                logger.info(
                    f"  ✓ Adaptive contacts initialized (threshold={phase5_cfg.get('proximity_threshold', 0.015)}m)")

                # ============================================================
                # Component 4: Phase 5 Training Scheduler
                # ============================================================
                logger.info("Initializing Phase 5 Training Scheduler...")

                # ================================================================
                # FIX 3: Extract phase3_start from config
                # ================================================================
                phase3_start = opt.phase3.get('phase3_start_iter', 0)
                phase4_start = self.contact_start_iter if hasattr(self, 'contact_start_iter') else 500

                self.phase5_scheduler = Phase5TrainingScheduler(
                    total_iterations=phase5_cfg.get('total_iterations', 1000),
                    warmup_iters=phase5_cfg.get('warmup_iters', 0),      # Will be 0 after fix
                    phase3_start=phase3_start,                            # ✅ From config, not hardcoded
                    phase4_start=phase4_start,                            # = 20
                    phase5_start=phase5_cfg.get('phase5_start_iter', 100),  # = 100
                    finetune_start=phase5_cfg.get('finetune_start_iter', 800)  # = 800
                )

                # Store Phase 5 hyperparameters
                self.phase5_start_iter = phase5_cfg.get('phase5_start_iter', 600)
                self.w_temporal = phase5_cfg.get('w_temporal', 1.0)
                self.log_phase5_every = phase5_cfg.get('log_phase5_every', 50)
                self.enable_geometry_sampling = phase5_cfg.get('enable_geometry_sampling', False)
                self.phase5_enabled = True

                logger.info(
                    f"✓ Phase 5 initialized successfully\n"
                    f"  - Diffusion guidance scale: {phase5_cfg.get('guidance_scale', 4.0)}\n"
                    f"  - Temporal window: {phase5_cfg.get('temporal_window', 5)} frames\n"
                    f"  - Contact proximity: {phase5_cfg.get('proximity_threshold', 0.015)}m\n"
                    f"  - Phase 5 start iteration: {self.phase5_start_iter}\n"
                    f"  - Fine-tuning starts: {phase5_cfg.get('finetune_start_iter', 800)}"
                )
                logger.info("=" * 70)
        else:
            self.phase5_enabled = False
            self.diffusion_prior = None
            self.temporal_module = None
            self.adaptive_contacts = None
            self.phase5_scheduler = None
            logger.info("Phase 5: Disabled - configure phase5.enabled=true in config to enable")

        # ================================================================
        # ADD: Logging frequency attributes
        # ================================================================
        self.log_ghop_every = args.log_every if args is not None else 10
        self.log_contact_every = getattr(args, 'log_contact_every', 10)
        self.log_phase5_every = getattr(args, 'log_phase5_every', 10)

        logger.debug(f"[HOLD] Logging frequencies initialized:")
        logger.debug(f"  log_ghop_every: {self.log_ghop_every}")
        logger.debug(f"  log_contact_every: {self.log_contact_every}")
        logger.debug(f"  log_phase5_every: {self.log_phase5_every}")

    def save_misc(self):
        out = {}

        dataset = self.trainset.dataset
        K = dataset.intrinsics_all[0]
        w2c = dataset.extrinsics_all[0]

        for node in self.model.nodes.values():
            if "object" in node.node_id:
                out[f"{node.node_id}.obj_scale"] = node.server.object_model.obj_scale

        out["img_paths"] = dataset.img_paths
        out["K"] = K
        out["w2c"] = w2c
        out["scale"] = dataset.scale
        mesh_dict = self.meshing_cano("misc")
        out.update(mesh_dict)
        out_p = f"{self.args.log_dir}/misc/{self.global_step:09d}.npy"
        os.makedirs(op.dirname(out_p), exist_ok=True)
        np.save(out_p, out)
        print(f"Exported misc to {out_p}")

    def configure_optimizers(self):
        base_lr = self.args.lr
        node_params = set()
        params = []

        # Collect pose parameters for each node
        for node in self.model.nodes.values():
            node_parameters = set(node.params.parameters())
            node_params.update(node_parameters)
            params.append(
                {
                    "params": list(node_parameters),
                    "lr": base_lr * 0.1,
                }
            )

        # Neural network parameters
        main_params = [p for p in self.model.parameters() if p not in node_params]
        if main_params:
            params.append({"params": main_params, "lr": base_lr})

        self.optimizer = optim.Adam(params, lr=base_lr, eps=1e-8)

        return [self.optimizer], []

    def condition_training(self):
        import common.torch_utils as torch_utils

        if self.global_step in []:
            logger.info(f"Decaying learning rate at step {self.global_step}")
            torch_utils.decay_lr(self.optimizer, gamma=0.5)

    def training_step(self, batch, batch_idx):
        """Training step with Phase 3 GHOP + Phase 4 Contact + Phase 5 Advanced integration."""

        # ============================================================
        # PHASE 5: Dynamic Loss Weight Scheduling
        # ============================================================
        if self.phase5_enabled and hasattr(self, 'phase5_scheduler'):
            loss_weights = self.phase5_scheduler.get_loss_weights(self.global_step)
            lr_multiplier = self.phase5_scheduler.get_learning_rate_multiplier(self.global_step)

            # Apply learning rate adjustment
            for param_group in self.optimizers().param_groups:
                base_lr = param_group.get('initial_lr', param_group['lr'])
                param_group['lr'] = base_lr * lr_multiplier

            # Log dynamic weights
            if self.global_step % self.log_phase5_every == 0:
                self.log('phase5/weight_sds', loss_weights['sds'], prog_bar=False)
                self.log('phase5/weight_contact', loss_weights['contact'], prog_bar=False)
                self.log('phase5/weight_temporal', loss_weights['temporal'], prog_bar=False)
                self.log('phase5/lr_multiplier', lr_multiplier, prog_bar=False)
        else:
            loss_weights = {'sds': 1.0, 'contact': 1.0, 'temporal': 1.0}

        # Existing preprocessing code continues...
        self.condition_training()

        batch["idx"] = torch.stack(batch["idx"], dim=1)
        batch = hold_utils.wubba_lubba_dub_dub(batch)
        batch = xdict(batch)
        batch["current_epoch"] = self.current_epoch
        batch["global_step"] = self.global_step

        for node in self.model.nodes.values():
            params = node.params(batch["idx"])
            batch.update(params)

        debug.debug_params(self)
        model_outputs = self.model(batch)

        # ====================================================================
        # COMPUTE BASE LOSSES
        # ====================================================================
        loss_output = self.loss(batch, model_outputs)
        # ================================================================
        # ADD: First-step diagnostic logging (INSERT AFTER LINE 695)
        # ================================================================
        if self.global_step == 0 and (self.phase3_enabled or self.phase4_enabled or self.phase5_enabled):
            logger.info("=" * 70)
            logger.info("[Diagnostic] Batch structure at step 0:")
            logger.info("=" * 70)

            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    logger.info(
                        f"  {key:30s}: shape={str(value.shape):20s}, "
                        f"dtype={value.dtype}, device={value.device}"
                    )
                elif isinstance(value, dict):
                    logger.info(f"  {key:30s}: dict with {len(value)} keys")
                elif isinstance(value, (list, tuple)):
                    logger.info(f"  {key:30s}: {type(value).__name__} with {len(value)} elements")
                else:
                    logger.info(f"  {key:30s}: {type(value).__name__}")

            logger.info("=" * 70)
            logger.info("[Diagnostic] Searching for MANO parameters (dim 45 or 48):")

            found_mano = []
            for key, value in batch.items():
                if isinstance(value, torch.Tensor) and len(value.shape) > 0:
                    if value.shape[-1] in [45, 48]:
                        found_mano.append((key, value.shape))
                        logger.info(f"  ✓ CANDIDATE: {key:30s} shape={value.shape}")

            if not found_mano:
                logger.warning("  ✗ No MANO parameter candidates found!")

            logger.info("=" * 70)
        # PHASE 3 & PHASE 5: UNIFIED SDS COMPUTATION
        if self.phase3_enabled and self.ghop_enabled:
            # ============================================================
            # INITIALIZE: Default values in case of early exception
            # ============================================================
            ghop_losses = {}
            ghop_info = {
                'stage': 'unknown',
                'stage_progress': 0.0,
                'error': None
            }

            try:
                # ============================================================
                # STEP 1: Extract hand parameters with validation
                # ============================================================
                logger.debug(f"[Phase 3] Extracting hand params at step {self.global_step}")

                hand_params = self._extract_hand_params_from_batch(batch)

                # Validate extraction result
                if hand_params is None or 'pose' not in hand_params:
                    logger.warning(
                        f"[Phase 3] hand_params extraction failed at step {self.global_step}. "
                        f"Result: {type(hand_params)}. Skipping GHOP."
                    )
                    raise ValueError("hand_params extraction failed")

                # Validate pose shape
                hand_pose = hand_params['pose']
                if hand_pose.shape[-1] not in [45, 48]:
                    logger.error(
                        f"[Phase 3] Invalid hand pose shape: {hand_pose.shape}. "
                        f"Expected [..., 45] or [..., 48]. Skipping GHOP."
                    )
                    raise ValueError(f"Invalid hand pose shape: {hand_pose.shape}")

                logger.debug(f"[Phase 3] ✓ Hand params valid: pose={hand_pose.shape}")

                # ============================================================
                # STEP 2: Extract object SDF
                # ============================================================
                object_sdf = self._extract_sdf_grid_from_nodes(batch)

                # Validate SDF
                if object_sdf is None or object_sdf.numel() == 0:
                    logger.warning(
                        f"[Phase 3] object_sdf extraction failed at step {self.global_step}. "
                        f"Shape: {object_sdf.shape if object_sdf is not None else None}. "
                        f"Skipping GHOP."
                    )
                    raise ValueError("object_sdf extraction failed")

                # Check for degenerate SDF (all zeros)
                sdf_std = object_sdf.std()
                if sdf_std < 1e-6:
                    logger.warning(
                        f"[Phase 3] object_sdf is degenerate (std={sdf_std:.6f}). "
                        f"This may indicate initialization issues."
                    )

                logger.debug(
                    f"[Phase 3] ✓ Object SDF valid: shape={object_sdf.shape}, "
                    f"std={sdf_std:.4f}, range=[{object_sdf.min():.4f}, {object_sdf.max():.4f}]"
                )

                # ============================================================
                # STEP 3: Get text prompt
                # ============================================================
                category = batch.get('category', batch.get('object_category', 'Object'))
                if isinstance(category, (list, tuple)):
                    category = category[0]

                text_prompt = f"a hand grasping a {category}"

                # ============================================================
                # STEP 4: Compute SDS loss via ghop_manager
                # ============================================================
                logger.debug(f"[Phase 3] Computing SDS via ghop_manager at step {self.global_step}")

                # ✅ REMOVED: Lines that extracted hand_pose_tensor
                # The manager expects the full hand_params dict

                # Prepare text prompts list
                if isinstance(hand_params, dict):
                    B = hand_params['pose'].shape[0]  # ✅ FIXED: Get batch size from dict
                else:
                    B = hand_params.shape[0]  # ✅ Fallback for tensor input

                text_prompts = [text_prompt] * B

                # Call ghop_manager with correct signature
                ghop_losses, ghop_info = self.ghop_manager.compute_losses(
                    object_sdf=object_sdf,        # [B, 1, 64, 64, 64]
                    hand_params=hand_params,      # ✅ FIXED: Pass dict directly
                    text_prompts=text_prompts,    # List[str]
                    iteration=self.global_step    # int
                )

                # ============================================================
                # STEP 5: Apply Phase 5 dynamic weighting if enabled
                # ============================================================
                if self.phase5_enabled:
                    sds_weight = loss_weights['sds']
                else:
                    sds_weight = 1.0

                # ============================================================
                # STEP 6: Add GHOP losses to total loss
                # ============================================================
                ghop_total = ghop_losses.get('total', 0.0)

                if isinstance(ghop_total, torch.Tensor):
                    weighted_ghop = ghop_total * sds_weight
                else:
                    weighted_ghop = torch.tensor(0.0, device=loss_output['loss'].device)

                loss_output['loss'] = loss_output['loss'] + weighted_ghop
                loss_output['ghop_loss'] = weighted_ghop

                # ============================================================
                # STEP 7: Log GHOP metrics
                # ============================================================
                stage_map = {'sds': 1, 'contact': 2, 'contact_only': 3, 'unknown': 0}
                stage_numeric = stage_map.get(ghop_info.get('stage', 'unknown'), 0)

                self.log('ghop/stage_numeric', float(stage_numeric), prog_bar=False)
                self.log('ghop/stage_progress', ghop_info.get('stage_progress', 0.0), prog_bar=True)
                self.log('ghop/total_loss', weighted_ghop.item() if isinstance(weighted_ghop, torch.Tensor) else float(weighted_ghop), prog_bar=True)

                # Log SDS-specific metrics
                if 'sds' in ghop_losses:
                    sds_value = ghop_losses['sds']
                    if isinstance(sds_value, torch.Tensor):
                        self.log('ghop/sds_loss', sds_value.item(), prog_bar=True)

                # Console logging
                if self.global_step % self.log_ghop_every == 0:
                    logger.info(
                        f"\n[Phase 3 - Step {self.global_step}] GHOP SDS:\n"
                        f"  Stage:            {ghop_info.get('stage', 'unknown')}\n"
                        f"  Stage progress:   {ghop_info.get('stage_progress', 0.0):.3f}\n"
                        f"  Total loss:       {weighted_ghop.item() if isinstance(weighted_ghop, torch.Tensor) else weighted_ghop:.4f}\n"
                        f"  SDS weight:       {sds_weight:.3f}"
                    )

                logger.debug(f"[Phase 3] ✓ GHOP losses added to total loss")

            except ValueError as e:
                # Expected extraction failures - log at debug level
                logger.debug(f"[Phase 3] Skipping GHOP due to extraction issue: {e}")
                ghop_info['error'] = str(e)

                # ✓ ADD: Ensure zero loss is added
                zero_loss = torch.tensor(0.0, device=loss_output['loss'].device, requires_grad=True)
                loss_output['ghop_loss'] = zero_loss

            except Exception as e:
                # Unexpected errors - log with traceback
                logger.error(f"[Phase 3] Unexpected error in GHOP computation: {e}")
                import traceback
                logger.error(traceback.format_exc())
                ghop_info['error'] = str(e)

                # ✓ ADD: Ensure zero loss is added
                zero_loss = torch.tensor(0.0, device=loss_output['loss'].device, requires_grad=True)
                loss_output['ghop_loss'] = zero_loss

        # ====================================================================
        # PHASE 4: Contact Refinement (Enhanced with Phase 5 Adaptive Zones)
        # ====================================================================
        if self.phase4_enabled and self.global_step >= self.contact_start_iter:
            try:
                logger.debug(f"[Phase 4] Contact loss computation at step {self.global_step}")

                # Extract hand mesh
                hand_verts, hand_faces = self._extract_hand_mesh(batch)

                # Extract object mesh from SDF
                obj_verts_list, obj_faces_list = self._extract_object_mesh_from_sdf(batch)

                # ============================================================
                # PHASE 5 ENHANCEMENT: Adaptive Contact Zone Detection
                # ============================================================
                contact_zones = None

                if self.phase5_enabled and hasattr(self, 'adaptive_contacts') and hasattr(self, 'phase5_scheduler'):
                    # Check if update is needed via scheduler
                    if self.phase5_scheduler.should_update_contact_zones(self.global_step):
                        logger.debug(f"[Phase 5] Detecting adaptive contact zones at step {self.global_step}")

                        try:
                            # Detect adaptive contact zones
                            contact_zones = self.adaptive_contacts(
                                hand_verts=hand_verts,
                                obj_verts_list=obj_verts_list,
                                iteration=self.global_step,
                                batch_indices=list(range(hand_verts.shape[0]))
                            )

                            # Log contact statistics
                            if self.global_step % self.log_phase5_every == 0:
                                contact_stats = self.adaptive_contacts.get_contact_statistics(contact_zones)

                                self.log('phase5/contact_mean', contact_stats['mean'], prog_bar=False)
                                self.log('phase5/contact_min', contact_stats['min'], prog_bar=False)
                                self.log('phase5/contact_max', contact_stats['max'], prog_bar=False)
                                self.log('phase5/contact_std', contact_stats['std'], prog_bar=False)

                                logger.info(
                                    f"\n[Phase 5 - Step {self.global_step}] Adaptive Contacts:\n"
                                    f"  Mean contacts:   {contact_stats['mean']:.1f} vertices\n"
                                    f"  Range:           [{contact_stats['min']:.0f}, {contact_stats['max']:.0f}]\n"
                                    f"  Std deviation:   {contact_stats['std']:.2f}"
                                )

                        except Exception as e:
                            logger.warning(f"[Phase 5] Adaptive contact detection failed: {e}. Using fixed zones.")
                            contact_zones = None

                # ============================================================
                # Compute contact loss with adaptive or fixed zones
                # ============================================================
                batch_size = hand_verts.shape[0]
                total_contact_loss = 0.0
                num_valid_samples = 0

                contact_metrics_accum = {
                    'penetration': 0.0,
                    'attraction': 0.0,
                    'dist_mean': 0.0,
                    'num_contacts': 0,
                    'num_penetrations': 0
                }

                for b in range(batch_size):
                    # Get per-sample meshes
                    h_verts = hand_verts[b]  # [778, 3]
                    h_faces = hand_faces if hand_faces.dim() == 2 else hand_faces[b]
                    o_verts = obj_verts_list[b] if isinstance(obj_verts_list, list) else obj_verts_list[b]
                    o_faces = obj_faces_list[b] if isinstance(obj_faces_list, list) else obj_faces_list[b]

                    # Skip if object mesh is empty
                    if o_verts.shape[0] == 0:
                        logger.warning(f"[Phase 4] Empty object mesh for batch {b}, skipping")
                        continue

                    # Get contact zones for this sample
                    zones_b = contact_zones[b] if contact_zones is not None else None

                    # ============================================================
                    # CRITICAL FIX: Check contact_refiner's actual signature
                    # Expected: forward(hand_verts, obj_verts, weightpen, weightmiss)
                    # NOT: forward(hand_verts, hand_faces, obj_verts, obj_faces, contact_zones)
                    # ============================================================
                    try:
                        # Method 1: If contact_refiner has forward with faces
                        if hasattr(self.contact_refiner, 'forward_with_faces'):
                            contact_loss_b, contact_metrics_b = self.contact_refiner.forward_with_faces(
                                hand_verts=h_verts,           # ✓ [778, 3] no unsqueeze
                                hand_faces=h_faces,           # [F, 3]
                                obj_verts=o_verts,            # ✓ [V_obj, 3] no unsqueeze
                                obj_faces=o_faces,            # [F_obj, 3]
                                contact_zones=zones_b         # [K] or None
                            )
                        # Method 2: Standard forward (vertex-only, no faces)
                        else:
                            contact_loss_b, contact_metrics_b = self.contact_refiner(
                                hand_verts=h_verts,           # [778, 3]
                                obj_verts=o_verts,            # [V_obj, 3]
                                weightpen=100.0,
                                weightmiss=10.0
                            )

                        total_contact_loss += contact_loss_b
                        num_valid_samples += 1

                        # Accumulate metrics
                        for key in contact_metrics_accum:
                            if key in contact_metrics_b:
                                contact_metrics_accum[key] += contact_metrics_b[key]

                    except Exception as e:
                        logger.error(f"[Phase 4] Contact refiner failed for batch {b}: {e}")
                        continue
                # Average and apply weights
                if num_valid_samples > 0:
                    total_contact_loss /= num_valid_samples
                    for key in contact_metrics_accum:
                        contact_metrics_accum[key] /= num_valid_samples

                    # Progressive weight schedule (warmup from 0 to w_contact)
                    contact_progress = min(
                        (self.global_step - self.contact_start_iter) / self.contact_warmup_iters,
                        1.0
                    )

                    # Apply Phase 5 dynamic weighting if enabled
                    if self.phase5_enabled:
                        base_weight = self.w_contact * contact_progress
                        weighted_contact_loss = total_contact_loss * base_weight * loss_weights['contact']
                    else:
                        weighted_contact_loss = total_contact_loss * self.w_contact * contact_progress

                    # Add to total loss
                    loss_output["loss"] = loss_output["loss"] + weighted_contact_loss
                    loss_output['contact_loss'] = weighted_contact_loss

                    # Log metrics
                    self.log('phase4/contact_loss', weighted_contact_loss, prog_bar=True)
                    self.log('phase4/contact_weight',
                            base_weight * loss_weights['contact'] if self.phase5_enabled else self.w_contact * contact_progress)
                    self.log('phase4/penetration', contact_metrics_accum['penetration'])
                    self.log('phase4/attraction', contact_metrics_accum['attraction'])
                    self.log('phase4/dist_mean', contact_metrics_accum['dist_mean'])
                    self.log('phase4/num_contacts', float(contact_metrics_accum['num_contacts']))
                    self.log('phase4/num_penetrations', float(contact_metrics_accum['num_penetrations']))

                    # Console logging
                    if self.global_step % self.log_contact_every == 0:
                        contact_type = "Adaptive" if contact_zones is not None else "Fixed"
                        logger.info(
                            f"\n[Phase 4 - Step {self.global_step}] Contact Refinement ({contact_type}):\n"
                            f"  Valid samples:    {num_valid_samples}/{batch_size}\n"
                            f"  Penetration loss: {contact_metrics_accum['penetration']:.4f}\n"
                            f"  Attraction loss:  {contact_metrics_accum['attraction']:.4f}\n"
                            f"  Mean distance:    {contact_metrics_accum['dist_mean']:.4f}m\n"
                            f"  Num contacts:     {int(contact_metrics_accum['num_contacts'])}\n"
                            f"  Num penetrations: {int(contact_metrics_accum['num_penetrations'])}\n"
                            f"  Total loss:       {weighted_contact_loss:.4f}\n"
                        )
                else:
                    logger.warning(
                        f"[Phase 4] No valid samples at step {self.global_step}, "
                        f"skipping contact loss"
                    )

            except Exception as e:
                logger.error(f"[Phase 4] Contact loss computation failed: {e}")
                import traceback
                traceback.print_exc()

        # ====================================================================
        # PHASE 2: Backward Compatibility
        # ====================================================================
        if self.phase2_enabled and self.hold_loss_module is not None:
            try:
                object_node = None

                # ============================================================
                # FIX: Use consistent parameter extraction
                # BEFORE: hand_pose_params = self._extract_hand_params_from_batch(batch)
                # AFTER: Use same dict format as Phase 3
                # ============================================================
                hand_params = self._extract_hand_params_from_batch(batch)

                for node in self.model.nodes.values():
                    if "object" in node.node_id.lower():
                        object_node = node
                        break

                # Check if we got valid params
                if object_node is not None and hand_params and 'pose' in hand_params:
                    category = batch.get('object_category', batch.get('category', 'object'))
                    if isinstance(category, (list, tuple)):
                        category = category[0]

                    # Use the 'pose' key from dict
                    loss_sds, sds_info = self.hold_loss_module.compute_sds_loss(
                        object_node=object_node,
                        hand_pose=hand_params['pose'],  # ✓ Extract pose from dict
                        category=category,
                        iteration=self.global_step
                    )

                    loss_output["loss"] = loss_output["loss"] + loss_sds
                    loss_output["sds_loss"] = loss_sds

                    if loss_sds.item() > 0:
                        self.log('train/sds_loss', loss_sds, prog_bar=True)

            except Exception as e:
                logger.error(f"[Phase 2] SDS loss failed: {e}")

        # ====================================================================
        # PHASE 5: TEMPORAL CONSISTENCY FOR VIDEO SEQUENCES
        # COMPLETE IMPLEMENTATION WITH ERROR HANDLING
        # ====================================================================
        if self.phase5_enabled and self.global_step >= self.phase5_start_iter:
            try:
                # =============================================================
                # Step 1: Check if this is a video batch with consecutive frames
                # =============================================================
                if self._is_video_batch(batch):
                    logger.debug(f"[Phase 5] Computing temporal consistency at step {self.global_step}")

                    # =============================================================
                    # Step 2: Extract predicted hand pose for current frame
                    # =============================================================
                    frame_idx = batch.get('frame_idx', batch.get('temporal_idx', None))

                    if frame_idx is None:
                        logger.debug(
                            "[Phase 5] No frame_idx in batch. Using sequential ordering assumption."
                        )
                        # Fallback: assume sequential frames (idx, idx+1)
                        frame_idx = batch.get('idx', torch.arange(batch['hA'].shape[0]))

                    # Extract predicted pose
                    predicted_pose = self._extract_predicted_hand_pose(batch, model_outputs)

                    # Verify predicted_pose shape
                    if predicted_pose is None or predicted_pose.shape[-1] not in [45, 48]:
                        logger.warning(
                            f"[Phase 5] Invalid predicted_pose shape: {predicted_pose.shape if predicted_pose is not None else None}. "
                            f"Expected [..., 45] or [..., 48]. Skipping temporal consistency."
                        )
                        raise ValueError("Invalid predicted_pose shape")

                    # =============================================================
                    # Step 3: Verify batch contains required fields from hoi.py
                    # =============================================================
                    required_fields = ['hA_n', 'c2w', 'c2w_n']
                    missing_fields = [f for f in required_fields if f not in batch]

                    if missing_fields:
                        logger.debug(
                            f"[Phase 5] Batch missing temporal fields: {missing_fields}. "
                            f"Skipping temporal consistency."
                        )
                        raise ValueError(f"Missing temporal fields: {missing_fields}")

                    # =============================================================
                    # Step 4: Get sequence ID for history tracking
                    # =============================================================
                    sequence_id = batch.get('video_id', batch.get('sequence_id', batch.get('scene_id', 'default')))

                    # Convert to string if tensor
                    if isinstance(sequence_id, torch.Tensor):
                        sequence_id = str(sequence_id.item())
                    elif isinstance(sequence_id, (list, tuple)):
                        sequence_id = str(sequence_id[0])

                    # =============================================================
                    # Step 5: Compute temporal consistency losses
                    # =============================================================
                    temporal_loss, temporal_metrics = self.temporal_module(
                        sample=batch,  # Contains hA, hA_n, c2w, c2w_n from hoi.py
                        predicted_hand_pose=predicted_pose,
                        sequence_id=sequence_id,
                        frame_idx=frame_idx  # ✓ Added frame indices
                    )

                    # =============================================================
                    # Step 6: Apply Phase 5 dynamic weighting
                    # =============================================================
                    weighted_temporal = temporal_loss * loss_weights['temporal'] * self.w_temporal

                    # =============================================================
                    # Step 7: Add to total loss
                    # =============================================================
                    loss_output['loss'] = loss_output['loss'] + weighted_temporal
                    loss_output['temporal_loss'] = weighted_temporal

                    # =============================================================
                    # Step 8: Log temporal metrics
                    # =============================================================
                    self.log('phase5/temporal_loss', weighted_temporal.item(), prog_bar=True)
                    self.log('phase5/velocity_loss', temporal_metrics.get('velocity', 0.0), prog_bar=False)
                    self.log('phase5/acceleration_loss', temporal_metrics.get('acceleration', 0.0), prog_bar=False)
                    self.log('phase5/camera_motion_loss', temporal_metrics.get('camera_motion', 0.0), prog_bar=False)
                    self.log('phase5/temporal_adaptive_weight', temporal_metrics.get('adaptive_weight', 1.0), prog_bar=False)

                    # Console logging at reduced frequency
                    if self.global_step % self.log_phase5_every == 0:
                        logger.info(
                            f"\n[Phase 5 - Step {self.global_step}] Temporal Consistency:\n"
                            f"  Sequence ID:      {sequence_id}\n"
                            f"  Total loss:       {weighted_temporal.item():.4f}\n"
                            f"  Velocity:         {temporal_metrics.get('velocity', 0.0):.4f}\n"
                            f"  Acceleration:     {temporal_metrics.get('acceleration', 0.0):.4f}\n"
                            f"  Camera motion:    {temporal_metrics.get('camera_motion', 0.0):.4f}\n"
                            f"  Adaptive weight:  {temporal_metrics.get('adaptive_weight', 1.0):.3f}"
                        )

            except ValueError as e:
                # Expected errors (missing fields, wrong shapes) - log at debug level
                logger.debug(f"[Phase 5] Temporal consistency skipped: {e}")

            except Exception as e:
                # Unexpected errors - log at warning level with traceback
                logger.warning(f"[Phase 5] Temporal consistency computation failed: {e}")
                import traceback
                traceback.print_exc()

        # Logging
        if self.global_step % self.args.log_every == 0:
            self.metrics(model_outputs, batch, self.global_step, self.current_epoch)
            comet_utils.log_dict(
                self.args.experiment,
                loss_output,
                step=self.global_step,
                epoch=self.current_epoch,
            )

        return loss_output['loss']

    # ====================================================================
    # HELPER METHODS
    # ====================================================================

    def _extract_sdf_grid_from_nodes(self, batch, resolution=64):
        """Extract SDF values on regular grid from object node."""
        # Step 1: Determine batch size
        if 'idx' in batch:
            B = batch['idx'].shape[0]
        else:
            for node in self.model.nodes.values():
                if "hand" in node.node_id.lower() or "right" in node.node_id.lower():
                    mano_pose_key = f"{node.node_id}.mano_pose"
                    if mano_pose_key in batch:
                        B = batch[mano_pose_key].shape[0]
                        break
            else:
                B = 1

        H = resolution
        device = next(self.model.parameters()).device

        # Step 2: Create coordinate grid in canonical space [-1.5, 1.5]³
        x = torch.linspace(-1.5, 1.5, H, device=device)
        try:
            grid = torch.stack(torch.meshgrid(x, x, x, indexing='ij'), dim=-1)
        except TypeError:
            grid = torch.stack(torch.meshgrid(x, x, x), dim=-1)
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1, -1)  # (B, H, H, H, 3)
        # Step 3: Find object node
        object_node = None
        for node in self.model.nodes.values():
            if "object" in node.node_id.lower():
                object_node = node
                break
        if object_node is None:
            logger.warning("[Helper] No object node found, returning zero SDF")
            return torch.zeros(B, 1, resolution, resolution, resolution, device=device)
        grid_flat = grid.reshape(B, -1, 3)
        # Step 4: Try multiple SDF extraction methods
        with torch.no_grad():
            sdf_values = None

            # METHOD 1: server.forward_sdf (preferred)
            if hasattr(object_node, "server") and hasattr(object_node.server, "forward_sdf"):
                try:
                    sdf_output = object_node.server.forward_sdf(grid_flat)
                    if isinstance(sdf_output, dict) and 'sdf' in sdf_output:
                        sdf_values = sdf_output['sdf']
                    elif isinstance(sdf_output, torch.Tensor):
                        sdf_values = sdf_output
                    logger.debug(f"[Helper] METHOD 1 SUCCESS: server.forward_sdf: {sdf_values.shape}")
                except Exception as e:
                    logger.debug(f"[Helper] METHOD 1 FAILED: server.forward_sdf: {e}")

            # METHOD 2: server.shape_net (alternative)
            if sdf_values is None and hasattr(object_node, "server") and hasattr(object_node.server, "shape_net"):
                try:
                    latent_code = getattr(object_node.server, 'latent_code', None)
                    if latent_code is not None:
                        sdf_output = object_node.server.shape_net(grid_flat, latent_code)
                        if isinstance(sdf_output, dict) and 'sdf' in sdf_output:
                            sdf_values = sdf_output['sdf']
                        elif isinstance(sdf_output, torch.Tensor):
                            sdf_values = sdf_output
                        logger.debug(f"[Helper] METHOD 2 SUCCESS: server.shape_net: {sdf_values.shape}")
                except Exception as e:
                    logger.debug(f"[Helper] METHOD 2 FAILED: server.shape_net: {e}")

            # METHOD 3: node.forward (requires proper inputs)
            if sdf_values is None and hasattr(object_node, "forward"):
                try:
                    # Build proper forward call with points
                    forward_input = {
                        'points': grid_flat,  # [B, H³, 3]
                        'indices': batch.get('idx', torch.zeros(B, dtype=torch.long, device=device))
                    }

                    node_output = object_node(**forward_input)

                    if isinstance(node_output, dict):
                        if 'sdf' in node_output:
                            sdf_values = node_output['sdf']
                        elif 'geometry' in node_output:
                            sdf_values = node_output['geometry']
                    elif isinstance(node_output, torch.Tensor):
                        sdf_values = node_output

                    if sdf_values is not None:
                        logger.debug(f"[Helper] METHOD 3 SUCCESS: node.forward: {sdf_values.shape}")
                except Exception as e:
                    logger.debug(f"[Helper] METHOD 3 FAILED: node.forward: {e}")

            # ============================================================
            # METHOD 4: Query via model's render_core (CORRECTED - FINAL)
            # ============================================================
            if sdf_values is None and hasattr(object_node, "implicit_network"):
                try:
                    logger.debug("[Helper] Attempting METHOD 4: model render_core query")

                    # HOLD's model has a unified way to query SDF values
                    # We'll use the model's forward method properly

                    # Get batch indices
                    idx = batch.get('idx', torch.zeros(B, dtype=torch.long, device=device))

                    # Query SDF for all grid points using model's interface
                    # The model internally handles implicit network calling
                    num_points = grid_flat.shape[1]  # H³

                    # Call model's implicit network through proper interface
                    # HOLD expects: (points, indices, ...)
                    model_input = {
                        'points': grid_flat,  # [B, H³, 3]
                        'indices': idx.unsqueeze(1).expand(-1, num_points),  # [B, H³]
                    }

                    # Forward through model
                    with torch.no_grad():
                        # Use object node's implicit network if available
                        if hasattr(object_node, 'implicit_network'):
                            # Get feature vector for this sample
                            if hasattr(object_node, 'embedding'):
                                features = object_node.embedding(idx)  # [B, feature_dim]
                            elif hasattr(object_node, 'feature_vector'):
                                features = object_node.feature_vector.weight[idx]  # [B, feature_dim]
                            else:
                                # Use zero features
                                features = None

                            # Query implicit network point-by-point to avoid shape issues
                            sdf_list = []
                            for b in range(B):
                                # Get points for this batch sample
                                points_b = grid_flat[b]  # [H³, 3]

                                # Call implicit network with correct interface
                                if features is not None:
                                    # Concatenate features to each point
                                    feat_b = features[b].unsqueeze(0).expand(points_b.shape[0], -1)  # [H³, feat_dim]
                                    input_b = torch.cat([points_b, feat_b], dim=-1)  # [H³, 3+feat_dim]
                                else:
                                    input_b = points_b  # [H³, 3]

                                # Forward (assuming implicit_network expects single input)
                                try:
                                    output_b = object_node.implicit_network(input_b.unsqueeze(0))  # [1, H³, ...]

                                    # Extract SDF
                                    if isinstance(output_b, dict):
                                        sdf_b = output_b.get('sdf', output_b.get('model_out', output_b.get('output')))
                                    else:
                                        sdf_b = output_b

                                    # Ensure shape [1, H³, 1]
                                    if sdf_b.dim() == 2:
                                        sdf_b = sdf_b.unsqueeze(-1)
                                    if sdf_b.shape[-1] != 1:
                                        sdf_b = sdf_b[..., :1]

                                    sdf_list.append(sdf_b)
                                except Exception as e_inner:
                                    logger.debug(f"[Helper] Batch {b} failed: {e_inner}, using zeros")
                                    sdf_list.append(torch.zeros(1, points_b.shape[0], 1, device=device))

                            # Stack results
                            if len(sdf_list) == B:
                                sdf_values = torch.cat(sdf_list, dim=0)  # [B, H³, 1]
                                logger.debug(f"[Helper] METHOD 4 SUCCESS: Per-batch query: {sdf_values.shape}")

                except Exception as e:
                    logger.debug(f"[Helper] METHOD 4 FAILED: render_core: {e}")
                    import traceback
                    logger.debug(f"[Helper] Traceback: {traceback.format_exc()}")

            # ============================================================
            # FALLBACK: Return zero grid with clear warning
            # ============================================================
            if sdf_values is None:
                logger.warning(
                    "[Helper] All SDF extraction methods failed, using zero grid. "
                    "This is expected in early training before object geometry is initialized. "
                    f"Attempted methods: forward_sdf, shape_net, forward, implicit_network on node '{object_node.node_id}'"
                )
                return torch.zeros(B, 1, resolution, resolution, resolution, device=device)

        # Step 5: Reshape to (B, 1, H, H, H) format
        # Ensure correct shape [B, H^3, 1], then gridify
        if sdf_values.dim() == 2:
            sdf_values = sdf_values.unsqueeze(-1)
        if sdf_values.shape[-1] != 1:
            sdf_values = sdf_values[..., :1]  # Only first channel if needed
        object_sdf = sdf_values.reshape(B, resolution, resolution, resolution, 1).permute(0, 4, 1, 2, 3)

        # ================================================================
        # Step 6: Validate extracted SDF (warn if degenerate)
        # ================================================================
        sdf_std = object_sdf.std()
        sdf_mean = object_sdf.mean()

        if sdf_std < 1e-6:
            logger.debug(
                f"[Helper] Degenerate SDF detected (std={sdf_std:.6f}, mean={sdf_mean:.6f}). "
                f"Object geometry may not be initialized yet."
            )
        else:
            logger.debug(
                f"[Helper] Valid SDF extracted: shape={object_sdf.shape}, "
                f"std={sdf_std:.4f}, mean={sdf_mean:.4f}, "
                f"range=[{object_sdf.min():.4f}, {object_sdf.max():.4f}]"
            )

        return object_sdf

    # ====================================================================
    # PHASE 4: Mesh Extraction Helper Methods
    # ====================================================================

    def _extract_hand_mesh(self, batch):
        """
        Extract hand mesh from MANO parameters in batch.

        Args:
            batch: Training batch containing hand parameters

        Returns:
            hand_verts: [B, 778, 3] hand mesh vertices
            hand_faces: [1538, 3] hand mesh faces (MANO topology)
        """
        # ================================================================
        # STEP 1: Find hand node
        # ================================================================
        hand_node = None
        node_id = None

        for node in self.model.nodes.values():
            if 'right' in node.node_id.lower() or 'left' in node.node_id.lower():
                hand_node = node
                node_id = node.node_id
                break

        if hand_node is None:
            raise ValueError("[Phase 4] No hand node found in model")

        logger.debug(f"[Phase 4] Found hand node: {node_id}")

        # ================================================================
        # STEP 2: Extract MANO parameters using CORRECT key names
        # ================================================================

        fullpose_key = f"{node_id}.fullpose"
        pose_key = f"{node_id}.pose"
        global_orient_key = f"{node_id}.global_orient"
        shape_key = f"{node_id}.betas"
        trans_key = f"{node_id}.transl"

        # ================================================================
        # Extract pose (48-dim full pose)
        # ================================================================
        full_pose = None

        if fullpose_key in batch:
            full_pose = batch[fullpose_key]  # [B, 48]
            logger.debug(f"[Phase 4] Using {fullpose_key}: {full_pose.shape}")

        elif global_orient_key in batch and pose_key in batch:
            global_orient = batch[global_orient_key]  # [B, 3]
            pose = batch[pose_key]                     # [B, 45]
            full_pose = torch.cat([global_orient, pose], dim=-1)  # [B, 48]
            logger.debug(f"[Phase 4] Concatenated {global_orient_key} + {pose_key}: {full_pose.shape}")

        elif pose_key in batch:
            pose = batch[pose_key]  # [B, 45]
            batch_size = pose.shape[0]
            device = pose.device
            zero_global = torch.zeros(batch_size, 3, device=device)
            full_pose = torch.cat([zero_global, pose], dim=-1)  # [B, 48]
            logger.warning(f"[Phase 4] No global_orient found, using zeros")

        else:
            available_keys = [k for k in batch.keys() if node_id in k]
            raise ValueError(
                f"[Phase 4] Cannot find MANO pose in batch.\n"
                f"  Tried keys: {fullpose_key}, {pose_key}, {global_orient_key}\n"
                f"  Available keys for '{node_id}': {available_keys}"
            )

        # ================================================================
        # Extract shape parameters (betas)
        # ================================================================
        mano_shape = batch.get(shape_key, None)

        if mano_shape is None:
            batch_size = full_pose.shape[0]
            device = full_pose.device
            mano_shape = torch.zeros(batch_size, 10, device=device)
            logger.warning(f"[Phase 4] No shape parameters at {shape_key}, using mean shape")
        else:
            logger.debug(f"[Phase 4] Using {shape_key}: {mano_shape.shape}")

        # ================================================================
        # CRITICAL FIX: Extract translation (was missing!)
        # ================================================================
        mano_trans = batch.get(trans_key, None)

        if mano_trans is None:
            batch_size = full_pose.shape[0]
            device = full_pose.device
            mano_trans = torch.zeros(batch_size, 3, device=device)
            logger.warning(f"[Phase 4] No translation at {trans_key}, using zero translation")
        else:
            logger.debug(f"[Phase 4] Using {trans_key}: {mano_trans.shape}")

        # ================================================================
        # CRITICAL FIX: Extract scene_scale (required by MANOServer)
        # ================================================================
        # MANOServer.forward expects: forward(scene_scale, transl, thetas, betas)

        # Try to get scene scale from batch or node
        scene_scale = None

        # Option 1: From batch
        scale_keys = [
            f"{node_id}.scene_scale",
            "scene_scale",
            f"{node_id}.scale"
        ]

        for scale_key in scale_keys:
            if scale_key in batch:
                scene_scale = batch[scale_key]
                logger.debug(f"[Phase 4] Using {scale_key}: {scene_scale.shape}")
                break

        # Option 2: From node attributes
        if scene_scale is None and hasattr(hand_node, 'scene_scale'):
            scene_scale = hand_node.scene_scale
            logger.debug(f"[Phase 4] Using hand_node.scene_scale: {scene_scale}")

        # Option 3: Default to 1.0 (no scaling)
        if scene_scale is None:
            batch_size = full_pose.shape[0]
            device = full_pose.device
            scene_scale = torch.ones(batch_size, device=device)
            logger.warning("[Phase 4] No scene_scale found, using 1.0")

        # Ensure scene_scale is correct shape
        if scene_scale.dim() == 0:
            # Scalar: expand to batch size
            batch_size = full_pose.shape[0]
            scene_scale = scene_scale.unsqueeze(0).expand(batch_size)
        elif scene_scale.shape[0] != full_pose.shape[0]:
            # Wrong batch size: repeat or broadcast
            batch_size = full_pose.shape[0]
            scene_scale = scene_scale.view(-1)[0].unsqueeze(0).expand(batch_size)

        # ================================================================
        # STEP 3: Forward through MANO to generate mesh
        # ================================================================
        with torch.no_grad():
            mano_server = hand_node.server

            logger.debug(
                f"[Phase 4] Calling MANO with:\n"
                f"  scene_scale: {scene_scale.shape}\n"
                f"  transl: {mano_trans.shape}\n"
                f"  thetas: {full_pose.shape}\n"
                f"  betas: {mano_shape.shape}"
            )

            # ================================================================
            # Call MANOServer with CORRECT argument order:
            # forward(scene_scale, transl, thetas, betas, absolute=False)
            # ================================================================
            try:
                mano_output = mano_server(
                    scene_scale,    # arg1: [B] scene scaling
                    mano_trans,     # arg2: [B, 3] translation
                    full_pose,      # arg3: [B, 48] full pose
                    mano_shape,     # arg4: [B, 10] shape
                    absolute=False  # Use relative pose (default)
                )
                logger.debug("[Phase 4] MANO server called successfully")

            except Exception as e:
                logger.error(f"[Phase 4] MANO server call failed: {e}")
                import traceback
                traceback.print_exc()
                raise

            # Extract vertices
            if isinstance(mano_output, dict):
                hand_verts = mano_output['verts']  # [B, 778, 3]
            else:
                hand_verts = mano_output  # [B, 778, 3]

            # ================================================================
            # Get faces and handle dtype conversion
            # ================================================================
            hand_faces = mano_server.faces  # Might be numpy array or tensor

            # Ensure it's a PyTorch tensor with compatible dtype
            if not isinstance(hand_faces, torch.Tensor):
                import numpy as np

                # Step 1: Convert to numpy if needed
                if not isinstance(hand_faces, np.ndarray):
                    hand_faces = np.array(hand_faces)

                # Step 2: Handle uint32 dtype (not supported by PyTorch)
                if hand_faces.dtype in [np.uint32, np.uint64]:
                    # Convert to int32 or int64 (PyTorch compatible)
                    hand_faces = hand_faces.astype(np.int32)
                    logger.debug(f"[Phase 4] Converted faces from {hand_faces.dtype} to int32")

                # Step 3: Convert to PyTorch tensor
                hand_faces = torch.from_numpy(hand_faces).long()

            # Ensure faces are on same device as vertices
            if hand_faces.device != hand_verts.device:
                hand_faces = hand_faces.to(hand_verts.device)

        logger.debug(
            f"[Phase 4] Extracted hand mesh:\n"
            f"  Vertices: {hand_verts.shape} (range: [{hand_verts.min():.3f}, {hand_verts.max():.3f}])\n"
            f"  Faces: {hand_faces.shape}, dtype: {hand_faces.dtype}, device: {hand_faces.device}"
        )

        return hand_verts, hand_faces

    def _extract_object_mesh_from_sdf(self, batch):
        """Extract object mesh from implicit SDF via Marching Cubes."""
        from skimage import measure

        # Extract SDF grid
        resolution = self.mesh_resolution
        object_sdf = self._extract_sdf_grid_from_nodes(batch, resolution=resolution)

        batch_size = object_sdf.shape[0]
        obj_verts_list = []
        obj_faces_list = []

        for b in range(batch_size):
            sdf_grid = object_sdf[b, 0].cpu().numpy()  # [H, H, H]

            try:
                # Apply Marching Cubes
                verts, faces, normals, values = measure.marching_cubes(
                    sdf_grid,
                    level=0.0,
                    spacing=(3.0 / resolution, 3.0 / resolution, 3.0 / resolution)
                )

                # Shift to [-1.5, 1.5] coordinate system
                verts = verts - 1.5

                # Convert to tensors
                obj_verts_list.append(torch.from_numpy(verts).float().to(object_sdf.device))
                obj_faces_list.append(torch.from_numpy(faces).long().to(object_sdf.device))

                logger.debug(f"[Phase 4] Extracted object mesh {b}: {verts.shape[0]} verts, {faces.shape[0]} faces")

            except Exception as e:
                logger.warning(f"[Phase 4] Marching Cubes failed for batch {b}: {e}")
                # Fallback to empty mesh
                obj_verts_list.append(torch.zeros((0, 3), device=object_sdf.device))
                obj_faces_list.append(torch.zeros((0, 3), dtype=torch.long, device=object_sdf.device))

        return obj_verts_list, obj_faces_list
    # ====================================================================

    def _is_video_batch(self, batch: Dict) -> bool:
        """
        Check if batch contains video sequence data from hoi.py.

        Verifies presence of temporal fields required for consecutive frame pairs:
        - 'hA_n': Next frame hand pose
        - 'c2w_n': Next frame camera pose
        - Sequence identifier: 'video_id', 'sequence_id', or 'scene_id'

        Args:
            batch: Training batch dictionary

        Returns:
            bool: True if batch contains video sequences with frame pairs
        """
        # Check for next frame fields (from hoi.py Lines 204-206)
        has_next_hand = 'hA_n' in batch
        has_next_camera = 'c2w_n' in batch

        # Check for sequence identifier
        has_sequence_id = any(k in batch for k in ['video_id', 'sequence_id', 'scene_id'])

        # Also check for temporal index if available
        has_temporal_idx = 'frame_idx' in batch or 'temporal_idx' in batch

        is_video = has_next_hand and has_next_camera and (has_sequence_id or has_temporal_idx)

        if not is_video:
            logger.debug(
                f"[Phase 5] Batch is not video sequence: "
                f"has_next_hand={has_next_hand}, has_next_camera={has_next_camera}, "
                f"has_sequence_id={has_sequence_id}, has_temporal_idx={has_temporal_idx}"
            )

        return is_video


    def _extract_predicted_hand_pose(
        self,
        batch: Dict,
        model_outputs: Dict
    ) -> torch.Tensor:
        """
        Extract predicted hand pose from model outputs for temporal consistency.

        CRITICAL: Returns [B, 45] (articulation only) for TemporalConsistencyModule.
        Strips global_orient if present (first 3 dims of 48-dim MANO pose).

        Args:
            batch: Training batch dictionary
            model_outputs: Model forward pass outputs

        Returns:
            hand_pose: [B, 45] predicted hand articulation parameters
                       (global_orient stripped for temporal consistency)
        """
        hand_pose = None

        # Method 1: Extract from model node parameters (primary method)
        for node_id, node in self.model.nodes.items():
            if 'right' in node_id.lower() or 'left' in node_id.lower():
                # Get predicted pose parameters from node
                params = node.params(batch['idx'])  # [B, 48] or [B, 51]

                # Extract pose and convert to [B, 45] format
                if params.shape[-1] >= 48:
                    # Params are [B, 48+]: Take first 48, then strip global_orient
                    full_pose = params[..., :48]  # [B, 48]
                    hand_pose = full_pose[..., 3:]  # [B, 45] - strip first 3 dims
                elif params.shape[-1] == 45:
                    # Already 45-dim (articulation only)
                    hand_pose = params
                else:
                    # Unexpected shape
                    logger.warning(
                        f"[Phase 5] Unexpected param shape from node '{node_id}': "
                        f"{params.shape}. Expected [..., 45] or [..., 48+]"
                    )
                    hand_pose = params

                logger.debug(
                    f"[Phase 5] Extracted hand pose from node '{node_id}': "
                    f"original_shape={params.shape}, output_shape={hand_pose.shape}"
                )
                break  # Use first hand node found

        # ================================================================
        # Method 2: Fallback to batch data if node extraction fails
        # ================================================================
        if hand_pose is None:
            if 'hA' in batch:
                # hoi.py dataset provides [B, 45] articulation
                hand_pose = batch['hA']  # [B, 45]
                logger.debug(
                    f"[Phase 5] Using batch 'hA' as hand pose fallback: "
                    f"shape={hand_pose.shape}"
                )

            elif 'right.fullpose' in batch:
                # MANO fullpose [B, 48]: strip global_orient
                full_pose = batch['right.fullpose']  # [B, 48]
                hand_pose = full_pose[..., 3:]  # [B, 45]
                logger.debug(
                    f"[Phase 5] Using batch 'right.fullpose', stripped to [B, 45]: "
                    f"shape={hand_pose.shape}"
                )

            elif 'right.pose' in batch:
                # Already articulation only [B, 45]
                hand_pose = batch['right.pose']  # [B, 45]
                logger.debug(
                    f"[Phase 5] Using batch 'right.pose': shape={hand_pose.shape}"
                )

            else:
                logger.warning(
                    "[Phase 5] Could not extract hand pose from model or batch. "
                    "Temporal consistency will be skipped for this batch."
                )
                return None

        # ================================================================
        # Validation: Ensure [B, 45] format
        # ================================================================
        # Ensure batch dimension
        if hand_pose.ndim == 1:
            hand_pose = hand_pose.unsqueeze(0)

        # Final shape check and correction
        if hand_pose.shape[-1] == 48:
            # Still 48-dim: strip global_orient
            hand_pose = hand_pose[..., 3:]  # [B, 45]
            logger.debug("[Phase 5] Stripped global_orient: final shape [B, 45]")

        elif hand_pose.shape[-1] > 48:
            # More than 48 dims (includes shape/trans): take first 48, then strip
            hand_pose = hand_pose[..., 3:48]  # [B, 45]
            logger.debug("[Phase 5] Extracted dims 3-48 from extended params: [B, 45]")

        elif hand_pose.shape[-1] != 45:
            # Unexpected dimension
            raise ValueError(
                f"[Phase 5] Invalid hand pose shape: {hand_pose.shape}. "
                f"After processing, expected [B, 45], got [B, {hand_pose.shape[-1]}]. "
                f"TemporalConsistencyModule requires exactly 45 dimensions (articulation only)."
            )

        logger.debug(f"[Phase 5] Final hand pose shape: {hand_pose.shape} (validated [B, 45])")

        return hand_pose


    def _extract_hand_params_from_batch(self, batch: Dict) -> Dict:
        """
        Extract hand parameters dictionary for GHOP modules.

        Updated logic to handle HOLD's 62-dim composite params by:
        1. Prioritizing decomposed fields (right.full_pose, right.pose)
        2. Extracting from composite right.params as fallback
        3. Handling HOLD's specific parameter structure

        Args:
            batch: Training batch dictionary

        Returns:
            hand_params: Dictionary with keys:
                - 'pose': [B, 45] or [B, 48] MANO parameters
                - 'shape': [B, 10] MANO shape parameters (beta)
                - 'trans': [B, 3] global translation

        Raises:
            ValueError: If hand parameters cannot be extracted
        """
        # ================================================================
        # STEP 1: Try decomposed fields FIRST (highest priority)
        # ================================================================
        hand_pose = None
        source_key = None

        # Priority 1: Try 'right.full_pose' (48-dim: 3 global + 45 joints)
        if 'right.full_pose' in batch:
            hand_pose = batch['right.full_pose']
            source_key = 'right.full_pose'
            logger.debug(f"[_extract_hand_params] Found right.full_pose: {hand_pose.shape}")

        # Priority 2: Try 'right.pose' (45-dim: joint angles only)
        elif 'right.pose' in batch:
            hand_pose = batch['right.pose']
            source_key = 'right.pose'
            logger.debug(f"[_extract_hand_params] Found right.pose: {hand_pose.shape}")

        # Priority 3: Try 'left.full_pose'
        elif 'left.full_pose' in batch:
            hand_pose = batch['left.full_pose']
            source_key = 'left.full_pose'
            logger.debug(f"[_extract_hand_params] Found left.full_pose: {hand_pose.shape}")

        # Priority 4: Try 'left.pose'
        elif 'left.pose' in batch:
            hand_pose = batch['left.pose']
            source_key = 'left.pose'
            logger.debug(f"[_extract_hand_params] Found left.pose: {hand_pose.shape}")

        # ================================================================
        # STEP 2: Try composite 'right.params' (62-dim or 58-dim)
        # ================================================================
        elif 'right.params' in batch:
            params_full = batch['right.params']
            logger.debug(f"[_extract_hand_params] Found right.params: {params_full.shape}")

            # HOLD format: [0:48] or [0:45] = pose, [48:58] = shape, [58:62] = extra
            dim = params_full.shape[-1]

            if dim == 62:
                # Extract first 48 dims (global + joints)
                hand_pose = params_full[..., :48]
                source_key = 'right.params[0:48]'
                logger.debug(f"[_extract_hand_params] Extracted pose from 62-dim params: {hand_pose.shape}")

            elif dim == 58:
                # Extract first 45 dims (joints only)
                hand_pose = params_full[..., :45]
                source_key = 'right.params[0:45]'
                logger.debug(f"[_extract_hand_params] Extracted pose from 58-dim params: {hand_pose.shape}")

            else:
                logger.warning(
                    f"[_extract_hand_params] Unexpected right.params dimension: {dim}. "
                    f"Expected 58 or 62. Attempting extraction anyway..."
                )
                # Try to extract first 48 or 45
                if dim >= 48:
                    hand_pose = params_full[..., :48]
                    source_key = f'right.params[0:48]'
                elif dim >= 45:
                    hand_pose = params_full[..., :45]
                    source_key = f'right.params[0:45]'
                else:
                    logger.error(f"[_extract_hand_params] right.params too small: {dim} < 45")
                    hand_pose = None

        # ================================================================
        # STEP 3: Try 'hA' (HOLD convention)
        # ================================================================
        elif 'hA' in batch:
            hand_pose = batch['hA']
            source_key = 'hA'
            logger.debug(f"[_extract_hand_params] Found hA: {hand_pose.shape}")

        # ================================================================
        # STEP 4: Search all batch keys for compatible tensors
        # ================================================================
        if hand_pose is None:
            logger.warning("[_extract_hand_params] Standard keys not found. Searching batch...")
            logger.debug(f"[_extract_hand_params] Available keys: {list(batch.keys())}")

            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    # Look for tensors with MANO-compatible dimensions
                    if value.shape[-1] in [45, 48]:
                        logger.info(f"[_extract_hand_params] Found candidate '{key}': {value.shape}")
                        hand_pose = value
                        source_key = key
                        break

        # ================================================================
        # STEP 5: Validate extraction
        # ================================================================
        if hand_pose is None:
            logger.error("[_extract_hand_params] Cannot find hand parameters!")
            logger.error(f"Batch keys: {list(batch.keys())}")

            # Log all tensor shapes for debugging
            tensor_info = []
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    tensor_info.append(f"  {key}: {value.shape}")

            if tensor_info:
                logger.error("Available tensors:\n" + "\n".join(tensor_info))

            raise ValueError(
                "Cannot extract hand parameters from batch. "
                "No compatible tensor found."
            )

        # ================================================================
        # STEP 6: Normalize shape to [B, 45] or [B, 48]
        # ================================================================
        if hand_pose.dim() == 1:
            hand_pose = hand_pose.unsqueeze(0)
            logger.debug(f"[_extract_hand_params] Added batch dimension: {hand_pose.shape}")

        # Validate dimension
        if hand_pose.shape[-1] not in [45, 48]:
            logger.error(
                f"[_extract_hand_params] Invalid pose dimension: {hand_pose.shape}. "
                f"Expected [..., 45] or [..., 48]. Source: '{source_key}'"
            )
            raise ValueError(f"Invalid hand pose shape: {hand_pose.shape}")

        # ================================================================
        # STEP 7: Extract shape (beta) parameters
        # ================================================================
        hand_shape = None

        # Try decomposed field first
        if 'right.betas' in batch:
            hand_shape = batch['right.betas']
            logger.debug(f"[_extract_hand_params] Found right.betas: {hand_shape.shape}")

        elif 'left.betas' in batch:
            hand_shape = batch['left.betas']
            logger.debug(f"[_extract_hand_params] Found left.betas: {hand_shape.shape}")

        # Try extracting from composite params
        elif 'right.params' in batch:
            params_full = batch['right.params']
            if params_full.shape[-1] >= 58:
                hand_shape = params_full[..., 48:58]  # Extract beta [48:58]
                logger.debug(f"[_extract_hand_params] Extracted shape from params: {hand_shape.shape}")

        elif 'beta' in batch:
            hand_shape = batch['beta']

        # Fallback: zero shape
        if hand_shape is None:
            B = hand_pose.shape[0]
            hand_shape = torch.zeros(B, 10, device=hand_pose.device)
            logger.debug(f"[_extract_hand_params] Using zero shape: {hand_shape.shape}")

        # ================================================================
        # STEP 8: Extract translation parameters
        # ================================================================
        hand_trans = None

        # Try decomposed field first
        if 'right.transl' in batch:
            hand_trans = batch['right.transl']
            logger.debug(f"[_extract_hand_params] Found right.transl: {hand_trans.shape}")

        elif 'left.transl' in batch:
            hand_trans = batch['left.transl']
            logger.debug(f"[_extract_hand_params] Found left.transl: {hand_trans.shape}")

        elif 'trans' in batch:
            hand_trans = batch['trans']

        # Fallback: zero translation
        if hand_trans is None or not isinstance(hand_trans, torch.Tensor):
            B = hand_pose.shape[0]
            hand_trans = torch.zeros(B, 3, device=hand_pose.device)
            logger.debug(f"[_extract_hand_params] Using zero trans: {hand_trans.shape}")

        # ================================================================
        # STEP 9: Construct and return dictionary
        # ================================================================
        hand_params = {
            'pose': hand_pose,      # [B, 45] or [B, 48]
            'shape': hand_shape,    # [B, 10]
            'trans': hand_trans,    # [B, 3]
        }

        logger.debug(
            f"[_extract_hand_params] ✓ Extracted from '{source_key}': "
            f"pose={hand_params['pose'].shape}, "
            f"shape={hand_params['shape'].shape}, "
            f"trans={hand_params['trans'].shape}"
        )

        return hand_params

    def _unwrap_xdict_to_tensor(self, obj):
        import torch
        if isinstance(obj, torch.Tensor):
            return obj
        # xdict: try values as method first, then attribute
        if hasattr(obj, 'values'):
            if callable(obj.values):
                # CRITICAL FIX: Call .values() with parentheses
                try:
                    values_iter = obj.values()  # NOT obj.values
                    for val in values_iter:
                        tensor = self._unwrap_xdict_to_tensor(val)
                        if isinstance(tensor, torch.Tensor):
                            return tensor
                except Exception as e:
                    logger.debug(f"Failed to iterate .values(): {e}")
            else:
                # Use as attribute
                tensor = self._unwrap_xdict_to_tensor(obj.values)
                if isinstance(tensor, torch.Tensor):
                    return tensor
        # dict: walk all .values()
        if isinstance(obj, dict):
            for val in obj.values():
                tensor = self._unwrap_xdict_to_tensor(val)
                if isinstance(tensor, torch.Tensor):
                    return tensor
        if isinstance(obj, (list, tuple)):
            for val in obj:
                tensor = self._unwrap_xdict_to_tensor(val)
                if isinstance(tensor, torch.Tensor):
                    return tensor
        return obj

    def training_epoch_end(self, outputs) -> None:
        current_step = self.global_step
        current_epoch = self.current_epoch

        # Canonical mesh update every 3 epochs
        if (
            current_epoch > 0 and current_epoch % 3 == 0 and not self.args.no_meshing
        ) or (current_step > 0 and self.args.fast_dev_run and not self.args.no_meshing):
            self.meshing_cano(current_step)
            self.save_misc()

        return super().training_epoch_end(outputs)

    def meshing_cano(self, current_step):
        mesh_dict = {}
        for node in self.model.nodes.values():
            try:
                mesh_c = node.meshing_cano()
                out_p = op.join(
                    self.args.log_dir,
                    "mesh_cano",
                    f"mesh_cano_{node.node_id}_step_{current_step}.obj",
                )
                os.makedirs(op.dirname(out_p), exist_ok=True)
                mesh_c.export(out_p)
                print(f"Exported canonical to {out_p}")
                mesh_dict[f"{node.node_id}_cano"] = mesh_c
            except:
                logger.error(f"Failed to mesh out {node.node_id}")
        return mesh_dict

    def inference_step(self, batch, *args, **kwargs):
        batch = xdict(batch).to("cuda")
        self.model.eval()
        batch = xdict(batch)
        batch["current_epoch"] = self.current_epoch
        batch["global_step"] = self.global_step

        for node in self.model.nodes.values():
            params = node.params(batch["idx"])
            batch.update(params)

        output = xdict()
        if not self.args.no_vis:
            batch = hold_utils.downsample_rendering(batch, self.args.render_downsample)
            split = split_input(
                batch,
                batch["total_pixels"][0],
                n_pixels=batch["pixel_per_batch"],
            )
            out_list = []
            pbar = tqdm(split)
            for s in pbar:
                pbar.set_description("Rendering")
                out = self.model(s).detach().to("cpu")
                vis_dict = {}
                vis_dict["rgb"] = out["rgb"]
                vis_dict["instance_map"] = out["instance_map"]
                vis_dict["bg_rgb_only"] = out["bg_rgb_only"]
                vis_dict.update(out.search("fg_rgb.vis"))
                vis_dict.update(out.search("mask_prob"))
                vis_dict.update(out.search("normal"))

                out_list.append(vis_dict)

            batch_size = batch["gt.rgb"].shape[0]
            model_outputs = merge_output(out_list, batch["total_pixels"][0], batch_size)
            output.update(model_outputs)

        output.update(batch)
        return output

    def inference_step_end(self, batch_parts):
        return batch_parts

    def validation_step(self, batch, *args, **kwargs):
        return self.inference_step(batch, *args, **kwargs)

    def test_step(self, batch, *args, **kwargs):
        out = self.inference_step(batch, *args, **kwargs)
        img_size = out["img_size"]
        normal = out["normal"]
        normal = normal.view(img_size[0], img_size[1], -1)
        normal_np = normal.numpy().astype(np.float16)

        exp_key = self.args.exp_key
        out_p = f"./exports/{exp_key}/normal/{out['idx']:04}.npy"
        os.makedirs(op.dirname(out_p), exist_ok=True)
        np.save(out_p, normal_np)
        print(f"Exported normal to {out_p}")
        return out

    def validation_step_end(self, batch_parts):
        return self.inference_step_end(batch_parts)

    def test_step_end(self, batch_parts):
        return self.inference_step_end(batch_parts)

    def validation_epoch_end(self, outputs) -> None:
        if not self.args.no_vis:
            img_size = outputs[0]["img_size"]
            idx = outputs[0]["idx"]
            vis_dict = vis_utils.output2images(outputs, img_size)
            vis_utils.record_vis(
                idx, self.global_step, self.args.log_dir, self.args.experiment, vis_dict
            )
