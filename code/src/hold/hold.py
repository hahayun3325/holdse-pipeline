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
                from src.model.ghop.ghop_prior import TwoStageTrainingManager  # FIX: Added import

                # Get device
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                # Initialize VQ-VAE
                logger.info(f"Loading VQ-VAE from {phase3_cfg.ghop.vqvae_checkpoint}")
                self.vqvae = GHOPVQVAEWrapper(
                    vqvae_ckpt_path=phase3_cfg.ghop.vqvae_checkpoint,
                    device=device,
                    use_hand_field=phase3_cfg.get('use_hand_field', True)
                )

                # Initialize 3D U-Net
                logger.info(f"Loading U-Net from {phase3_cfg.ghop.unet_checkpoint}")
                self.unet = GHOP3DUNetWrapper(
                    unet_ckpt_path=phase3_cfg.ghop.unet_checkpoint,
                    device=device
                )

                # Initialize Hand Field Builder
                mano_server = None
                for node in self.model.nodes.values():
                    if 'right' in node.node_id.lower() or 'left' in node.node_id.lower():
                        mano_server = node.server
                        break

                if mano_server is None:
                    raise ValueError("No hand node found in model. Cannot initialize HandFieldBuilder.")

                logger.info("Initializing Hand Field Builder...")
                # FIX: Consistent variable naming - use hand_field_builder (with underscore)
                self.hand_field_builder = HandFieldBuilder(
                    mano_server=mano_server,
                    resolution=phase3_cfg.get('grid_resolution', 64),
                    spatial_limit=phase3_cfg.get('spatial_limit', 1.5)
                )

                # Initialize SDS Loss Module
                logger.info("Initializing SDS Loss Module...")
                self.sds_loss = SDSLoss(
                    vqvae_wrapper=self.vqvae,
                    unet_wrapper=self.unet,
                    hand_field_builder=self.hand_field_builder,  # FIX: Consistent naming
                    guidance_scale=phase3_cfg.sds.get('guidance_scale', 4.0),
                    min_step_ratio=phase3_cfg.sds.get('min_step_ratio', 0.02),
                    max_step_ratio=phase3_cfg.sds.get('max_step_ratio', 0.98),
                    diffusion_steps=phase3_cfg.sds.get('diffusion_steps', 1000)
                )

                # Initialize Two-Stage Training Manager
                logger.info("Initializing Two-Stage Training Manager...")
                self.ghop_manager = TwoStageTrainingManager(
                    sds_loss_module=self.sds_loss,
                    sds_iters=phase3_cfg.get('sds_iters', 500),
                    contact_iters=phase3_cfg.get('contact_iters', 100),
                    max_sds_weight=phase3_cfg.get('w_sds', 5000.0),
                    max_contact_weight=phase3_cfg.get('w_contact', 10.0)
                )

                # Not using HOLDLoss wrapper in modular mode
                self.hold_loss_module = None

                logger.info(
                    f"✓ Phase 3 (Modular) initialized successfully:\n"
                    f"   - VQ-VAE: {phase3_cfg.ghop.vqvae_checkpoint}\n"
                    f"   - U-Net: {phase3_cfg.ghop.unet_checkpoint}\n"
                    f"   - Hand Field: {phase3_cfg.get('grid_resolution', 64)}³ resolution\n"
                    f"   - SDS guidance scale: {phase3_cfg.sds.get('guidance_scale', 4.0)}\n"
                    f"   - Stage 1 (SDS): {phase3_cfg.get('sds_iters', 500)} iterations\n"
                    f"   - Stage 2 (Contact): {phase3_cfg.get('contact_iters', 100)} iterations\n"
                    f"   - Max SDS weight: {phase3_cfg.get('w_sds', 5000.0)}\n"
                    f"   - Max Contact weight: {phase3_cfg.get('w_contact', 10.0)}"
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
                self.phase5_scheduler = Phase5TrainingScheduler(
                    total_iterations=phase5_cfg.get('total_iterations', 1000),
                    warmup_iters=phase5_cfg.get('warmup_iters', 100),
                    phase3_start=100,  # Standard warmup period
                    phase4_start=self.contact_start_iter if hasattr(self, 'contact_start_iter') else 500,
                    phase5_start=phase5_cfg.get('phase5_start_iter', 600),
                    finetune_start=phase5_cfg.get('finetune_start_iter', 800)
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

    def training_step(self, batch):
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

        # ====================================================================
        # PHASE 3 & PHASE 5: UNIFIED SDS COMPUTATION
        # ====================================================================
        if self.phase3_enabled and self.ghop_enabled:
            try:
                # Extract hand pose and object SDF
                hand_pose = self._extract_hand_params_from_batch(batch)
                object_sdf = self._extract_sdf_grid_from_nodes(batch)

                # Get text prompts
                category = batch.get('object_category', batch.get('category', 'object'))
                if isinstance(category, str):
                    text_prompt = f"a hand grasping a {category}"
                else:
                    text_prompt = "a hand grasping an object"

                # PHASE 5 ENHANCED SDS: Use GHOPDiffusionPrior if available
                if self.phase5_enabled and hasattr(self, 'diffusion_prior') and self.diffusion_prior is not None:
                    logger.debug(f"[Phase 5] Computing enhanced SDS via diffusion_prior at step {self.global_step}")

                    # Compute SDS loss with Phase 5 enhancements
                    sds_loss, sds_metrics = self.diffusion_prior(
                        hand_pose=hand_pose['pose'],
                        object_sdf=object_sdf,
                        iteration=self.global_step,
                        total_iterations=self.phase5_scheduler.total_iterations,
                        text_prompt=text_prompt
                    )

                    # Apply Phase 5 dynamic weighting
                    weighted_sds = sds_loss * loss_weights['sds']

                    # Add to total loss
                    loss_output['loss'] = loss_output['loss'] + weighted_sds
                    loss_output['sds_loss'] = weighted_sds

                    # Enhanced Phase 5 logging
                    self.log('phase5/sds_loss', weighted_sds, prog_bar=True)
                    self.log('phase5/sds_timestep', sds_metrics['timestep'], prog_bar=False)
                    self.log('phase5/sds_weight_mean', sds_metrics['weight_mean'], prog_bar=False)
                    self.log('phase5/sds_latent_norm', sds_metrics['latent_norm'], prog_bar=False)
                    self.log('phase5/sds_grad_norm', sds_metrics['grad_norm'], prog_bar=False)

                    if self.global_step % self.log_phase5_every == 0:
                        logger.info(
                            f"\n[Phase 5 - Step {self.global_step}] Enhanced SDS:\n"
                            f"  Loss:        {weighted_sds.item():.4f}\n"
                            f"  Timestep:    {sds_metrics['timestep']:.1f}\n"
                            f"  Weight:      {sds_metrics['weight_mean']:.3f}\n"
                            f"  Latent norm: {sds_metrics['latent_norm']:.3f}\n"
                            f"  Grad norm:   {sds_metrics['grad_norm']:.3f}"
                        )

                # PHASE 3 LEGACY SDS: Use TwoStageTrainingManager
                elif self.ghop_manager is not None:
                    logger.debug(f"[Phase 3] Computing SDS via ghop_manager at step {self.global_step}")

                    # Prepare text prompts
                    if isinstance(category, str):
                        text_prompts = [text_prompt] * hand_pose['pose'].shape[0]
                    else:
                        text_prompts = None

                    # Compute GHOP losses via Phase 3 manager
                    ghop_losses, ghop_info = self.ghop_manager(
                        hand_pose=hand_pose,
                        object_sdf=object_sdf,
                        iteration=self.global_step,
                        text_prompts=text_prompts
                    )

                    # Apply Phase 5 dynamic weighting if enabled
                    if self.phase5_enabled:
                        weighted_sds = ghop_losses.get('sds', 0.0) * loss_weights['sds']
                        weighted_contact = ghop_losses.get('contact', 0.0) * loss_weights['contact']
                        total_ghop_loss = weighted_sds + weighted_contact
                    else:
                        total_ghop_loss = ghop_losses.get('total', 0.0)

                    # Add to total loss
                    loss_output['loss'] = loss_output['loss'] + total_ghop_loss

                    # Logging
                    stage = ghop_info.get('stage', 'unknown')
                    self.log('ghop/stage', stage, prog_bar=True)
                    if 'sds' in ghop_losses:
                        self.log('ghop/sds_loss', ghop_losses['sds'].item(), prog_bar=True)
                    if 'contact' in ghop_losses:
                        self.log('ghop/contact_loss', ghop_losses['contact'].item())

            except Exception as e:
                logger.error(
                    f"[Phase 3/5] SDS computation failed at step {self.global_step}: {e}\n"
                    f"Continuing with standard HOLD losses only."
                )
                import traceback
                traceback.print_exc()

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

                    # Contact refiner signature: (hand_verts, hand_faces, obj_verts, obj_faces, contact_zones)
                    contact_loss_b, contact_metrics_b = self.contact_refiner(
                        hand_verts=h_verts.unsqueeze(0),  # [1, 778, 3]
                        hand_faces=h_faces,                # [F, 3]
                        obj_verts=o_verts.unsqueeze(0),   # [1, V_obj, 3]
                        obj_faces=o_faces,                 # [F_obj, 3]
                        contact_zones=zones_b              # [K] adaptive indices or None
                    )

                    total_contact_loss += contact_loss_b
                    num_valid_samples += 1

                    # Accumulate metrics
                    for key in contact_metrics_accum:
                        if key in contact_metrics_b:
                            contact_metrics_accum[key] += contact_metrics_b[key]

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
                hand_pose_params = self._extract_hand_params_from_batch(batch)

                for node in self.model.nodes.values():
                    if "object" in node.node_id.lower():
                        object_node = node
                        break

                if object_node is not None and hand_pose_params is not None:
                    category = batch.get('object_category', batch.get('category', 'object'))
                    if isinstance(category, (list, tuple)):
                        category = category[0]

                    loss_sds, sds_info = self.hold_loss_module.compute_sds_loss(
                        object_node=object_node,
                        hand_pose=hand_pose_params,
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
                    predicted_pose = self._extract_predicted_hand_pose(batch, model_outputs)

                    # Verify predicted_pose shape
                    if predicted_pose is None or predicted_pose.shape[-1] != 45:
                        logger.warning(
                            f"[Phase 5] Invalid predicted_pose shape: {predicted_pose.shape if predicted_pose is not None else None}. "
                            f"Expected [..., 45]. Skipping temporal consistency."
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
                            f"This may be a single-frame dataset. Skipping temporal consistency."
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
                        sequence_id=sequence_id
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
        # Determine batch size
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

        # Create coordinate grid in canonical space [-1.5, 1.5]³
        x = torch.linspace(-1.5, 1.5, H, device=device)
        grid = torch.stack(torch.meshgrid(x, x, x, indexing='ij'), dim=-1)
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1, -1)  # (B, H, H, H, 3)

        # Find object node
        object_node = None
        for node in self.model.nodes.values():
            if "object" in node.node_id.lower():
                object_node = node
                break

        if object_node is None:
            logger.warning("[Helper] No object node found, returning zero SDF")
            return torch.zeros(B, 1, H, H, H, device=device)

        # Query object node's SDF network
        grid_flat = grid.reshape(B, -1, 3)  # (B, H³, 3)

        with torch.no_grad():
            try:
                sdf_output = object_node.server.forward_sdf(grid_flat)
                if isinstance(sdf_output, dict):
                    sdf_values = sdf_output['sdf']
                else:
                    sdf_values = sdf_output
            except AttributeError:
                try:
                    sdf_values = object_node.server.shape_net(
                        grid_flat,
                        object_node.server.latent_code
                    )['sdf']
                except:
                    logger.warning("[Helper] SDF extraction failed, using zero grid")
                    return torch.zeros(B, 1, H, H, H, device=device)

        # Reshape to grid
        if sdf_values.dim() == 2:
            sdf_values = sdf_values.unsqueeze(-1)

        object_sdf = sdf_values.reshape(B, H, H, H, 1).permute(0, 4, 1, 2, 3)

        return object_sdf

    def _extract_hand_params_from_batch(self, batch):
        """Extract hand pose parameters from batch."""
        hand_pose_params = None

        for node in self.model.nodes.values():
            if "hand" in node.node_id.lower() or "right" in node.node_id.lower() or "left" in node.node_id.lower():
                mano_pose_key = f"{node.node_id}.mano_pose"
                mano_rot_key = f"{node.node_id}.mano_rot"

                mano_pose = batch.get(mano_pose_key, None)
                mano_rot = batch.get(mano_rot_key, None)

                if mano_rot is not None and mano_pose is not None:
                    if mano_pose.shape[-1] == 45:
                        hand_pose_params = torch.cat([mano_rot, mano_pose], dim=-1)
                    elif mano_pose.shape[-1] == 48:
                        hand_pose_params = mano_pose
                elif mano_pose is not None:
                    if mano_pose.shape[-1] == 48:
                        hand_pose_params = mano_pose
                    else:
                        device = mano_pose.device
                        zero_rot = torch.zeros(mano_pose.shape[0], 3, device=device)
                        hand_pose_params = torch.cat([zero_rot, mano_pose], dim=-1)

                if hand_pose_params is not None:
                    break

        return hand_pose_params

    # ====================================================================
    # PHASE 4: Mesh Extraction Helper Methods
    # ====================================================================

    def _extract_hand_mesh(self, batch):
        """Extract hand mesh from MANO parameters."""
        # Find hand node
        hand_node = None
        node_id = None

        for node in self.model.nodes.values():
            if 'right' in node.node_id.lower() or 'left' in node.node_id.lower():
                hand_node = node
                node_id = node.node_id
                break

        if hand_node is None:
            raise ValueError("[Phase 4] No hand node found in model")

        # Get MANO parameters from batch
        mano_pose_key = f"{node_id}.mano_pose"
        mano_rot_key = f"{node_id}.mano_rot"
        mano_shape_key = f"{node_id}.mano_shape"
        mano_trans_key = f"{node_id}.mano_trans"

        mano_pose = batch.get(mano_pose_key, None)
        mano_rot = batch.get(mano_rot_key, None)
        mano_shape = batch.get(mano_shape_key, None)
        mano_trans = batch.get(mano_trans_key, None)

        if mano_pose is None:
            raise ValueError(f"[Phase 4] Missing MANO pose in batch (key: {mano_pose_key})")

        # Forward MANO server to get mesh
        with torch.no_grad():
            mano_server = hand_node.server

            # Construct full pose [B, 48]
            if mano_rot is not None and mano_pose.shape[-1] == 45:
                full_pose = torch.cat([mano_rot, mano_pose], dim=-1)
            elif mano_pose.shape[-1] == 48:
                full_pose = mano_pose
            else:
                device = mano_pose.device
                zero_rot = torch.zeros(mano_pose.shape[0], 3, device=device)
                full_pose = torch.cat([zero_rot, mano_pose], dim=-1)

            # Get MANO output
            mano_output = mano_server(
                pose=full_pose,
                shape=mano_shape if mano_shape is not None else torch.zeros(
                    full_pose.shape[0], 10, device=full_pose.device
                ),
                trans=mano_trans if mano_trans is not None else torch.zeros(
                    full_pose.shape[0], 3, device=full_pose.device
                )
            )

            # Extract vertices and faces
            if isinstance(mano_output, dict):
                hand_verts = mano_output['vertices']  # [B, 778, 3]
            else:
                hand_verts = mano_output

            hand_faces = mano_server.faces  # [1538, 3]

        logger.debug(f"[Phase 4] Extracted hand mesh: {hand_verts.shape}")
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

        Retrieves the predicted hand parameters from HOLD's node-based architecture
        and formats them for TemporalConsistencyModule.

        Args:
            batch: Training batch dictionary
            model_outputs: Model forward pass outputs

        Returns:
            hand_pose: [B, 45] predicted hand pose parameters
                       Format: global_orient(3) + hand_pose(45) from MANO
        """
        hand_pose = None

        # Method 1: Extract from model node parameters (primary method)
        for node_id, node in self.model.nodes.items():
            if 'right' in node_id.lower() or 'left' in node_id.lower():
                # Get predicted pose parameters from node
                # node.params(batch['idx']) returns optimized parameters
                params = node.params(batch['idx'])  # [B, 48] or [B, 51]

                # Extract pose (first 48 parameters: 3 global_orient + 45 hand_pose)
                if params.shape[-1] >= 48:
                    hand_pose = params[..., :48]  # [B, 48]
                else:
                    hand_pose = params

                logger.debug(
                    f"[Phase 5] Extracted hand pose from node '{node_id}': "
                    f"shape={hand_pose.shape}"
                )
                break  # Use first hand node found

        # Method 2: Fallback to batch data if node extraction fails
        if hand_pose is None:
            if 'hA' in batch:
                hand_pose = batch['hA']  # [B, 45] or [B, 48]
                logger.debug(
                    f"[Phase 5] Using batch 'hA' as hand pose fallback: "
                    f"shape={hand_pose.shape}"
                )
            else:
                logger.warning(
                    "[Phase 5] Could not extract hand pose from model or batch. "
                    "Temporal consistency will be skipped."
                )
                return None

        # Ensure correct shape [B, 45] or [B, 48]
        if hand_pose.ndim == 1:
            hand_pose = hand_pose.unsqueeze(0)

        # Truncate to 45 if needed (remove translation/shape if present)
        if hand_pose.shape[-1] > 48:
            hand_pose = hand_pose[..., :48]

        return hand_pose


    def _extract_hand_params_from_batch(self, batch: Dict) -> Dict:
        """
        Extract hand parameters dictionary for GHOP modules.

        This method is used by Phase 3 SDS and Phase 5 temporal modules.

        Args:
            batch: Training batch dictionary

        Returns:
            hand_params: Dictionary with keys:
                - 'pose': [B, 45 or 48] hand joint rotations
                - 'shape': [B, 10] MANO shape parameters (beta)
                - 'trans': [B, 3] global translation
        """
        hand_params = {}

        for node in self.model.nodes.values():
            if 'right' in node.node_id.lower() or 'left' in node.node_id.lower():
                params = node.params(batch['idx'])
                hand_params['pose'] = params  # [B, 48+]

                # Extract or initialize shape and translation
                if params.shape[-1] >= 58:  # 48 (pose) + 10 (shape)
                    hand_params['shape'] = params[..., 48:58]
                else:
                    hand_params['shape'] = torch.zeros(
                        params.shape[0], 10,
                        device=params.device
                    )

                # Translation
                if 'trans' in batch:
                    hand_params['trans'] = batch['trans']
                else:
                    hand_params['trans'] = torch.zeros(
                        params.shape[0], 3,
                        device=params.device
                    )

                break

        return hand_params

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
