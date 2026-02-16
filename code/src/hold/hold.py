import os
import os.path as op
import sys

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
from src.model.ghop.temporal_diagnostic import TemporalMemoryDiagnostic
from src.model.ghop.adaptive_contact_zones import AdaptiveContactZones
from src.training.phase5_scheduler import Phase5TrainingScheduler
from src.training.sds_weight_scheduler import create_sds_scheduler_from_config
import gc
from src.utils.memory_profiler import MemoryProfiler
import subprocess
from src.hold.loss_terms import get_smoothness_loss
from src.hold.geometric_losses import normal_consistency_loss, depth_smoothness_loss
from pytorch3d.structures import Meshes
from pytorch3d.loss import mesh_normal_consistency
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

        # GHOP FIX: Disable automatic optimization for manual control
        self.automatic_optimization = False

        self.opt = opt
        self.args = args
        num_frames = args.n_images

        data_path = os.path.join("./data", args.case, f"build/data.npy")
        entities = np.load(data_path, allow_pickle=True).item()["entities"]

        betas_r = entities["right"]["mean_shape"] if "right" in entities else None
        betas_l = entities["left"]["mean_shape"] if "left" in entities else None

        # Force FiLM configuration before model creation (around line 130)
        if hasattr(opt.model, 'obj_implicit_network'):
            # opt.model.obj_implicit_network.cond_dim = 0  # FiLM doesn't concatenate
            opt.model.obj_implicit_network.use_film = True
            # opt.model.obj_implicit_network.film_cond_dim = 128  # Separate FiLM dim


        self.model = HOLDNet(
            opt.model,
            betas_r,
            betas_l,
            num_frames,
            args,
        )

        # Log trainable parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logger.info("=" * 70)
        logger.info("MODEL PARAMETER SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Frozen parameters: {total_params - trainable_params:,}")

        # Breakdown by component
        for node_name, node in self.model.nodes.items():
            node_params = sum(p.numel() for p in node.parameters())
            node_trainable = sum(p.numel() for p in node.parameters() if p.requires_grad)
            logger.info(f"  {node_name}: {node_trainable:,} / {node_params:,} trainable")

            # Check object vertices specifically
            if 'object' in node_name.lower():
                if hasattr(node, 'server') and hasattr(node.server, 'object_model'):
                    om = node.server.object_model
                    if hasattr(om, 'v3d_cano'):
                        v3d = om.v3d_cano
                        logger.info(
                            f"    - v3d_cano: shape={v3d.shape}, requires_grad={v3d.requires_grad}, numel={v3d.numel():,}")

        logger.info("=" * 70)

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
        # ✅ ADD: Initialize diagnostic as None (will be set if Phase 5 enabled)
        self.temporal_diagnostic = None
        self._diagnostic_enabled = False
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

            self.grid_resolution = phase3_cfg.get('grid_resolution', 32)  # match with the config

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
                from src.model.ghop.ghop_loss import GHOPSDSLoss
                from src.model.ghop.ghop_prior import TwoStageTrainingManager

                # Get device
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                # ============================================================
                # CRITICAL FIX: Check pretrained flags BEFORE file access
                # ============================================================
                vqvae_use_pretrained = phase3_cfg.ghop.get('vqvae_use_pretrained', False)
                unet_use_pretrained = phase3_cfg.ghop.get('unet_use_pretrained', False)

                # ============================================================
                # DETERMINE CHECKPOINT PATHS
                # ============================================================
                # CORRECTED PRIORITY ORDER:
                # 1. args.infer_ckpt - Explicit GHOP checkpoint override
                # 2. Config file - Default GHOP checkpoint paths (RECOMMENDED)
                # 3. args.ckpt_p - Legacy fallback (NOT recommended - this is HOLD checkpoint)

                if hasattr(args, 'infer_ckpt') and args.infer_ckpt:
                    # Command-line override (use same checkpoint for both)
                    # This is for rendering/inference with specific GHOP checkpoint
                    vqvae_checkpoint = args.infer_ckpt
                    unet_checkpoint = args.infer_ckpt
                    model_checkpoint = args.infer_ckpt
                    logger.info(f"[GHOP] Using checkpoint from args.infer_ckpt: {args.infer_ckpt}")

                elif hasattr(phase3_cfg, 'ghop') and phase3_cfg.ghop.get('unified_checkpoint'):
                    # ============================================================
                    # USE CONFIG FILE (CORRECT PATH FOR TRAINING)
                    # ============================================================
                    # This is the recommended path for training
                    # Config specifies separate GHOP checkpoint from HOLD checkpoint

                    # Try unified checkpoint first (recommended)
                    unified_path = phase3_cfg.ghop.get('unified_checkpoint')
                    if unified_path:
                        vqvae_checkpoint = unified_path
                        unet_checkpoint = unified_path
                        model_checkpoint = unified_path
                        logger.info(f"[GHOP] Using unified checkpoint from config: {unified_path}")
                    else:
                        # Fall back to separate paths
                        vqvae_checkpoint = phase3_cfg.ghop.get('vqvae_checkpoint', 'checkpoints/ghop/last.ckpt')
                        unet_checkpoint = phase3_cfg.ghop.get('unet_checkpoint', 'checkpoints/ghop/last.ckpt')
                        model_checkpoint = vqvae_checkpoint  # Use VQ-VAE path as default
                        logger.info(f"[GHOP] Using separate checkpoints from config:")
                        logger.info(f"  VQ-VAE: {vqvae_checkpoint}")
                        logger.info(f"  U-Net: {unet_checkpoint}")

                elif hasattr(args, 'ckpt_p') and args.ckpt_p:
                    # ============================================================
                    # LEGACY FALLBACK (NOT RECOMMENDED)
                    # ============================================================
                    # args.ckpt_p is typically the HOLD checkpoint for resuming training
                    # Using it for GHOP is NOT recommended

                    logger.warning("=" * 70)
                    logger.warning("USING HOLD CHECKPOINT FOR GHOP (NOT RECOMMENDED)")
                    logger.warning("=" * 70)
                    logger.warning(f"  Checkpoint: {args.ckpt_p}")
                    logger.warning("")
                    logger.warning("This is args.ckpt_p (from --load_ckpt), which is typically")
                    logger.warning("a HOLD checkpoint, NOT a GHOP checkpoint.")
                    logger.warning("")
                    logger.warning("RECOMMENDED FIX:")
                    logger.warning("  Set phase3.ghop.unified_checkpoint in config file to:")
                    logger.warning("  /path/to/ghop/checkpoint/last.ckpt")
                    logger.warning("=" * 70)

                    vqvae_checkpoint = args.ckpt_p
                    unet_checkpoint = args.ckpt_p
                    model_checkpoint = args.ckpt_p
                    logger.info(f"[GHOP] Using checkpoint from args.ckpt_p: {args.ckpt_p}")

                else:
                    # No checkpoint specified
                    logger.warning("No GHOP checkpoint path specified")
                    logger.warning("Using default path: checkpoints/ghop/last.ckpt")
                    vqvae_checkpoint = 'checkpoints/ghop/last.ckpt'
                    unet_checkpoint = 'checkpoints/ghop/last.ckpt'
                    model_checkpoint = 'checkpoints/ghop/last.ckpt'

                need_checkpoint = vqvae_use_pretrained or unet_use_pretrained
                logger.info(f"\nCheckpoint configuration:")
                logger.info(f"  VQ-VAE pretrained: {vqvae_use_pretrained}")
                logger.info(f"  U-Net pretrained: {unet_use_pretrained}")
                logger.info(f"  Checkpoint needed: {need_checkpoint}")
                if need_checkpoint:
                    logger.info(f"  Checkpoint path: {model_checkpoint}")

                    # Verify checkpoint exists
                    if not os.path.exists(model_checkpoint):
                        raise FileNotFoundError(
                            f"GHOP checkpoint not found: {model_checkpoint}\n"
                            f"Please ensure the checkpoint exists at the specified path."
                        )

                # ============================================================
                # Initialize VQ-VAE
                # ============================================================
                # Determine if we need the checkpoint file
                logger.info("Initializing VQ-VAE...")
                self.vqvae = GHOPVQVAEWrapper(
                    vqvae_ckpt_path=vqvae_checkpoint if vqvae_use_pretrained else None,
                    device=device,
                    use_hand_field=phase3_cfg.get('use_hand_field', True)
                )

                if vqvae_use_pretrained:
                    logger.info(f"✓ VQ-VAE initialized with pretrained weights from: {vqvae_checkpoint}")
                else:
                    logger.info("✓ VQ-VAE initialized with RANDOM weights")

                # ============================================================
                # Initialize 3D U-Net
                # ============================================================
                logger.info("Initializing U-Net...")
                self.unet = GHOP3DUNetWrapper(
                    unet_ckpt_path=unet_checkpoint if unet_use_pretrained else None,
                    device=device
                )

                if unet_use_pretrained:
                    logger.info(f"✓ U-Net initialized with pretrained weights from: {unet_checkpoint}")
                else:
                    logger.info("✓ U-Net initialized with RANDOM weights")


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
                    resolution=phase3_cfg.get('grid_resolution', 32),
                    spatial_limit=phase3_cfg.get('spatial_limit', 1.5)
                )
                logger.info("✓ Hand Field Builder initialized")

                # ============================================================
                # Load GHOP Checkpoint for CLIP Text Encoder
                # ============================================================
                logger.info("Initializing CLIP text encoder...")

                # ✅ NEW: Check if we should use Hugging Face CLIP
                use_hf_clip = phase3_cfg.ghop.get('use_huggingface_clip', False)
                clip_model_name = phase3_cfg.ghop.get('clip_model_name', 'openai/clip-vit-large-patch14')

                if use_hf_clip:
                    # ============================================================
                    # PATH A: Use Hugging Face CLIP (RECOMMENDED)
                    # ============================================================
                    logger.info(f"Using Hugging Face CLIP: {clip_model_name}")
                    logger.critical(f"[INIT] Using Hugging Face CLIP path (use_huggingface_clip=true)")

                    from transformers import CLIPTokenizer, CLIPTextModel

                    class HuggingFaceCLIPPrior:
                        """CLIP text encoder wrapper using Hugging Face transformers."""
                        def __init__(self, model_name, device='cuda'):
                            self.device = device
                            self.model_name = model_name

                            logger.info(f"Loading CLIP model: {model_name}")
                            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
                            self.text_encoder = CLIPTextModel.from_pretrained(model_name).to(device)
                            self.text_encoder.eval()

                            # Freeze CLIP weights (we're using it as a frozen feature extractor)
                            for param in self.text_encoder.parameters():
                                param.requires_grad = False

                            num_params = sum(p.numel() for p in self.text_encoder.parameters())
                            logger.info(f"✅ Loaded {model_name} ({num_params:,} parameters)")

                        def encode_text(self, text_prompts):
                            """
                            Encode text prompts using CLIP.

                            Args:
                                text_prompts: List of text strings

                            Returns:
                                text_embeddings: (B, 1, 768) tensor for U-Net conditioning
                            """
                            # ============================================================
                            # CRITICAL: Validate input format
                            # ============================================================
                            if not isinstance(text_prompts, list):
                                logger.error(f"[CLIP] Invalid input type: {type(text_prompts)}, expected list")
                                raise TypeError(f"Expected list, got {type(text_prompts)}")

                            for i, prompt in enumerate(text_prompts):
                                if not isinstance(prompt, str):
                                    logger.error(f"[CLIP] Prompt {i} is not a string: {type(prompt)} = {repr(prompt)}")
                                    raise TypeError(f"Prompt {i} must be string, got {type(prompt)}: {repr(prompt)}")

                            logger.debug(f"[CLIP] Encoding {len(text_prompts)} prompts")

                            with torch.no_grad():
                                try:
                                    inputs = self.tokenizer(
                                        text_prompts,
                                        padding=True,
                                        truncation=True,
                                        max_length=77,
                                        return_tensors="pt"
                                    ).to(self.device)

                                    outputs = self.text_encoder(**inputs)
                                    embeddings = outputs.pooler_output  # (B, 768)

                                    # Validate output dimensions
                                    logger.debug(f"[CLIP] pooler_output shape: {embeddings.shape}")
                                    logger.debug(f"[CLIP] Expected: (B, 768) for ViT-L/14")

                                    if embeddings.shape[-1] != 768:
                                        logger.error(
                                            f"[CLIP] ❌ WRONG CLIP MODEL! Got {embeddings.shape[-1]} dims, need 768! "
                                            f"Model {self.model_name} is incompatible (use ViT-L/14, not ViT-B/32)"
                                        )
                                        raise ValueError(f"CLIP dimension mismatch: {embeddings.shape[-1]} != 768")

                                    # Add sequence dimension for U-Net cross-attention
                                    return embeddings.unsqueeze(1)  # (B, 1, 768)

                                except Exception as e:
                                    logger.error(f"[CLIP] Encoding failed: {e}")
                                    logger.error(f"[CLIP] Input prompts: {text_prompts}")
                                    raise

                    self.ghop_prior = HuggingFaceCLIPPrior(clip_model_name, device='cuda')
                    logger.info("✅ GHOP Prior Module initialized with Hugging Face CLIP")

                else:
                    # ============================================================
                    # PATH B: Use CLIP from GHOP checkpoint (ORIGINAL)
                    # ============================================================
                    logger.info("Loading GHOP checkpoint for CLIP text encoder...")

                    if need_checkpoint and os.path.exists(model_checkpoint):
                        logger.info(f"Loading GHOP checkpoint: {model_checkpoint}")
                        logger.critical(f"[INIT] Using MinimalGHOPPrior path (checkpoint exists)")
                        logger.critical(f"[INIT] Checkpoint path: {model_checkpoint}")

                        ghop_checkpoint = torch.load(model_checkpoint, map_location='cuda')

                        # ============================================================
                        # Create Minimal Prior Wrapper with CLIP Encoder
                        # ============================================================
                        class MinimalGHOPPrior:
                            """
                            Minimal wrapper providing text encoding interface for SDSLoss.
                            Extracts and uses CLIP encoder from GHOP checkpoint.
                            """
                            def __init__(self, checkpoint, device='cuda'):
                                self.device = device

                                # Extract state dict
                                if 'state_dict' in checkpoint:
                                    state_dict = checkpoint['state_dict']
                                else:
                                    state_dict = checkpoint

                                # Try to load CLIP encoder from checkpoint
                                try:
                                    # Import CLIP encoder class from GHOP codebase
                                    from ldm.modules.encoders.modules import FrozenCLIPEmbedder

                                    self.clip_encoder = FrozenCLIPEmbedder().to(device)

                                    # Filter CLIP encoder weights
                                    clip_state = {
                                        k.replace('glide_model.text_cond_model.', ''): v
                                        for k, v in state_dict.items()
                                        if k.startswith('glide_model.text_cond_model.')
                                    }

                                    if clip_state:
                                        # Load filtered weights
                                        missing, unexpected = self.clip_encoder.load_state_dict(clip_state, strict=False)
                                        logger.info(f"✅ Loaded CLIP encoder from checkpoint")
                                        logger.info(f"  - Loaded parameters: {len(clip_state)}")
                                        if missing:
                                            logger.info(f"  - Missing keys: {len(missing)}")
                                        if unexpected:
                                            logger.info(f"  - Unexpected keys: {len(unexpected)}")
                                    else:
                                        logger.warning("No CLIP encoder found in checkpoint, using pretrained CLIP")
                                        # Fall back to Hugging Face CLIP
                                        from transformers import CLIPTokenizer, CLIPTextModel
                                        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
                                        self.clip_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
                                        self.use_hf_clip = True

                                    self.clip_encoder.eval()
                                    self.use_hf_clip = False

                                except ImportError:
                                    # If FrozenCLIPEmbedder not available, use Hugging Face CLIP
                                    logger.warning("FrozenCLIPEmbedder not available, using Hugging Face CLIP")
                                    from transformers import CLIPTokenizer, CLIPTextModel

                                    self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
                                    self.clip_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
                                    self.clip_encoder.eval()
                                    self.use_hf_clip = True

                            def encode_text(self, text_prompts):
                                """Encode text prompts using CLIP encoder."""
                                logger.info(f"[MINIMAL-GHOP-CLIP] Encoding {len(text_prompts)} prompts")

                                with torch.no_grad():
                                    if self.use_hf_clip:
                                        # Hugging Face CLIP path
                                        inputs = self.tokenizer(
                                            text_prompts,
                                            padding=True,
                                            truncation=True,
                                            max_length=77,
                                            return_tensors="pt"
                                        ).to(self.device)

                                        outputs = self.clip_encoder(**inputs)
                                        embeddings = outputs.pooler_output  # (B, 768)

                                        logger.info(f"[MINIMAL-GHOP-CLIP] HF CLIP output shape: {embeddings.shape}")
                                    else:
                                        # FrozenCLIPEmbedder path (original GHOP encoder)
                                        embeddings = self.clip_encoder.encode(text_prompts)  # (B, 768)
                                        logger.info(f"[MINIMAL-GHOP-CLIP] Frozen CLIP output shape: {embeddings.shape}")

                                    # Add sequence dimension for U-Net cross-attention
                                    return embeddings.unsqueeze(1)  # (B, 1, 768)

                        # Create the prior module
                        self.ghop_prior = MinimalGHOPPrior(ghop_checkpoint, device='cuda')
                        logger.info("✅ GHOP Prior Module initialized with CLIP encoder from checkpoint")

                    else:
                        # No checkpoint available - use Hugging Face CLIP directly
                        logger.warning("No GHOP checkpoint available, falling back to Hugging Face CLIP")
                        logger.critical(f"[INIT] Using HuggingFaceCLIPPrior fallback path (no checkpoint)")

                        # Reuse the same HuggingFaceCLIPPrior class defined above
                        from transformers import CLIPTokenizer, CLIPTextModel

                        class SimpleGHOPPrior:
                            """CLIP text encoder wrapper using Hugging Face transformers."""
                            def __init__(self, device='cuda'):
                                from transformers import CLIPTokenizer, CLIPTextModel

                                self.device = device
                                self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
                                self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
                                self.text_encoder.eval()

                                logger.info("✅ Initialized CLIP text encoder from Hugging Face")

                            def encode_text(self, text_prompts):
                                """Encode text prompts using CLIP."""
                                if not isinstance(text_prompts, list):
                                    raise TypeError(f"Expected list, got {type(text_prompts)}")

                                for i, prompt in enumerate(text_prompts):
                                    if not isinstance(prompt, str):
                                        raise TypeError(f"Prompt {i} must be string, got {type(prompt)}")

                                with torch.no_grad():
                                    inputs = self.tokenizer(
                                        text_prompts,
                                        padding=True,
                                        truncation=True,
                                        max_length=77,
                                        return_tensors="pt"
                                    ).to(self.device)

                                    outputs = self.text_encoder(**inputs)
                                    embeddings = outputs.pooler_output  # (B, 768)

                                    return embeddings.unsqueeze(1)  # (B, 1, 768)

                        self.ghop_prior = SimpleGHOPPrior(device='cuda')
                        logger.info("✅ GHOP Prior Module initialized with Hugging Face CLIP (fallback)")

                # ============================================================
                # Initialize SDS Loss Module with Weight Scheduler
                # ============================================================
                logger.info("Initializing GHOP SDS Loss Module...")

                try:
                    self.sds_weight_scheduler = create_sds_scheduler_from_config(phase3_cfg)
                    logger.info("✓ SDS Weight Scheduler initialized")
                    logger.info(self.sds_weight_scheduler.get_schedule_info())
                except Exception as e:
                    logger.error(f"Failed to create SDS scheduler: {e}")
                    logger.warning("Falling back to fixed SDS weight")
                    # Fallback: create dummy scheduler with fixed weight
                    from src.model.ghop.sds_weight_scheduler import SDSWeightScheduler
                    fixed_weight = phase3_cfg.get('w_sds', 10.0)
                    self.sds_weight_scheduler = SDSWeightScheduler(
                        schedule={0: fixed_weight},
                        enabled=False
                    )

                # Initialize RGB Weight Scheduler
                try:
                    rgb_schedule_cfg = getattr(self.opt, 'loss', {}).get('w_rgb_schedule', None)
                    if rgb_schedule_cfg and rgb_schedule_cfg.get('enabled', False):
                        from src.training.sds_weight_scheduler import SDSWeightScheduler  # Reuse same scheduler class
                        self.rgb_weight_scheduler = SDSWeightScheduler(
                            schedule={
                                0: rgb_schedule_cfg.get('step_0', 1.0),
                                8000: rgb_schedule_cfg.get('step_8000', 0.40),
                                15000: rgb_schedule_cfg.get('step_15000', 0.40),
                                25000: rgb_schedule_cfg.get('step_25000', 0.40),
                                30000: rgb_schedule_cfg.get('step_30000', 1.0),
                            },
                            interpolation=rgb_schedule_cfg.get('interpolation', 'linear')
                        )
                        logger.info("✓ RGB Weight Scheduler initialized")
                    else:
                        self.rgb_weight_scheduler = None
                        logger.info("RGB Weight Scheduler: Disabled (using fixed w_rgb)")
                except Exception as e:
                    logger.warning(f"Failed to create RGB scheduler: {e}")
                    self.rgb_weight_scheduler = None

                # Initialize SDS loss (weight will be set dynamically in training_step)
                self.sds_loss = GHOPSDSLoss(
                    vqvae_wrapper=self.vqvae,
                    unet_wrapper=self.unet,
                    hand_field_builder=self.hand_field_builder,
                    prior_module=self.ghop_prior,
                    sds_weight=1.0,  # ← Changed: placeholder, will be overridden in training_step
                    guidance_scale=phase3_cfg.sds.get('guidance_scale', 4.0),
                    start_iter=phase3_cfg.get('phase3_start_iter', 0),
                    end_iter=phase3_cfg.get('phase3_end_iter', 99999),
                    device='cuda'
                )
                logger.info(f"✓ GHOP SDS Loss Module initialized")

                # Store diagnostic config
                self.sds_diagnostics = phase3_cfg.get('diagnostics', {})
                self.log_sds_weight_every = self.sds_diagnostics.get('log_sds_weight_every', 100)

                # ============================================================
                # PHASE 3 SHAPE VERIFICATION TEST
                # ============================================================
                logger.info("\n" + "="*70)
                logger.info("PHASE 3 ARCHITECTURE VERIFICATION")
                logger.info("="*70)

                try:
                    # Create test tensors
                    # ✅ FIXED: Use self.grid_resolution for spatial dimensions
                    test_sdf = torch.randn(1, 1, self.grid_resolution,
                                                 self.grid_resolution,
                                                 self.grid_resolution, device='cuda')
                    test_hand = {
                        'pose': torch.randn(1, 48, device='cuda'),
                        'shape': torch.randn(1, 10, device='cuda'),
                        'trans': torch.randn(1, 3, device='cuda')
                    }

                    logger.info("Testing component dimensions:")

                    # Test 1: Hand field output
                    with torch.no_grad():
                        test_hand_field = self.hand_field_builder(hand_params=test_hand)
                        logger.info(f"  Hand field output: {test_hand_field.shape}")
                        # ✅ FIXED: Update expected shape to use self.grid_resolution
                        logger.info(f"    Expected: [1, 15, {self.grid_resolution}, {self.grid_resolution}, {self.grid_resolution}]")
                        if test_hand_field.shape[1] != 15:
                            logger.error(f"    ❌ Wrong channel count! Expected 15, got {test_hand_field.shape[1]}")
                        if test_hand_field.shape[2] != self.grid_resolution:
                            logger.error(f"    ❌ Wrong resolution! Expected {self.grid_resolution}, got {test_hand_field.shape[2]}")

                    # Test 2: VQ-VAE encode output
                    with torch.no_grad():
                        test_z0, _, _ = self.vqvae.encode(test_sdf, test_hand_field)
                        logger.info(f"  VQ-VAE latent output: {test_z0.shape}")
                        # ✅ FIXED: Calculate expected latent size dynamically
                        expected_latent_size = self.grid_resolution // 8  # VQ-VAE typically downsamples by 8x
                        logger.info(f"    Expected: [1, 3, {expected_latent_size}, {expected_latent_size}, {expected_latent_size}]")
                        if test_z0.shape[1] != 3:
                            logger.error(f"    ❌ Wrong channel count! Expected 3, got {test_z0.shape[1]}")

                    # Test 3: U-Net input (this is where the 23-channel error happens)
                    with torch.no_grad():
                        test_t = torch.tensor([500], device='cuda')
                        test_text = torch.zeros(1, 77, 768, device='cuda')

                        logger.info(f"  Calling U-Net with:")
                        logger.info(f"    z_t shape: {test_z0.shape}")
                        logger.info(f"    timestep: {test_t.shape}")
                        logger.info(f"    text_emb: {test_text.shape}")

                        try:
                            test_output = self.unet(test_z0, test_t, test_text)
                            logger.info(f"  ✅ U-Net output: {test_output.shape}")
                        except Exception as e:
                            logger.error(f"  ❌ U-Net FAILED: {e}")
                            logger.error(f"     This is the 23-channel error - U-Net wrapper concatenates incorrectly")

                            # Extract channel count from error message
                            if "got input of shape" in str(e):
                                import re
                                match = re.search(r'\[(\d+), (\d+), (\d+), (\d+), (\d+)\]', str(e))
                                if match:
                                    actual_channels = int(match.group(2))
                                    logger.error(f"     Actual channels entering U-Net: {actual_channels}")
                                    logger.error(f"     Required channels (divisible by 32): 32, 64, 96, ...")
                                    logger.error(f"     Delta: need +{32 - (actual_channels % 32)} channels")

                    logger.info("=" * 70 + "\n")
                except Exception as e:
                    logger.error(f"❌ Shape verification test failed: {e}")
                    import traceback
                    logger.error(traceback.format_exc())

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

                # ================================================================
                # CRITICAL: Verify GHOP Manager Initialization
                # ================================================================
                logger.info("\n" + "="*70)
                logger.info("GHOP MANAGER VERIFICATION")
                logger.info("="*70)

                if hasattr(self, 'ghop_manager') and self.ghop_manager is not None:
                    logger.info("✅ GHOP Manager: INITIALIZED")
                    logger.info(f"   Manager type: {type(self.ghop_manager).__name__}")
                    logger.info(f"   Has SDS loss module: {hasattr(self.ghop_manager, 'sds_loss')}")

                    # Verify sub-components
                    if hasattr(self, 'vqvae'):
                        logger.info(f"   VQ-VAE: {'✅ Present' if self.vqvae is not None else '❌ None'}")
                    if hasattr(self, 'unet'):
                        logger.info(f"   U-Net: {'✅ Present' if self.unet is not None else '❌ None'}")
                    if hasattr(self, 'hand_field_builder'):
                        logger.info(f"   Hand Field Builder: {'✅ Present' if self.hand_field_builder is not None else '❌ None'}")
                    if hasattr(self, 'sds_loss'):
                        logger.info(f"   SDS Loss: {'✅ Present' if self.sds_loss is not None else '❌ None'}")

                    logger.info("\n✅ GHOP READY FOR TRAINING")
                else:
                    logger.error("❌ GHOP Manager: NOT INITIALIZED!")
                    logger.error("   phase3_enabled = True but ghop_manager is None")
                    logger.error("   GHOP WILL NOT WORK DURING TRAINING!")

                logger.info("="*70 + "\n")
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

            # ================================================================
            # Store Phase 3 Hyperparameters for Training Step
            # ================================================================
            self.phase3_start_iter = phase3_cfg.get('phase3_start_iter', 0)
            self.phase3_end_iter = phase3_cfg.get('phase3_end_iter', 2000)  # ← ADD THIS LINE
            self.warmup_iters = phase3_cfg.get('warmup_iters', 0)
            self.sds_iters = phase3_cfg.get('sds_iters', 500)
            self.w_sds = phase3_cfg.get('w_sds', 5000.0)

            self.phase3_enabled = True
            self.ghop_enabled = True

            # ================================================================
            # CRITICAL: Log Phase 3 Training Parameters
            # ================================================================
            logger.info("\n" + "="*70)
            logger.info("PHASE 3 TRAINING PARAMETERS")
            logger.info("="*70)
            logger.info(f"  phase3_start_iter: {self.phase3_start_iter}")
            logger.info(f"  warmup_iters: {self.warmup_iters}")
            logger.info(f"  sds_iters: {self.sds_iters}")
            logger.info(f"  w_sds: {self.w_sds}")
            logger.info(f"  grid_resolution: {phase3_cfg.get('grid_resolution', 32)}")
            logger.info(f"  spatial_lim: {phase3_cfg.get('spatial_lim', 1.5)}")
            logger.info(f"  guidance_scale: {phase3_cfg.sds.get('guidance_scale', 4.0)}")
            logger.info("\n  🔍 GHOP will activate when:")
            logger.info(f"     global_step >= {self.phase3_start_iter}")
            logger.info(f"     AND global_step >= {self.warmup_iters}")
            logger.info(f"     AND global_step % {self.sds_iters} == 0")
            logger.info("="*70 + "\n")

        else:
            # No Phase 3 - initialize to None
            self.hold_loss_module = None
            self.ghop_manager = None
            self.vqvae = None
            self.unet = None
            self.hand_field_builder = None
            self.sds_loss = None

            # ================================================================
            # ADDED: Initialize parameters even when Phase 3 disabled
            # ================================================================
            self.phase3_start_iter = 0
            self.warmup_iters = 0
            self.sds_iters = 1
            self.w_sds = 0.0
            # ================================================================

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
            elif self.vqvae is None or self.hand_field_builder is None:
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

                # ============================================================
                # Initialize Mesh Extractor
                # ============================================================
                logger.info("Initializing GHOP Mesh Extractor...")
                self.mesh_resolution = phase4_cfg.get('mesh_resolution', 32)
                self.mesh_extractor = GHOPMeshExtractor(
                    vqvae_wrapper=self.vqvae,
                    resolution=self.mesh_resolution
                )
                logger.info(f"✓ Mesh extractor initialized (resolution: {self.mesh_resolution}³)")

                # ============================================================
                # Initialize Contact Refiner
                # ============================================================
                logger.info("Initializing GHOP Contact Refinement...")
                self.contact_refiner = GHOPContactRefinement(
                    contact_thresh=phase4_cfg.get('contact_thresh', 0.01),
                    collision_thresh=phase4_cfg.get('collision_thresh', 0.005),
                    contact_zones='zones'
                )
                logger.info("✓ Contact refiner initialized")

                # ============================================================
                # ✅ FIX: Set Phase 4 Boundaries (CONSOLIDATED - NO DUPLICATES)
                # ============================================================
                # Iteration boundaries
                self.contact_start_iter = phase4_cfg.get('contact_start_iter', 500)
                self.contact_end_iter = phase4_cfg.get('contact_end_iter', 999999)
                self.contact_warmup_iters = phase4_cfg.get('contact_warmup_iters', 100)

                # Calculate duration from boundaries (for logging/reference only)
                self.contact_duration = self.contact_end_iter - self.contact_start_iter

                # Loss weights
                self.w_contact = phase4_cfg.get('w_contact', 1.0)

                # Logging frequency
                self.log_contact_every = phase4_cfg.get('log_contact_every', 10)

                # Enable Phase 4
                self.phase4_enabled = True

                # Log configuration
                logger.info("\n✓ Phase 4 Boundaries Set:")
                logger.info(f"    - contact_start_iter:    {self.contact_start_iter}")
                logger.info(f"    - contact_end_iter:      {self.contact_end_iter}")
                logger.info(f"    - contact_duration:      {self.contact_duration} iterations")
                logger.info(f"    - contact_warmup_iters:  {self.contact_warmup_iters}")
                logger.info(f"    - w_contact:             {self.w_contact}")
                logger.info(f"    - log_contact_every:     {self.log_contact_every}")
                logger.info(f"    - mesh_resolution:       {self.mesh_resolution}³ voxels")
                logger.info(f"    - contact_thresh:        {phase4_cfg.get('contact_thresh', 0.01)}m")
                logger.info(f"    - collision_thresh:      {phase4_cfg.get('collision_thresh', 0.005)}m")
                logger.info(f"    - Active window:         steps [{self.contact_start_iter}, {self.contact_end_iter})")
                logger.info("=" * 70 + "\n")
        else:
            # ================================================================
            # Phase 4 Disabled - Initialize attributes to safe defaults
            # ================================================================
            self.phase4_enabled = False

            # Set iteration boundaries to never activate
            self.contact_start_iter = float('inf')
            self.contact_end_iter = float('inf')
            self.contact_duration = 0
            self.contact_warmup_iters = 0

            # Set weights and parameters
            self.w_contact = 0.0
            self.mesh_resolution = 32
            self.log_contact_every = 50

            # Set component references to None
            self.mesh_extractor = None
            self.contact_refiner = None

            logger.info("[Phase 4] Disabled - configure phase4.enabled=true in config to enable\n")

        # ============================================================
        # OBJECT TEMPLATE CHAMFER: Load Templates
        # ============================================================
        loss_config = getattr(self.opt, 'loss', None)
        obj_chamfer_enabled = getattr(loss_config, 'w_obj_chamfer', 0.0) > 0 if loss_config else False
        if obj_chamfer_enabled:
            logger.info("=" * 70)
            logger.info("OBJECT TEMPLATE CHAMFER: Loading Templates")
            logger.info("=" * 70)

            self.object_templates = self._load_object_templates()

            if len(self.object_templates) == 0:
                logger.error("Object Chamfer enabled but no templates loaded!")
                logger.error("Set loss.w_obj_chamfer=0 or add templates to data/templates/")
            else:
                logger.info(f"Available categories: {list(self.object_templates.keys())}")

            logger.info("=" * 70)
        else:
            self.object_templates = {}
            logger.info("Object Template Chamfer: Disabled (w_obj_chamfer=0)")

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
            elif self.vqvae is None or self.hand_field_builder is None:
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

                # ============================================================
                # Component 1: Diffusion Prior for Geometry Guidance
                # ============================================================
                logger.info("Initializing GHOP Diffusion Prior...")
                self.diffusion_prior = GHOPDiffusionPrior(
                    vqvae_wrapper=self.vqvae,
                    unet_wrapper=self.unet,
                    handfield_builder=self.hand_field_builder,
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

                # Initialize temporal diagnostic tool
                self.temporal_diagnostic = TemporalMemoryDiagnostic()
                self._diagnostic_enabled = True
                logger.info("[Phase 5] Temporal diagnostic tool initialized")

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
                logger.info(f"  ✓ Adaptive contacts initialized (threshold={phase5_cfg.get('proximity_threshold', 0.015)}m)")

                # ============================================================
                # ✅ FIX: Store Phase 5 hyperparameters FIRST
                # These must be set regardless of scheduler configuration
                # ============================================================
                self.phase5_start_iter = phase5_cfg.get('phase5_start_iter', 8000)
                self.w_temporal = phase5_cfg.get('w_temporal', 1.0)
                self.log_phase5_every = phase5_cfg.get('log_phase5_every', 50)
                self.enable_geometry_sampling = phase5_cfg.get('enable_geometry_sampling', False)
                self.phase5_enabled = True

                logger.info("  Phase 5 Hyperparameters Set:")
                logger.info(f"    - phase5_start_iter: {self.phase5_start_iter}")
                logger.info(f"    - w_temporal: {self.w_temporal}")
                logger.info(f"    - log_phase5_every: {self.log_phase5_every}")
                logger.info(f"    - enable_geometry_sampling: {self.enable_geometry_sampling}")

                # ============================================================
                # Component 4: Phase 5 Training Scheduler
                # ============================================================
                logger.info("Initializing Phase 5 Training Scheduler...")
                use_scheduler = phase5_cfg.get('use_scheduler', False)

                if not use_scheduler:
                    logger.info("  [Phase5Scheduler] Disabled (use_scheduler=false)")
                    logger.info("  Using boundary-based phase switching for two-stage training")
                    self.phase5_scheduler = None
                else:
                    # ============================================================
                    # CRITICAL FIX: Read scheduler config from correct nested level
                    # Config structure: phase5 -> scheduler -> {keys}
                    # ============================================================
                    scheduler_cfg = phase5_cfg.get('scheduler', {})

                    # Read timing parameters from scheduler config
                    warmup_iters = scheduler_cfg.get('warmup_iters', 0)
                    phase3_start = scheduler_cfg.get('phase3_start', opt.phase3.get('phase3_start_iter', 0))
                    phase4_start = scheduler_cfg.get('phase4_start', opt.phase4.get('contact_start_iter', 999999))
                    phase5_start = scheduler_cfg.get('phase5_start', self.phase5_start_iter)
                    finetune_start = scheduler_cfg.get('finetune_start', 50000)
                    total_iterations = scheduler_cfg.get('total_iterations', 60000)

                    # Log loaded configuration
                    logger.info("  [Phase5Scheduler] Configuration loaded from config:")
                    logger.info(f"    - total_iterations:  {total_iterations:>6d}")
                    logger.info(f"    - warmup_iters:      {warmup_iters:>6d}")
                    logger.info(f"    - phase3_start:      {phase3_start:>6d}")
                    logger.info(f"    - phase4_start:      {phase4_start:>6d}")
                    logger.info(f"    - phase5_start:      {phase5_start:>6d}")
                    logger.info(f"    - finetune_start:    {finetune_start:>6d}")

                    # Validate phase ordering
                    if not (warmup_iters <= phase3_start <= phase4_start <= phase5_start <= finetune_start <= total_iterations):
                        logger.error(
                            f"[Phase5Scheduler] ❌ Invalid phase ordering:\n"
                            f"  warmup_iters={warmup_iters}\n"
                            f"  phase3_start={phase3_start}\n"
                            f"  phase4_start={phase4_start}\n"
                            f"  phase5_start={phase5_start}\n"
                            f"  finetune_start={finetune_start}\n"
                            f"  total_iterations={total_iterations}\n"
                            f"\n  Required: warmup <= phase3 <= phase4 <= phase5 <= finetune <= total"
                        )
                        raise ValueError("Invalid Phase5Scheduler configuration - see error above")

                    # Initialize scheduler with validated parameters
                    self.phase5_scheduler = Phase5TrainingScheduler(
                        total_iterations=total_iterations,
                        warmup_iters=warmup_iters,
                        phase3_start=phase3_start,
                        phase4_start=phase4_start,
                        phase5_start=phase5_start,
                        finetune_start=finetune_start
                    )
                    logger.info("  ✓ Phase5Scheduler initialized successfully")
                    if self.phase5_scheduler is not None:
                        # Log the initialization message
                        logger.info("=" * 70)
                        logger.info("PHASE 5 SCHEDULER INITIALIZED")
                        logger.info("=" * 70)
                        logger.info(f"  Total iterations: {phase5_cfg.scheduler.total_iterations}")
                        logger.info(f"  Phase 3 (SDS) start: {phase5_cfg.scheduler.phase3_start}")
                        logger.info(f"  Phase 4 (Contact) start: {phase5_cfg.scheduler.phase4_start}")
                        logger.info(f"  Phase 5 (Temporal) start: {phase5_cfg.scheduler.phase5_start}")
                        logger.info(f"  Fine-tuning start: {phase5_cfg.scheduler.finetune_start}")
                        logger.info("=" * 70)
                # ============================================================
                # ✅ FIX: Verify Phase Boundaries (ALL PHASES)
                # ============================================================
                logger.info("\n" + "=" * 70)
                logger.info("[Phase Boundaries] Verification")
                logger.info("=" * 70)

                # Phase 3 boundaries
                if hasattr(self, 'phase3_start_iter') and hasattr(self, 'phase3_end_iter'):
                    logger.info(f"  Phase 3 (SDS):")
                    logger.info(f"    Start:  {self.phase3_start_iter}")
                    logger.info(f"    End:    {self.phase3_end_iter}")
                    logger.info(f"    Active: [{self.phase3_start_iter}, {self.phase3_end_iter})")
                    logger.info("")
                else:
                    logger.warning("  ⚠️  Phase 3 boundaries not set")

                # Phase 4 boundaries (NEW CHECK)
                if hasattr(self, 'contact_start_iter') and hasattr(self, 'contact_end_iter'):
                    logger.info(f"  Phase 4 (Contact):")
                    logger.info(f"    Start:  {self.contact_start_iter}")
                    logger.info(f"    End:    {self.contact_end_iter}")
                    logger.info(f"    Active: [{self.contact_start_iter}, {self.contact_end_iter})")
                    logger.info(f"    Warmup: {self.contact_warmup_iters} iters")
                    logger.info("")
                else:
                    logger.error("  ❌ Phase 4 boundaries NOT SET - Phase 4 will not activate!")
                    logger.error("     This is a critical configuration error.")

                # Phase 5 boundaries
                if hasattr(self, 'phase5_start_iter'):
                    logger.info(f"  Phase 5 (Temporal):")
                    logger.info(f"    Start:  {self.phase5_start_iter}")
                    logger.info(f"    Active: [{self.phase5_start_iter}, ∞)")
                    logger.info("")
                else:
                    logger.warning("  ⚠️  Phase 5 boundaries not set")

                # Check for phase overlaps/gaps
                if hasattr(self, 'phase3_end_iter') and hasattr(self, 'contact_start_iter'):
                    gap_3_4 = self.contact_start_iter - self.phase3_end_iter
                    if gap_3_4 == 0:
                        logger.info(f"  ✅ Phase 3 → Phase 4 transition: seamless (no gap)")
                    elif gap_3_4 > 0:
                        logger.warning(f"  ⚠️  Phase 3 → Phase 4 gap: {gap_3_4} iterations")
                    else:
                        logger.warning(f"  ⚠️  Phase 3 and Phase 4 overlap: {abs(gap_3_4)} iterations")

                if hasattr(self, 'contact_end_iter') and hasattr(self, 'phase5_start_iter'):
                    overlap_4_5 = self.contact_end_iter - self.phase5_start_iter
                    if overlap_4_5 > 0:
                        logger.info(f"  ✅ Phase 4 and Phase 5 overlap: {overlap_4_5} iterations (intentional)")
                    elif overlap_4_5 == 0:
                        logger.info(f"  ✅ Phase 4 → Phase 5 transition: seamless")
                    else:
                        logger.warning(f"  ⚠️  Phase 4 → Phase 5 gap: {abs(overlap_4_5)} iterations")

                if hasattr(self, 'phase3_end_iter') and hasattr(self, 'phase5_start_iter'):
                    if self.phase3_end_iter < self.phase5_start_iter:
                        logger.info(f"  ✅ Phase 3 and Phase 5 are MUTUALLY EXCLUSIVE")
                    else:
                        logger.error(f"  ❌ Phase 3 and Phase 5 OVERLAP (mutual exclusivity violated!)")

                logger.info("=" * 70 + "\n")
        else:
            self.phase5_enabled = False
            self.diffusion_prior = None
            self.temporal_module = None
            self.adaptive_contacts = None
            self.phase5_scheduler = None
            self.phase5_start_iter = 0
            self.w_temporal = 0.0
            self.log_phase5_every = 0
            self.enable_geometry_sampling = False
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

        # ✅ ADD: Memory profiler for diagnostics
        self.memory_profiler = MemoryProfiler()
        self.profile_memory = True  # Set to False to disable
        # ✅ ADD: Setup and validate phase boundaries
        self.setup_phase_boundaries()

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

        # ================================================================
        # NEW: Geometry-share controller state
        # ================================================================
        # Target fraction of total loss that geometry should occupy in Phase 3
        self.target_geom_share = getattr(getattr(self.opt, 'loss', {}), 'target_geom_share', 0.05)
        self.max_geom_scale = getattr(getattr(self.opt, 'loss', {}), 'max_geom_scale', 5.0)
        self.geom_boost_factor = getattr(getattr(self.opt, 'loss', {}), 'geom_boost_factor', 1.5)

        # Runtime state
        self._geom_scale = 1.0
        self._geom_share_ema = 0.0
        self._geom_share_ema_momentum = 0.9
        if self._geom_share_ema < self.target_geom_share:
            old_scale = self._geom_scale
            self._geom_scale = min(self._geom_scale * self.geom_boost_factor,
                                   self.max_geom_scale)
            logger.info(
                f"[Geom-Controller] step={self.global_step} "
                f"share={self._geom_share_ema:.4f} < target={self.target_geom_share:.4f} → "
                f"scale {old_scale:.3f} → {self._geom_scale:.3f}"
            )

        '''
        Object Chamfer Debugging Section Temporary
        '''
        if self.global_step == 0 and len(self.object_templates) > 0:
            logger.info("=" * 70)
            logger.info("OBJECT TEMPLATE CHAMFER: Diagnostic")
            logger.info("=" * 70)

            for category, verts in self.object_templates.items():
                logger.info(f"  {category}: {verts.shape[0]} vertices, device={verts.device}")

            # Test Chamfer distance computation
            try:
                from pytorch3d.loss import chamfer_distance
                test_verts = torch.randn(100, 3, device=self.device)
                test_template = list(self.object_templates.values())[0]
                test_chamfer, _ = chamfer_distance(
                    test_verts.unsqueeze(0),
                    test_template.unsqueeze(0)
                )
                logger.info(f"  Test Chamfer computation: {test_chamfer.item():.6f} ✅")
            except Exception as e:
                logger.error(f"  Test Chamfer FAILED: {e}")

            logger.info("=" * 70)

        # Logging / export intervals
        self.save_misc_every = getattr(self.opt.logging, "save_misc_every", 100)
        logger.info(f"save_misc_every: {self.save_misc_every}")

    def _load_object_templates(self):
        """
        Load preprocessed object templates for Chamfer loss.
        Called during initialization if object Chamfer loss is enabled.
        """
        import os
        from pathlib import Path

        template_dir = Path("data/templates")

        if not template_dir.exists():
            logger.warning(f"Template directory {template_dir} not found. Object Chamfer disabled.")
            return {}

        templates = {}
        template_files = list(template_dir.glob("*.pt"))

        if len(template_files) == 0:
            logger.warning(f"No .pt template files found in {template_dir}")
            return {}

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for template_path in template_files:
            category = template_path.stem  # e.g., 'mug', 'bottle'

            try:
                # Load normalized vertices
                vertices = torch.load(template_path, map_location=device)

                # Validate shape
                if vertices.dim() != 2 or vertices.shape[1] != 3:
                    logger.warning(f"Invalid template shape for {category}: {vertices.shape}")
                    continue

                templates[category] = vertices
                logger.debug(f"  Loaded template '{category}': {vertices.shape[0]} vertices")

            except Exception as e:
                logger.warning(f"Failed to load template {category}: {e}")

        logger.info(f"Loaded {len(templates)} object templates from {template_dir}")
        return templates

    def setup_phase_boundaries(self):
        """Validate and log phase boundaries (values already set from config)."""

        # ================================================================
        # All boundaries should already be set by phase init sections above
        # This method only validates and logs them
        # ================================================================

        # Get phase enable flags (already set)
        self.phase3_enabled = getattr(self, 'phase3_enabled', False)
        self.phase4_enabled = getattr(self, 'phase4_enabled', False)
        self.phase5_enabled = getattr(self, 'phase5_enabled', False)

        # Validate Phase 3 boundaries
        if not hasattr(self, 'phase3_start_iter'):
            logger.warning("[Phase Boundaries] phase3_start_iter not set, defaulting to 0")
            self.phase3_start_iter = 0

        if not hasattr(self, 'phase3_end_iter'):
            logger.warning("[Phase Boundaries] phase3_end_iter not set, defaulting to 2000")
            self.phase3_end_iter = 2000

        # Validate Phase 4 boundaries (should be set by Phase 4 init)
        if not hasattr(self, 'contact_start_iter'):
            logger.warning("[Phase Boundaries] contact_start_iter not set, defaulting to 2000")
            self.contact_start_iter = 2000

        if not hasattr(self, 'contact_end_iter'):
            logger.warning("[Phase Boundaries] contact_end_iter not set, defaulting to 999999")
            self.contact_end_iter = 999999

        # Validate Phase 5 boundaries (should be set by Phase 5 init)
        if not hasattr(self, 'phase5_start_iter'):
            logger.warning("[Phase Boundaries] phase5_start_iter not set, defaulting to 2100")
            self.phase5_start_iter = 2100

        # ================================================================
        # Log all phase boundaries
        # ================================================================
        logger.info("[Phase Boundaries] Final Configuration:")

        if self.phase3_enabled:
            logger.info(f"  Phase 3 (SDS): [{self.phase3_start_iter}, {self.phase3_end_iter})")

        if self.phase4_enabled:
            logger.info(f"  Phase 4 (Contact): [{self.contact_start_iter}, {self.contact_end_iter})")

        if self.phase5_enabled:
            logger.info(f"  Phase 5 (Temporal): [{self.phase5_start_iter}, ∞)")

        # ================================================================
        # Validate Phase 3 vs Phase 4 boundaries
        # ================================================================
        if self.phase3_enabled and self.phase4_enabled:
            # Check if Phase 3 is meant to run continuously (end_iter >> contact_start)
            phase3_continuous = self.phase3_end_iter > 100000  # Arbitrary large threshold

            if phase3_continuous:
                # Phase 3 runs throughout - this is intentional overlap
                logger.info(
                    f"  Phase 3 runs continuously (end_iter={self.phase3_end_iter})\n"
                    f"  Phase 4 starts at {self.contact_start_iter} (both active simultaneously)\n"
                    f"  ✅ Intentional overlap - scheduler will control weights"
                )
            else:
                # Phase 3 has explicit end - check for unintended overlap
                if self.phase3_end_iter >= self.contact_start_iter:
                    logger.error(
                        f"[Phase Boundaries] ❌ OVERLAP DETECTED!\n"
                        f"  Phase 3 ends at:   {self.phase3_end_iter}\n"
                        f"  Phase 4 starts at: {self.contact_start_iter}\n"
                        f"  This causes conflicts!\n"
                        f"  Either:\n"
                        f"    1. Set phase3_end_iter < {self.contact_start_iter}\n"
                        f"    2. Set phase3_end_iter = 999999 (continuous mode)"
                    )
                    raise ValueError("Phase boundaries overlap - see error above")

            # Check for gaps
            gap = self.contact_start_iter - self.phase3_end_iter
            if gap == 1:
                logger.info(f"  ✅ Phase 3 → Phase 4: Seamless transition (no gap)")
            elif gap > 1:
                logger.warning(
                    f"  ⚠️  Phase 3 → Phase 4: {gap - 1} iteration gap\n"
                    f"     Steps {self.phase3_end_iter} to {self.contact_start_iter - 1} have no guidance"
                )

        # ================================================================
        # Check Phase 3 vs Phase 5 overlap
        # ================================================================
        phase3_overlap_with_phase5 = (
            self.phase3_enabled and
            self.phase5_enabled and
            self.phase3_end_iter >= self.phase5_start_iter
        )

        if phase3_overlap_with_phase5:
            # Check if Phase 3 is in continuous mode (intentional overlap)
            phase3_continuous = self.phase3_end_iter > 100000

            if phase3_continuous:
                # Phase 3 runs throughout - intentional overlap
                phase3_overlap_duration = self.phase3_end_iter - self.phase5_start_iter
                logger.info(
                    f"  ✅ Phase 3 and Phase 5 overlap: {phase3_overlap_duration} iterations (intentional)\n"
                    f"     Phase 3 (SDS) runs continuously for guidance\n"
                    f"     Phase 5 (Temporal) adds temporal consistency on top\n"
                    f"     Scheduler controls relative weights"
                )
            else:
                # Phase 3 has explicit end - overlap is likely a mistake
                logger.error(
                    f"[Phase Boundaries] ❌ MUTUAL EXCLUSIVITY VIOLATED!\n"
                    f"  Phase 3 ends at:   {self.phase3_end_iter}\n"
                    f"  Phase 5 starts at: {self.phase5_start_iter}\n"
                    f"  Phase 3 and Phase 5 overlap!\n"
                    f"  Either:\n"
                    f"    1. Set phase3_end_iter < {self.phase5_start_iter} (clean transition)\n"
                    f"    2. Set phase3_end_iter = 999999 (continuous SDS mode)"
                )
                raise ValueError(
                    f"Phase 3 and Phase 5 overlap detected. "
                    f"Set phase3_end_iter < {self.phase5_start_iter} or use continuous mode (999999)."
                )
        else:
            logger.info(f"  ✅ Phase 3 and Phase 5 are mutually exclusive (no overlap)")

        logger.info("[Phase Boundaries] ✅ Validation complete\n")

    def _validate_ghop_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Validate GHOP checkpoint contains required components.

        Args:
            checkpoint_path: Path to the checkpoint file

        Returns:
            bool: True if valid, raises exception if invalid
        """

        # ============================================================
        # Step 1: File existence and size check
        # ============================================================
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"[GHOP Validation] Checkpoint not found: {checkpoint_path}"
            )

        file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        logger.info(f"[GHOP Validation] Checkpoint size: {file_size_mb:.1f} MB")

        if file_size_mb < 10:
            logger.warning(
                f"[GHOP Validation] Checkpoint is very small ({file_size_mb:.1f} MB). "
                f"Expected >100 MB for full model with optimizer state."
            )

        # ============================================================
        # Step 2: Load and inspect state dict
        # ============================================================
        try:
            # ckpt = torch.load(checkpoint_path, map_location='cpu')
            ckpt = patch_checkpoint_for_film(checkpoint_path, new_cond_dim=128)
            logger.info("[GHOP Validation] Checkpoint loaded successfully")
        except Exception as e:
            raise RuntimeError(f"[GHOP Validation] Failed to load checkpoint: {e}")

        # Extract state dict (handle both direct and nested formats)
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        elif isinstance(ckpt, dict) and any(k.startswith('ae.model.') or k.startswith('glide_model.') for k in ckpt.keys()):
            state_dict = ckpt
        else:
            raise ValueError(
                f"[GHOP Validation] Checkpoint format unrecognized. "
                f"Top-level keys: {list(ckpt.keys())}"
            )

        # ============================================================
        # Step 3: Check for required key prefixes (architecture-agnostic)
        # ============================================================
        vqvae_keys = [k for k in state_dict.keys() if k.startswith('ae.model.')]
        unet_keys = [k for k in state_dict.keys() if k.startswith('glide_model.')]
        hold_keys = [k for k in state_dict.keys() if 'implicit' in k or 'nodes' in k]

        logger.info(f"[GHOP Validation] Key counts:")
        logger.info(f"  - VQ-VAE (ae.model.*):     {len(vqvae_keys)}")
        logger.info(f"  - U-Net (glide_model.*):   {len(unet_keys)}")
        logger.info(f"  - HOLD contamination:      {len(hold_keys)}")

        # ============================================================
        # Step 4: Validation checks
        # ============================================================
        issues = []

        # Check 1: VQ-VAE must be present
        if len(vqvae_keys) == 0:
            issues.append("❌ No VQ-VAE parameters found (ae.model.* keys missing)")
        elif len(vqvae_keys) < 50:
            issues.append(f"⚠️  Very few VQ-VAE parameters ({len(vqvae_keys)}). Expected >50.")
        else:
            logger.info(f"✅ VQ-VAE parameters present ({len(vqvae_keys)} keys)")

        # Check 2: U-Net must be present
        if len(unet_keys) == 0:
            issues.append("❌ No U-Net parameters found (glide_model.* keys missing)")
        elif len(unet_keys) < 100:
            issues.append(f"⚠️  Very few U-Net parameters ({len(unet_keys)}). Expected >100.")
        else:
            logger.info(f"✅ U-Net parameters present ({len(unet_keys)} keys)")

        # Check 3: HOLD contamination
        if len(hold_keys) > 0:
            issues.append(
                f"⚠️  Found {len(hold_keys)} HOLD-related keys in checkpoint. "
                f"This might be a HOLD checkpoint, not GHOP!"
            )
        else:
            logger.info("✅ No HOLD contamination detected")

        # Check 4: Sample a few critical parameters to verify they're tensors
        sample_keys = []
        if vqvae_keys:
            sample_keys.append(vqvae_keys[0])
        if unet_keys:
            sample_keys.append(unet_keys[0])

        for key in sample_keys[:3]:
            param = state_dict[key]
            if not isinstance(param, torch.Tensor):
                issues.append(f"❌ Parameter '{key}' is not a tensor (type: {type(param)})")
            else:
                logger.debug(f"✅ Sampled '{key}': shape={param.shape}, dtype={param.dtype}")

        # ============================================================
        # Step 5: Report validation result
        # ============================================================
        if issues:
            error_msg = "[GHOP Validation] Checkpoint validation FAILED:\n" + "\n".join(issues)
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("=" * 70)
        logger.info("✅ GHOP CHECKPOINT VALIDATION PASSED")
        logger.info("=" * 70)
        logger.info(f"  Checkpoint: {os.path.basename(checkpoint_path)}")
        logger.info(f"  Size: {file_size_mb:.1f} MB")
        logger.info(f"  VQ-VAE params: {len(vqvae_keys)}")
        logger.info(f"  U-Net params: {len(unet_keys)}")
        logger.info("=" * 70)

        return True

    def patch_checkpoint_for_film(self, checkpoint, new_cond_dim=128):
        """
        Patch checkpoint dict to strip implicit network weights for FiLM reinit.
        Preserves hand model, encoder, and residual MLP.
        """
        if 'state_dict' not in checkpoint:
            print("[Checkpoint Patch] No state_dict found, returning as-is")
            return checkpoint

        state_dict = checkpoint['state_dict']

        # Keys to remove (implicit network weights)
        # These patterns match typical implicit network naming
        implicit_keys = [
            k for k in state_dict.keys()
            if any(x in k for x in ['implicit_network', 'object_mlp', 'shape_net', 'object.implicit'])
        ]

        # Also remove object-specific MLP layers if named differently
        film_keys = [
            k for k in state_dict.keys()
            if any(x in k for x in ['film_gamma', 'film_beta', 'film_mlp'])
        ]

        keys_to_remove = list(set(implicit_keys + film_keys))

        if len(keys_to_remove) > 0:
            print(f"[Checkpoint Patch] Removing {len(keys_to_remove)} implicit network keys:")
            for k in keys_to_remove:
                print(f"  - {k}")
                del state_dict[k]
            checkpoint['state_dict'] = state_dict
            print(f"[Checkpoint Patch] Preserved {len(state_dict)} keys")

        return checkpoint

    def on_load_checkpoint(self, checkpoint):
        """
        Called when loading checkpoint. Patch to reinitialize implicit network with FiLM.
        """
        # Apply the patch to strip old implicit network weights
        checkpoint = self.patch_checkpoint_for_film(checkpoint)

        # Now load the checkpoint properly
        super().on_load_checkpoint(checkpoint)

        # Force FiLM configuration after checkpoint load (redundant but safe)
        if hasattr(self.model, 'obj_implicit_network'):
            net = self.model.obj_implicit_network
            net.use_film = True
            net.cond_dim = 0  # No concatenation at input

            # Reinitialize FiLM layers with correct film_cond_dim
            if hasattr(net, 'film_cond_dim') and hasattr(net, 'film_gamma'):
                import torch.nn as nn
                for l in range(len(net.film_gamma)):
                    out_dim = net.film_gamma[l].out_features
                    device = net.film_gamma[l].weight.device
                    net.film_gamma[l] = nn.Linear(net.film_cond_dim, out_dim).to(device)
                    net.film_beta[l] = nn.Linear(net.film_cond_dim, out_dim).to(device)
                logger.info(f"[Checkpoint Load] Reinitialized FiLM with film_cond_dim={net.film_cond_dim}")

    def save_misc(self):
        """Save miscellaneous outputs (meshes, camera params, etc.)."""
        out = {}

        # ================================================================
        # FIX: Handle GHOPHOIDataset which doesn't have nested .dataset
        # ================================================================
        # Check if trainset has nested .dataset attribute
        if hasattr(self.trainset, 'dataset'):
            # Standard HOLD datasets (Subset wrapper)
            dataset = self.trainset.dataset
        else:
            # GHOP HOI4D dataset (no wrapper)
            dataset = self.trainset

        # ================================================================
        # Extract camera intrinsics
        # ================================================================
        if hasattr(dataset, 'intrinsics_all'):
            # Standard HOLD format
            K = dataset.intrinsics_all[0]
        elif hasattr(dataset, 'intrinsics'):
            # GHOP format: [N, 4, 4]
            K = dataset.intrinsics[0]
        else:
            # Fallback: identity matrix
            logger.warning("[save_misc] No intrinsics found, using identity")
            K = torch.eye(4)

        # ================================================================
        # Extract camera extrinsics (world-to-camera)
        # ================================================================
        if hasattr(dataset, 'extrinsics_all'):
            # Standard HOLD format
            w2c = dataset.extrinsics_all[0]
        elif hasattr(dataset, 'c2w'):
            # GHOP format: camera-to-world [N, 4, 4]
            # Need to invert to get world-to-camera
            c2w = dataset.c2w[0]
            w2c = torch.inverse(c2w)
        else:
            # Fallback: identity matrix
            logger.warning("[save_misc] No extrinsics found, using identity")
            w2c = torch.eye(4)

        # ================================================================
        # Extract object scale from nodes
        # ================================================================
        for node in self.model.nodes.values():
            if "object" in node.node_id:
                if hasattr(node.server, 'object_model') and hasattr(node.server.object_model, 'obj_scale'):
                    # Store under official key used in evaluation
                    out["obj_scale"] = node.server.object_model.obj_scale

        # ================================================================
        # Extract image paths
        # ================================================================
        if hasattr(dataset, 'img_paths'):
            # Standard HOLD format
            out["img_paths"] = dataset.img_paths
        elif hasattr(dataset, 'image_files'):
            # GHOP format: List of Path objects
            out["img_paths"] = [str(p) for p in dataset.image_files]
        else:
            # No image paths available
            logger.warning("[save_misc] No image paths found")
            out["img_paths"] = []

        # ================================================================
        # Extract scale
        # ================================================================
        if hasattr(dataset, 'scale'):
            out["scale"] = dataset.scale
        else:
            # Default scale
            out["scale"] = 1.0

        # Save camera params
        out["K"] = K
        out["w2c"] = w2c

        # Generate canonical meshes
        mesh_dict = self.meshing_cano(self.global_step)

        # Map to official HOLD keys expected by evaluation
        # 'object_cano' and 'right_cano' come from meshing_cano:
        #   mesh_dict[f"{node.node_id}_cano"] = mesh_c
        object_cano = mesh_dict.get("object_cano", None)
        right_cano = mesh_dict.get("right_cano", None)
        # Validate object mesh before saving
        if object_cano is not None and hasattr(object_cano, "faces"):
            if object_cano.faces is None or object_cano.faces.shape[0] == 0:
                logger.error(
                    f"[save_misc] Step {self.global_step}: mesh_c_o has zero or no faces; "
                    f"canonical extraction failed. Evaluation ICP will not be valid."
                )
                # Option 1: drop it so eval can detect missing mesh
                object_cano = None
        out["mesh_c_o"] = object_cano
        out["mesh_c_h"] = right_cano

        # Optionally keep the original keys too (harmless for evaluation)
        out.update(mesh_dict)
        # ⚠️ CRITICAL FIX: Restore model to training mode after meshing
        self.model.train()

        # Save to file
        out_p = f"{self.args.log_dir}/misc/{self.global_step:09d}.npy"
        os.makedirs(op.dirname(out_p), exist_ok=True)
        np.save(out_p, out)
        logger.info(f"Exported misc to {out_p}")

    def configure_optimizers(self):
        base_lr = self.args.lr
        mano_lr_multiplier = getattr(self.opt.training, 'mano_lr_multiplier', 0.1) \
            if hasattr(self.opt, 'training') else 0.1
        logger.info(f"[Optimizer] Base LR: {base_lr}, MANO LR multiplier: {mano_lr_multiplier}")

        params = []
        node_params = set()

        # ================================================================
        # GROUP 1: MANO Hand Parameters (Original LR)
        # ================================================================
        for node in self.model.nodes.values():
            node_parameters = set(node.params.parameters())
            node_params.update(node_parameters)
            params.append({
                "params": list(node_parameters),
                "lr": base_lr * mano_lr_multiplier,
                "name": f"mano_{node.node_id}"
            })

        # ================================================================
        # GROUP 2: Object Vertices - BRUTE FORCE SEARCH
        # ================================================================
        object_vertices_added = False

        # Search through ALL model parameters to find v3d_cano
        for name, param in self.named_parameters():
            if 'v3d_cano' in name and isinstance(param, nn.Parameter):
                params.append({
                    "params": [param],
                    "lr": base_lr * 1.0,  # Same as base_lr for vertex optimization
                    "name": "object_vertices"
                })
                node_params.add(param)
                object_vertices_added = True
                logger.info(f"✓ Object vertices added to optimizer: {name} ({param.numel():,} params at lr={base_lr * 1.0:.2e})")
                break  # Only add once

        if not object_vertices_added:
            logger.warning("⚠️ Object vertices NOT found in model parameters")
            logger.warning("   This means the object shape is FROZEN and cannot learn")

        # ================================================================
        # GROUP 3: Neural Network Parameters (Original LR)
        # ================================================================
        main_params = [p for p in self.model.parameters() if p not in node_params]
        if main_params:
            params.append({
                "params": main_params,
                "lr": base_lr,
                "name": "neural_networks"
            })

        # ================================================================
        # Create Optimizer
        # ================================================================
        self.optimizer = optim.Adam(params, eps=1e-8)

        # Log parameter groups
        logger.info("=" * 70)
        logger.info("Optimizer Parameter Groups:")
        for i, group in enumerate(params):
            num_params = sum(p.numel() for p in group['params'])
            logger.info(f"  [{i+1}] {group.get('name', 'unnamed'):20s}: lr={group['lr']:.2e}, params={num_params:,}")
        logger.info("=" * 70)

        return [self.optimizer], []

    def condition_training(self):
        import common.torch_utils as torch_utils

        if self.global_step in []:
            logger.info(f"Decaying learning rate at step {self.global_step}")
            torch_utils.decay_lr(self.optimizer, gamma=0.5)

    def training_step(self, batch, batch_idx):
        """Training step with Phase 3 GHOP + Phase 4 Contact + Phase 5 Advanced integration."""
        logger.info(f"[DEBUG BATCH] Keys: {list(batch.keys())}")

        # ================================================================
        # ✅ FIX: Correctly access Embedding.weight.requires_grad
        # ================================================================
        if self.global_step % 100 == 0:
            pose_param = self.model.nodes['right'].params.pose.weight
            transl_param = self.model.nodes['right'].params.transl.weight
            betas_param = self.model.nodes['right'].params.betas.weight

            logger.warning(f"[GRAD CHECK - Step {self.global_step}]")
            logger.warning(f"  pose.weight.requires_grad = {pose_param.requires_grad}")
            logger.warning(f"  transl.weight.requires_grad = {transl_param.requires_grad}")
            logger.warning(f"  betas.weight.requires_grad = {betas_param.requires_grad}")
            logger.warning(f"  pose.weight.shape = {pose_param.shape}")

        if 'rgb' in batch:
            logger.info(f"[DEBUG BATCH] rgb shape: {batch['rgb'].shape}")

        # ====================================================================
        # INITIALIZATION: Verify critical phase variables exist
        # ====================================================================
        # Ensure phase boundaries are set
        if not hasattr(self, 'phase3_end_iter'):
            # self.phase3_end_iter = getattr(self, 'phase5_start_iter', 99999) - 1
            logger.debug(f"[Init] Set phase3_end_iter to {self.phase3_end_iter}")

        if not hasattr(self, 'phase5_start_iter'):
            self.phase5_start_iter = 99999
            logger.debug(f"[Init] Set phase5_start_iter to {self.phase5_start_iter}")

        # Get device
        device = next(self.parameters()).device

        # Verify optimizer exists
        if not hasattr(self, 'optimizer') or self.optimizer is None:
            logger.warning("[Step 0] Optimizer not set, will use trainer's optimizer")

        # ================================================================
        # ✅ MEMORY PROFILER: Check if profiling is enabled
        # ================================================================
        should_profile = (
                hasattr(self, 'memory_profiler') and
                hasattr(self, 'profile_memory') and
                self.profile_memory and
                batch_idx % 10 == 0
        )

        if should_profile:
            self.memory_profiler.clear()
            self.memory_profiler.checkpoint("start")

        # ================================================================
        # ✅ OPTIMIZED: Reduced frequency cache clearing (every 50 steps)
        # REMOVED: Aggressive every-step clearing that caused overhead
        # ================================================================
        if batch_idx % 10 == 0 and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            # Log memory state
            allocated = torch.cuda.memory_allocated() / 1024 ** 2
            reserved = torch.cuda.memory_reserved() / 1024 ** 2
            logger.debug(
                f"[Epoch {self.current_epoch}, Iter {batch_idx}] "
                f"Memory checkpoint: Allocated={allocated:.1f}MB, Reserved={reserved:.1f}MB"
            )

        if should_profile:
            self.memory_profiler.checkpoint("after_initial_check")

        # PHASE 5: Dynamic Loss Weight Scheduling
        if self.phase5_enabled and self.phase5_scheduler is not None:
            # Scheduler enabled - use dynamic progressive weights
            loss_weights = self.phase5_scheduler.get_loss_weights(self.global_step)
            lr_multiplier = self.phase5_scheduler.get_learning_rate_multiplier(self.global_step)

            # Apply learning rate adjustment
            for param_group in self.optimizers().param_groups:
                base_lr = param_group.get('initial_lr', param_group['lr'])
                param_group['lr'] = base_lr * lr_multiplier

            # Log dynamic weights at specified frequency
            if self.global_step % self.log_phase5_every == 0:
                sds_weight = float(loss_weights['sds']) if isinstance(loss_weights['sds'], torch.Tensor) else loss_weights['sds']
                contact_weight = float(loss_weights['contact']) if isinstance(loss_weights['contact'], torch.Tensor) else loss_weights['contact']
                temporal_weight = float(loss_weights['temporal']) if isinstance(loss_weights['temporal'], torch.Tensor) else loss_weights['temporal']
                lr_mult = float(lr_multiplier) if isinstance(lr_multiplier, torch.Tensor) else lr_multiplier

                logger.info('phase5/weight_sds', sds_weight, prog_bar=False)
                logger.info('phase5/weight_contact', contact_weight, prog_bar=False)
                logger.info('phase5/weight_temporal', temporal_weight, prog_bar=False)
                logger.info('phase5/lr_multiplier', lr_mult, prog_bar=False)

                logger.debug(
                    f"[Step {self.global_step}] Phase 5 Scheduler: "
                    f"sds={sds_weight:.3f}, contact={contact_weight:.3f}, "
                    f"temporal={temporal_weight:.3f}, lr_mult={lr_mult:.3f}"
                )

        elif self.phase5_enabled:
            # ✅ Phase 5 enabled but scheduler is None (two-stage training mode)
            # Use constant weights and standard learning rate
            loss_weights = {'sds': 1.0, 'contact': 1.0, 'temporal': 1.0}

            if self.global_step % self.log_phase5_every == 0:
                logger.debug(
                    f"[Step {self.global_step}] Phase 5: Using fixed weights "
                    f"(scheduler disabled via use_scheduler=false)"
                )

        else:
            # Phase 5 not enabled - baseline configuration
            loss_weights = {'sds': 1.0, 'contact': 1.0, 'temporal': 1.0}

        # Existing preprocessing code continues...
        self.condition_training()

        # ============================================================
        # FIX: Generate c2w from extrinsics if missing
        # ============================================================
        if 'c2w' not in batch and 'extrinsics' in batch:
            try:
                # Always use CPU for stability (GPU inverse has CUDA solver issues)
                extrinsics_cpu = batch['extrinsics'].cpu().contiguous()
                c2w_cpu = torch.linalg.inv(extrinsics_cpu)
                batch['c2w'] = c2w_cpu.to(batch['extrinsics'].device)
                logger.debug(f"[FIX] Generated c2w from extrinsics (CPU): {batch['c2w'].shape}")
            except Exception as e:
                logger.error(f"[FIX] Failed to generate c2w: {e}")
                logger.error(f"[FIX] extrinsics shape={batch['extrinsics'].shape}, dtype={batch['extrinsics'].dtype}")
                # Don't add incorrect c2w - let Phase 5 skip

        # ================================================================
        # ✅ CRITICAL FIX: BATCH PREPROCESSING - Handle HOLD and GHOP formats
        # ================================================================
        # HOLD ImageDataset: batch["idx"] is list [tensor([0]), tensor([1])]
        # GHOP HOI4D: batch["idx"] is already Tensor [[0], [1]]

        # Fix 1: Handle idx field
        if isinstance(batch["idx"], list):
            # HOLD format: stack list of tensors
            batch["idx"] = torch.stack(batch["idx"], dim=1)
            logger.debug(f"[Batch] Stacked idx from list: {batch['idx'].shape}")
        elif isinstance(batch["idx"], torch.Tensor):
            # GHOP format: already tensor, ensure correct shape [B, 1]
            if batch["idx"].dim() == 1:
                batch["idx"] = batch["idx"].unsqueeze(1)  # [B] -> [B, 1]
            logger.debug(f"[Batch] idx already tensor: {batch['idx'].shape}")
        else:
            raise TypeError(
                f"[Batch] batch['idx'] must be list or Tensor, "
                f"got {type(batch['idx'])}"
            )

        # Fix 2: Handle other potential list fields
        list_fields = ['c2w', 'intrinsics', 'hA', 'right.betas']
        for key in list_fields:
            if key in batch and isinstance(batch[key], list):
                if len(batch[key]) > 0 and isinstance(batch[key][0], torch.Tensor):
                    batch[key] = torch.stack(batch[key], dim=0)
                    logger.debug(f"[Batch] Stacked {key} from list: {batch[key].shape}")

        # ================================================================
        # Continue with existing preprocessing (keep unchanged)
        # ================================================================
        batch = hold_utils.wubba_lubba_dub_dub(batch)
        batch = xdict(batch)
        batch["current_epoch"] = self.current_epoch
        batch["global_step"] = self.global_step

        # ================================================================
        # CRITICAL FIX: Preserve idx shape for nn.Embedding compatibility
        # ================================================================
        # wubba_lubba_dub_dub squeezes idx from [B, 1] to [B]
        # nn.Embedding behaves differently with these shapes:
        #   Input [B, 1] -> Output [B, 1, D] (correct)
        #   Input [B]    -> Output [B, D]    (missing dimension)

        if 'idx' in batch and isinstance(batch['idx'], torch.Tensor):
            original_idx_shape = batch['idx'].shape
            logger.debug(f"[FIX] batch['idx'] shape after wubba: {original_idx_shape}")

            if batch['idx'].dim() == 1:
                reshaped_idx = batch['idx'].unsqueeze(1)  # [B] -> [B, 1]
                batch.overwrite('idx', reshaped_idx)
                logger.debug(f"[FIX] Reshaped idx to: {batch['idx'].shape}")

        # ================================================================
        # NODE LOOP with DIAGNOSTICS
        # ================================================================
        for node in self.model.nodes.values():
            logger.info(f"\n[training_step] Processing node: {node.node_id}")

            node_id = node.node_id
            ghop_keys_to_preserve = [f"{node_id}.params"]
            preserved_values = {}

            for key in ghop_keys_to_preserve:
                if key in batch:
                    preserved_values[key] = batch[key]
                    logger.debug(f"[FIX] Preserving GHOP-extracted {key}: {batch[key].shape}")

            logger.info(f"[training_step] Calling node.params(batch['idx'])...")
            params_dict = node.params(batch['idx'])

            logger.info(f"[training_step] node.params() returned:")
            for k, v in params_dict.items():
                if isinstance(v, torch.Tensor):
                    logger.info(f"    {k}: {v.shape}")

            batch.update(params_dict)

            for key, value in preserved_values.items():
                batch.overwrite(key, value)
                logger.info(f"[FIX] Restored {key}: {value.shape}")

            logger.info(f"[training_step] After batch.update:")

        logger.info("=" * 70)
        logger.info("[training_step] NODE LOOP END - calling model(batch)")
        logger.info("=" * 70)

        # ================================================================
        # BEFORE FORWARD PASS - Final shape check
        # ================================================================
        logger.info("[Forward] ========== FINAL SHAPES BEFORE model(batch) ==========")
        logger.info(f"  batch['idx']: {batch['idx'].shape}")
        for key in ['right.full_pose', 'right.betas', 'right.transl']:
            if key in batch:
                logger.info(f"  batch['{key}']: {batch[key].shape}")
        logger.info("[Forward] =============================================")

        # ================================================================
        # ✅ FINAL FIX: Create params for all nodes with correct shapes
        # ================================================================
        for node in self.model.nodes.values():
            node_id = node.node_id
            params_key = f"{node_id}.params"

            # For MANO nodes (right/left): create from full_pose
            if node_id in ['right', 'left']:
                full_pose_key = f"{node_id}.full_pose"
                if full_pose_key in batch and params_key not in batch:
                    batch[params_key] = batch[full_pose_key]
                    logger.info(f"[FIX] Created {params_key} from {full_pose_key}: {batch[params_key].shape}")

            # For object node: create scene_scale (scalar per sample)
            elif node_id == 'object':
                if params_key not in batch:
                    batch_size = batch['idx'].shape[0]
                    object_params = torch.ones(batch_size, 1, 1, device=batch['idx'].device)
                    batch[params_key] = object_params
                    logger.info(f"[FIX] Created {params_key} (scene_scale=1.0): {object_params.shape}")

        # ================================================================
        # Generate UV coordinates if missing
        # ================================================================
        if 'uv' not in batch:
            logger.info("[FIX] Generating UV coordinates for ray sampling...")
            batch_size = batch['idx'].shape[0]
            num_sample = self.args.num_sample if hasattr(self.args, 'num_sample') else 128
            uv = torch.rand(batch_size, num_sample, 2, device=batch['idx'].device)
            batch['uv'] = uv
            logger.info(f"[FIX] Generated UV: {uv.shape}")

        # ================================================================
        # Generate extrinsics from c2w
        # ================================================================
        if 'extrinsics' not in batch and 'c2w' in batch:
            logger.info("[FIX] Generating extrinsics from c2w...")
            c2w = batch['c2w']
            try:
                extrinsics = torch.linalg.inv(c2w)
            except:
                logger.warning("[FIX] GPU inverse failed, using CPU...")
                c2w_cpu = c2w.cpu()
                extrinsics = torch.linalg.inv(c2w_cpu).to(c2w.device)
            batch['extrinsics'] = extrinsics
            logger.info(f"[FIX] Generated extrinsics: {extrinsics.shape}")

        # Get batch size and device
        B = batch['idx'].shape[0]
        T = 1
        device = batch['idx'].device
        scene_scale = 1.0

        if not hasattr(self, 'im_h') or not hasattr(self, 'im_w'):
            self.im_h = 480
            self.im_w = 640

        if "object.params" not in batch:
            batch["object.params"] = torch.full(
                (B, T, 1), scene_scale, dtype=torch.float32, device=device
            )
            logger.info(f"[FIX] Created object.params (scene_scale={scene_scale}): {batch['object.params'].shape}")

        if 'uv' not in batch:
            logger.info("[FIX] Generating UV coordinates for ray sampling...")
            uv = gen_random_sample(
                128, batch["idx"], np.array([self.im_h, self.im_w]), self.n_samples_per_pixel
            )
            batch["uv"] = uv
            logger.info(f"[FIX] Generated UV: {batch['uv'].shape}")

        if 'extrinsics' not in batch and 'c2w' in batch:
            logger.info("[FIX] Generating extrinsics from c2w...")
            c2w = batch["c2w"]
            try:
                w2c = torch.inverse(c2w)
            except RuntimeError:
                logger.warning("[FIX] GPU inverse failed, using CPU...")
                w2c = torch.inverse(c2w.cpu()).to(c2w.device)
            batch["extrinsics"] = w2c
            logger.info(f"[FIX] Generated extrinsics: {batch['extrinsics'].shape}")

        # ================================================================
        # ✅ FIX: Only detach metadata, NOT trainable parameters
        # CRITICAL: MANO parameters must keep gradients for optimization!
        # ================================================================
        logger.debug("[FIX] Detaching ONLY metadata tensors (preserving MANO gradients)...")

        keys_to_detach = [
            # Camera and scene metadata (no gradients needed)
            'uv',
            'extrinsics',
            'c2w',
            'intrinsics',
            'hA',
            'th_betas',
        ]

        # ================================================================
        # ⚠️  IMPORTANT: MANO hand parameters are NO LONGER DETACHED!
        # The following keys are intentionally EXCLUDED from detachment:
        #   - 'right.params'
        #   - 'right.full_pose'
        #   - 'right.pose'
        #   - 'right.global_orient'
        #   - 'right.transl'
        #   - 'right.betas'
        # This allows gradients to flow from loss → rendering → MANO
        # ================================================================

        for key in keys_to_detach:
            if key in batch and torch.is_tensor(batch[key]):
                detached_tensor = batch[key].detach()

                if hasattr(batch, 'overwrite'):
                    batch.overwrite(key, detached_tensor)
                else:
                    del batch[key]
                    batch[key] = detached_tensor

                logger.debug(f"  Detached: {key} (shape: {detached_tensor.shape})")

        for key in list(batch.keys()):
            if key.endswith('_n') and torch.is_tensor(batch[key]):
                detached_tensor = batch[key].detach()

                if hasattr(batch, 'overwrite'):
                    batch.overwrite(key, detached_tensor)
                else:
                    del batch[key]
                    batch[key] = detached_tensor

                logger.debug(f"  Detached: {key} (shape: {detached_tensor.shape})")

        logger.info("[FIX] ✓ Metadata detached, MANO parameters preserved for gradient flow")

        # ================================================================
        # UPDATE GEOMETRY LATENT (Option B - FiLM)
        # Must happen BEFORE forward pass to avoid inplace modification
        # ================================================================
        if hasattr(self.model, 'nodes') and 'object' in self.model.nodes:
            object_node = self.model.nodes['object']
            logger.info(f"update_geometry_latent hasattr result: {hasattr(object_node, 'update_geometry_latent')}")
            if hasattr(object_node, 'update_geometry_latent'):
                logger.info(f"[update_geometry_latent Step {self.global_step}] Pre-forward: Updating z_geo_refined...")
                obj_model = None
                if hasattr(object_node, 'server') and hasattr(object_node.server, 'object_model'):
                    obj_model = object_node.server.object_model
                try:
                    obj_verts = None
                    if hasattr(object_node, 'server') and hasattr(object_node.server, 'object_model'):
                        obj_model = object_node.server.object_model
                        if hasattr(obj_model, 'v3d_cano'):
                            obj_verts = obj_model.v3d_cano
                    if obj_verts is not None:
                        grid_res = getattr(self.opt.model, "grid_resolution", 16)
                        voxel_sdf = self._mesh_to_sdf_grid(
                            obj_verts,
                            grid_resolution=grid_res, # hard coded resolution
                            for_encoder=True
                        )
                        obj_model.cache_voxel_grid(voxel_sdf)
                        success = object_node.update_geometry_latent()
                        if success:
                            logger.info(f"[update_geometry_latent Step {self.global_step}] Updated z_geo_refined pre-forward")
                except Exception as e:
                    logger.debug(f"[update_geometry_latent Step {self.global_step}] Could not update geometry latent: {e}")

            # Check sdf_grid
            if hasattr(obj_model, 'sdf_grid'):
                sdf = obj_model.sdf_grid
                is_param = isinstance(sdf, nn.Parameter)
                requires_grad = sdf.requires_grad if hasattr(sdf, 'requires_grad') else False

                logger.info(f"[sdf_grid] Type: {type(sdf).__name__}")
                logger.info(f"[sdf_grid] Is Parameter: {is_param}")
                logger.info(f"[sdf_grid] Requires Grad: {requires_grad}")
                logger.info(f"[sdf_grid] Shape: {sdf.shape}")
                logger.info(f"[sdf_grid] Stats: mean={sdf.mean().item():.4f}, std={sdf.std().item():.4f}")
                logger.info(f"[sdf_grid] Range: [{sdf.min().item():.4f}, {sdf.max().item():.4f}]")

                has_zero_crossing = (sdf.min() < 0) and (sdf.max() > 0)
                logger.info(f"[sdf_grid] Has zero-crossing: {has_zero_crossing}")

                if not is_param:
                    logger.info("[sdf_grid] Not an nn.Parameter (treated as derived/cached grid).")
                if not requires_grad:
                    logger.info("[sdf_grid] requires_grad=False (OK for derived grid).")
                if not has_zero_crossing:
                    logger.warning("⚠️ [sdf_grid] No zero-crossing at initialization!")

        # ================================================================
        # FORWARD PASS
        # ================================================================
        logger.info("[Forward] Calling self.model(batch)...")
        model_outputs = self.model(batch)
        # Compute base loss first; this will initialize self.loss.im_h/im_w
        loss_output = self.loss(batch, model_outputs)
        # Check which nodes contributed to output
        if self.global_step % 100 == 0:
            logger.warning(f"\n[NODE CHECK - Step {self.global_step}]")
            logger.warning(f"  Model output keys: {list(model_outputs.keys())}")

            # Check if hand node was used
            for node_name, node in self.model.nodes.items():
                logger.warning(f"  Node '{node_name}' type: {type(node).__name__}")
        logger.info(f"[Forward] Model returned {len(model_outputs)} keys")
        if should_profile:
            self.memory_profiler.checkpoint("after_forward")
        # ------------------------------------------------------------------
        # BUILD LOW-RES DEPTH MAP FOR GEOMETRY PRIOR (Option A)
        # ------------------------------------------------------------------
        try:
            # Prefer object view depth; fall back to scene depth if needed
            cand_keys = ["object.depth", "right.depth", "depth"]
            depth_vec = None
            for k in cand_keys:
                if k in model_outputs:
                    depth_vec = model_outputs[k]
                    chosen_key = k
                    break

            if depth_vec is not None:
                # depth_vec is typically [N_rays, 1] or [N_rays]
                if depth_vec.dim() == 2 and depth_vec.shape[1] == 1:
                    depth_vec = depth_vec[:, 0]  # [N_rays]

                N = depth_vec.shape[0]

                # Build a compact H×W grid from N rays (e.g., 16×16 for N=256)
                H = int(N ** 0.5)
                while H > 1 and N % H != 0:
                    H -= 1
                W = N // H  # This will be >= H

                depth_img = depth_vec.view(H, W)  # [H, W]
                model_outputs["depth_map"] = depth_img

                if self.global_step == 0:
                    logger.info(
                        f"[depth_map] Built low-res depth_map from {chosen_key} "
                        f"with shape=(H={H}, W={W}), N={N}"
                    )

                depth_img = None
                if H is not None and W is not None and H * W == N:
                    # Single-view grid
                    depth_img = depth_vec.view(H, W)  # [H, W]
                elif H is not None and W is not None and N % (H * W) == 0:
                    # Multi-view grid: [B, H, W], take first view
                    B = N // (H * W)
                    depth_img = depth_vec.view(B, H, W)[0]  # [H, W]

                if depth_img is not None:
                    model_outputs["depthmap"] = depth_img
                    if self.global_step == 0:
                        logger.info(
                            f"[DEPTHMAP] Built low-res depthmap from {chosen_key} "
                            f"with shape={tuple(depth_img.shape)}"
                        )
                else:
                    if self.global_step == 0:
                        logger.warning(
                            f"[DEPTHMAP] Could not reshape {chosen_key} with N={N} "
                            f"into (H={H}, W={W}); depth smoothness will be skipped."
                        )
        except Exception as e:
            if self.global_step == 0:
                logger.warning(f"[DEPTHMAP] Failed to construct depthmap: {e}")

        # ================================================================
        # COMPUTE BASE LOSSES
        # ================================================================
        # loss_output = self.loss(batch, model_outputs)
        total_loss = loss_output["loss"]  # base HOLD loss
        # ================================================================
        # APPLY DYNAMIC RGB WEIGHT SCHEDULE (NEW)
        # ================================================================
        if hasattr(self, 'rgb_weight_scheduler') and self.rgb_weight_scheduler is not None:
            current_step = self.global_step
            effective_w_rgb = self.rgb_weight_scheduler.get_weight(current_step)
            base_w_rgb = self.opt.loss.w_rgb

            # Only adjust if weights differ
            if abs(effective_w_rgb - base_w_rgb) > 1e-6:
                # Extract original RGB loss (already weighted by base_w_rgb)
                original_rgb_loss = loss_output.get('loss/rgb', 0.0)

                if isinstance(original_rgb_loss, torch.Tensor):
                    original_rgb_loss = original_rgb_loss.item()

                if original_rgb_loss != 0:
                    # Calculate raw RGB loss (remove base weight)
                    rgb_loss_raw = original_rgb_loss / base_w_rgb

                    # Calculate new weighted RGB loss
                    new_rgb_loss = effective_w_rgb * rgb_loss_raw

                    # Adjust total loss: remove old RGB, add new RGB
                    adjusted_total = loss_output["loss"] - original_rgb_loss + new_rgb_loss

                    # Update loss_output
                    loss_output["loss"] = adjusted_total
                    total_loss = adjusted_total  # Update the variable used below
                    loss_output['loss/rgb'] = new_rgb_loss
                    loss_output['loss/rgb_raw'] = rgb_loss_raw
                    loss_output['w_rgb_effective'] = effective_w_rgb

                    # Log for monitoring
                    if self.global_step % 500 == 0 or self.global_step == 8000 or self.global_step == 30000:
                        logger.info(f"[RGB WEIGHT] Step {current_step}: "
                                    f"base_w={base_w_rgb:.4f} -> effective_w={effective_w_rgb:.4f}, "
                                    f"rgb_loss adjusted: {original_rgb_loss:.6f} -> {new_rgb_loss:.6f}, "
                                    f"total_loss: {loss_output['loss']:.6f}")
                else:
                    loss_output['w_rgb_effective'] = base_w_rgb
            else:
                loss_output['w_rgb_effective'] = base_w_rgb
        else:
            loss_output['w_rgb_effective'] = self.opt.loss.w_rgb
        # ================================================================
        # END DYNAMIC RGB WEIGHT
        # ================================================================

        # After line 2178, ADD COMMENT:
        # ================================================================
        # LOSS AGGREGATION ORDER (DO NOT MODIFY):
        # 1. Base losses (RGB, mask, etc.)
        # 2. Geometric losses (normal consistency, depth smoothness)
        # 3. Object shape losses (chamfer, smoothness, SDF regularization)
        # 4. Phase 3 losses (SDS diffusion)
        # 5. Phase 4 losses (contact)
        # 6. Phase 5 losses (temporal)
        # ALWAYS add to total_loss directly, NOT to geometric_loss after line 2353
        # ================================================================

        # Immediately after total_loss is computed:
        base_loss = total_loss  # define this first
        base_mult = 1.0  # default, can be overridden by warm-up logic

        # Track components for logging
        loss_components = {
            'base_raw': base_loss.item(),  # unscaled HOLD base loss
            'base': (base_mult * base_loss).item(),  # may be overwritten later after phase logic
            'geometric': 0.0,
            'object_shape': 0.0,
            'sds': 0.0,
            'contact': 0.0,
            'temporal': 0.0
        }
        # Keep an explicit copy of the base (HOLD) loss
        base_loss = total_loss

        step = self.global_step

        # Hyperparameters for warm-up (can be overridden in config)
        geom_warmup_end = getattr(self.opt.training, 'geom_warmup_end', 2000)   # Phase 2.1
        render_warmup_end = getattr(self.opt.training, 'render_warmup_end', 4000)  # Phase 2.2
        joint_warmup_end = getattr(self.opt.training, 'joint_warmup_end', 8000)    # Phase 2.3
        sdf_warmup_end = getattr(self.opt.training, 'sdf_warmup_end', 2000)

        # Persistent dynamic multipliers for closed-loop curriculum (INIT IF MISSING)
        if not hasattr(self, "dynamic_sds_mult"):
            self.dynamic_sds_mult = 1.0
        if not hasattr(self, "dynamic_geom_mult"):
            self.dynamic_geom_mult = 1.0

        # Optional: read freeze schedule to down-weight geometry after shape solidifies
        training_cfg = getattr(self.opt, 'training', None)
        freeze_at = getattr(training_cfg, 'object_freeze_at', None) if training_cfg is not None else None
        unfreeze_at = getattr(training_cfg, 'object_unfreeze_at', None) if training_cfg is not None else None

        # Default multipliers
        base_mult = 1.0   # scales base_loss (RGB + masks etc.)
        geom_mult = 1.0   # scales geometric_loss (normal, depth, chamfer, SDF reg, obj chamfer)
        sds_mult = 1.0    # scales SDS loss before adding to total
        contact_mult = 1.0
        temporal_mult = 1.0
        phase_tag = "main_training"

        # Determine warm-up phase and set multipliers
        if step < geom_warmup_end:
            # Phase 2.1: GEOMETRY-ONLY warm-up (build block shape)
            base_mult = 0.0
            geom_mult = 1.0
            sds_mult = 0.0
            contact_mult = 0.0
            phase_tag = "geom_warmup"
        elif step < render_warmup_end:
            # Phase 2.2: Add RGB, still no SDS/contact
            base_mult = 1.0
            geom_mult = 1.0
            sds_mult = 0.0
            contact_mult = 0.0
            phase_tag = "rgb_geom"
        elif step < joint_warmup_end:
            # Phase 2.3: Turn on SDS and start contact early to anchor pose
            base_mult = 1.0
            geom_mult = 1.0
            sds_mult = 1.0
            contact_mult = 1.0   # earlier, stronger contact
            phase_tag = "rgb_geom_sds_contact"
        else:
            # Main training: all terms active
            base_mult = 1.0
            geom_mult = 1.0
            sds_mult = 1.0
            contact_mult = 1.0
            phase_tag = "main_training"
        # After object is frozen, de-emphasize geometry and let contact/SDS fine-tune pose.
        # If object_unfreeze_at is set, restore full weights after that step.
        if freeze_at is not None and step >= freeze_at:
            if unfreeze_at is None or step < unfreeze_at:
                geom_mult *= 0.3     # geometry still present but reduced
                contact_mult *= 1.5  # let contact dominate pose refinement

        # Apply closed-loop dynamic multipliers
        geom_mult *= self.dynamic_geom_mult
        sds_mult *= self.dynamic_sds_mult

        if step % 100 == 0:
            logger.info(
                f"[Warmup] step={step} phase={phase_tag} "
                f"base_mult={base_mult:.3f}, geom_mult={geom_mult:.3f}, "
                f"sds_mult={sds_mult:.3f}, contact_mult={contact_mult:.3f}, "
                f"temporal_mult={temporal_mult:.3f}"
            )

        sds_disabled = (sds_mult == 0.0)

        # ================================================================
        # STEP 2: SELF-SUPERVISED GEOMETRIC LOSSES
        # ================================================================
        geometric_loss = torch.tensor(0.0, device=total_loss.device, requires_grad=True)

        # Diagnostic: Check model structure at first step
        if self.global_step == 0:
            logger.info(f"[Config Debug] Has opt.loss: {hasattr(self.opt, 'loss')}")
            if hasattr(self.opt, 'loss'):
                logger.info(f"[Config Debug] Loss config keys: {list(self.opt.loss.keys())[:20]}")
                logger.info(f"[Config Debug] w_chamfer: {self.opt.loss.get('w_chamfer', 'NOT FOUND')}")
                logger.info(
                    f"[Config Debug] chamfer_start_step: {self.opt.loss.get('chamfer_start_step', 'NOT FOUND')}")

            logger.info(f"[Loss Debug] Loss object type: {type(self.loss).__name__}")
            logger.info(f"[Loss Debug] Loss attributes: {[attr for attr in dir(self.loss) if not attr.startswith('_')][:30]}")

            # Check specific weights
            logger.info(f"[Loss Debug] Has w_normal_consistency: {hasattr(self.loss, 'w_normal_consistency')}")
            if hasattr(self.loss, 'w_normal_consistency'):
                logger.info(f"[Loss Debug] w_normal_consistency value: {self.loss.w_normal_consistency}")

            logger.info(f"[Loss Debug] Has w_chamfer: {hasattr(self.loss, 'w_chamfer')}")
            if hasattr(self.loss, 'w_chamfer'):
                logger.info(f"[Loss Debug] w_chamfer value: {self.loss.w_chamfer}")
        # 1. Normal Consistency
        # Model uses 'nodes', not 'servers' - confirmed via diagnostics
        logger.info(f"[GEOM-DEBUG] Before normal consistency: {geometric_loss.item()}")

        w_nc = getattr(self.opt.loss, "w_normal_consistency", 0.0)
        # In src/hold/hold.py, normal consistency section (around line 2510)
        # Ensure mesh is extracted before computing normal consistency
        if w_nc > 0.0:
            try:
                # FIX: Use mesh-based normal consistency instead of SDF gradients
                object_server = self.model.nodes['object'].server
                obj_model = object_server.object_model

                # NOTE: Why normal consistency may be 0.000000:
                # Templates (e.g., foam_brick.pt) are torch.Tensor objects without faces.
                # ObjectModel.__init__ checks for template.faces (line 114-119) but finds None,
                # so self.faces is set to None. Without triangular faces, mesh_normal_consistency
                # cannot compute surface smoothness. Fallback: use w_vertex_normal_prior (config)
                # for vertex-level normal regularization instead.
                if not hasattr(obj_model, 'faces') or obj_model.faces is None:
                    # Generate faces from voxel grid or use template
                    logger.warning("[Normal Consistency] No faces available, skipping")
                    normal_loss = None
                elif hasattr(obj_model, 'v3d_cano'):
                    # Create mesh
                    mesh = Meshes(
                        verts=[obj_model.v3d_cano],
                        faces=[obj_model.faces]
                    )
                    normal_loss = mesh_normal_consistency(mesh)

                    # CRITICAL DIAGNOSTIC: Verify non-zero loss
                    logger.info(
                        f"[NORMAL DEBUG] Step {self.global_step}: mesh_normal_consistency = "
                        f"{normal_loss.item():.6f}, verts={obj_model.v3d_cano.shape}, "
                        f"faces={obj_model.faces.shape}"
                    )
                else:
                    # Fallback: Skip if mesh not available (don't use dead SDF gradient loss)
                    logger.warning(
                        f"[Normal Consistency] Step {self.global_step}: v3d_cano/faces not available, "
                        f"skipping normal consistency."
                    )
                    normal_loss = None

                    # ADD to geometric_loss
                if normal_loss is not None:
                    geometric_loss += w_nc * normal_loss
                    loss_output["loss/normal_consistency"] = normal_loss.item()
                else:
                    logger.warning(f"[Normal Consistency] Step {self.global_step}: No normal loss computed")

            except Exception as e:
                logger.warning(
                    f"[Normal Consistency] Step {self.global_step}: failed with {str(e)}"
                )

        else:
            if self.global_step == 0:
                logger.info("Normal Consistency weight is zero; term disabled in config.")

        logger.info(f"[GEOM-DEBUG] After normal consistency: {geometric_loss.item()}")

        # 2. Depth Smoothness (edge-aware) - UNCHANGED
        depth_key = None
        for key in ["depth_map", "object.depth", "right.depth", "depth", "z_vals"]:
            if key in model_outputs:
                depth_key = key
                break

        rgb_key = None
        for key in ['gt.rgb', 'rgb', 'target_rgb', 'image']:
            if key in batch:
                rgb_key = key
                break

        if depth_key and rgb_key:
            try:
                rendered_depth = model_outputs[depth_key]
                image_rgb = batch[rgb_key]

                # Handle per-node depth (e.g., 'right.depth')
                if rendered_depth.dim() == 1:
                    # Single depth value per sample, skip smoothness
                    pass
                # Validate depth format for spatial smoothness
                elif rendered_depth.dim() == 2 and rendered_depth.shape[1] == 1:
                    # Per-ray depth [N_rays, 1] - skip spatial smoothness
                    if self.global_step == 0:
                        logger.warning(
                            f"⚠️ Depth smoothness disabled: per-ray rendering "
                            f"(shape={rendered_depth.shape}) incompatible with spatial gradients. "
                            f"Use normal consistency + GSD instead."
                        )
                # Skip this loss - proceed to next section
                elif rendered_depth.dim() >= 2:
                    # Normalize depth to [H, W]
                    if rendered_depth.dim() == 4:
                        # e.g., [B, 1, H, W] or [B, C, H, W] – take first sample, first channel
                        rendered_depth = rendered_depth[0, 0]  # [H, W]
                    elif rendered_depth.dim() == 3:
                        # e.g., [B, H, W] – take first sample
                        rendered_depth = rendered_depth[0]     # [H, W]
                    # else: dim == 2 already, assume [H, W] or [Nrays, ?]

                    # Ensure both tensors have compatible H, W
                    H_rgb, W_rgb = image_rgb.shape[-2:]
                    if rendered_depth.dim() == 2 and rendered_depth.shape != (H_rgb, W_rgb):
                        if rendered_depth.numel() == H_rgb * W_rgb:
                            rendered_depth = rendered_depth.reshape(H_rgb, W_rgb)

                    depth_smooth_loss = depth_smoothness_loss(rendered_depth, image_rgb)
                    geometric_loss = geometric_loss + self.opt.loss.w_depth_smoothness * depth_smooth_loss
                    logger.info("loss/depth_smoothness", depth_smooth_loss.item())
                    loss_output["loss/depth_smoothness"] = depth_smooth_loss.item()

                    if self.global_step % 100 == 0 and geometric_loss.item() != 0:
                        logger.info(
                            f"Geometric Loss Step {self.global_step} {geometric_loss.item():.6f}"
                        )
                    elif self.global_step % 100 == 0:
                        logger.debug(
                            f"[Geometric Loss] Step {self.global_step}: 0.000000 "
                            f"(normal_consistency skipped, depth_smoothness incompatible with ray rendering)"
                        )
            except Exception as e:
                if self.global_step == 0:
                    logger.error(f"❌ Depth smoothness failed: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
        elif self.global_step == 0:
            logger.warning(f"⚠️ Depth smoothness skipped:")
            logger.warning(f"   depth_key found: {depth_key}")
            logger.warning(f"   rgb_key found: {rgb_key}")
            logger.warning(f"   Available depth keys: {[k for k in model_outputs.keys() if 'depth' in k.lower()]}")
            logger.warning(f"   Available rgb keys: {[k for k in batch.keys() if 'rgb' in k.lower() or 'image' in k.lower()]}")
        logger.info(f"[GEOM-DEBUG] After depth smoothness: {geometric_loss.item()}")

        # 3. Template Chamfer (activation step configured in YAML)
        # Get Chamfer activation threshold from config (with fallback for backward compatibility)
        # NOTE: Parameters are loaded from 'loss:' section and flattened to self.args
        # ============================================================
        # EARLY PARAMETER VALIDATION (Run once at step 0)
        # ============================================================
        if self.global_step == 0:
            logger.info("=" * 60)
            logger.info("[PARAMETER VALIDATION] Checking learnable parameters...")

            # Check optimizer groups
            logger.info("\n[OPTIMIZER] Learning rates:")
            for idx, group in enumerate(self.optimizers().param_groups):
                group_name = group.get('name', f'group_{idx}')
                logger.info(f"  {group_name}: lr={group['lr']}")

                # NEW: Check if object vertices are in this group
                for param in group['params']:
                    # Try to identify if this is v3d_cano
                    for name, module_param in self.model.named_parameters():
                        if param is module_param and 'v3d_cano' in name:
                            logger.info(f"    ✓ Found v3d_cano in group '{group_name}'!")  # <-- Changed this line

                            # NEW: Log the multiplier being used
                            expected_lr = self.args.lr * 1.0  # Should match what we set
                            actual_lr = group['lr']
                            logger.info(f"    Base LR: {self.args.lr:.2e}, Expected: {expected_lr:.2e}, Actual: {actual_lr:.2e}")

                            if group['lr'] < 1e-6:
                                logger.error(
                                    f"    ❌ CRITICAL: v3d_cano has lr={group['lr']:.2e}, "
                                    f"should be ~1e-4!"
                                )
                            elif group['lr'] > 1e-3:
                                logger.warning(
                                    f"    ⚠️ v3d_cano lr={group['lr']:.2e} might be too high"
                                )
                            else:
                                logger.info(
                                    f"    ✅ v3d_cano lr={group['lr']:.2e} looks reasonable"
                                )
            logger.info("=" * 60)
        chamfer_start_step = getattr(self.opt.loss, 'chamfer_start_step', 20000)  # Default: 20000 for 30-epoch
        chamfer_weight = getattr(self.opt.loss, 'w_chamfer', 0.05)  # Default: 0.05

        # Log configuration on first step
        if self.global_step == 0:
            logger.info(f"[Chamfer Config] Activation threshold: step {chamfer_start_step}")
            logger.info(f"[Chamfer Config] Weight: {chamfer_weight}")
            logger.info(f"[Chamfer Config] Available args: {list(vars(self.args).keys())[:30]}")  # Debug: show available params

        if self.global_step >= chamfer_start_step:  # FIXED: >= instead of >
            try:
                # Verify methods exist
                if not hasattr(self, '_extract_hand_mesh_for_chamfer'):
                    if self.global_step == chamfer_start_step:  # Log once
                        logger.error("❌ Method '_extract_hand_mesh_for_chamfer' not implemented!")
                    raise NotImplementedError("_extract_hand_mesh_for_chamfer")

                if not hasattr(self, '_get_template_mesh'):
                    if self.global_step == chamfer_start_step:  # Log once
                        logger.error("❌ Method '_get_template_mesh' not implemented!")
                    raise NotImplementedError("_get_template_mesh")

                # Extract current hand mesh (with gradients)
                hand_verts = self._extract_hand_mesh_for_chamfer(batch)  # [B, 778, 3]

                # Get canonical template (constant reference)
                template_verts = self._get_template_mesh(batch)  # [B, 778, 3]

                # Verify shapes
                if hand_verts is None or template_verts is None:
                    raise ValueError(f"Mesh extraction failed: hand_verts={hand_verts}, template_verts={template_verts}")

                # Compute bidirectional Chamfer distance
                from pytorch3d.loss import chamfer_distance
                chamfer_loss, _ = chamfer_distance(hand_verts, template_verts)

                # Add to geometric loss with configured weight
                geometric_loss = geometric_loss + chamfer_weight * chamfer_loss
                loss_output['loss/chamfer'] = chamfer_loss.item()

                if self.global_step % 100 == 0:
                    logger.info(f"✅ [Chamfer Loss] Step {self.global_step}: {chamfer_loss.item():.6f}")

            except NotImplementedError as e:
                # Method doesn't exist - log once and skip
                pass

            except Exception as e:
                # Log ALL exceptions during first 5 activations
                if self.global_step <= chamfer_start_step + 500 or self.global_step % 100 == 0:
                    logger.error(f"❌ Template Chamfer failed at step {self.global_step}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())

                # Add zero loss to maintain gradient graph consistency
                loss_output['loss/chamfer'] = 0.0

        # ============================================================
        # GEOMETRY-SHARE CONTROLLER (Phase 3 only)
        # ============================================================
        phase3_active = (
                getattr(self, "phase3_enabled", False)
                and self.global_step >= getattr(self, "phase3_start_iter", 0)
                and self.global_step < getattr(self, "phase3_end_iter", 999999)
        )

        if phase3_active and geometric_loss.requires_grad:
            with torch.no_grad():
                # Use base (pre-geometry) loss as reference (unscaled)
                base_scalar = base_loss.detach().abs().clamp_min(1e-8)
                geom_scalar = geometric_loss.detach().abs()

                # Current share of geometry among (base + geometry)
                current_share = float(geom_scalar / (base_scalar + geom_scalar))

                # Exponential moving average for stability
                if self._geom_share_ema == 0.0:
                    self._geom_share_ema = current_share
                else:
                    m = self._geom_share_ema_momentum
                    self._geom_share_ema = m * self._geom_share_ema + (1.0 - m) * current_share

                # If EMA share is too low, boost geometry scale up to max_geom_scale
                if self._geom_share_ema < self.target_geom_share:
                    old_scale = self._geom_scale
                    self._geom_scale = min(self._geom_scale * self.geom_boost_factor,
                                           self.max_geom_scale)
                    logger.info(
                        f"[Geom-Controller] step={self.global_step} "
                        f"share={self._geom_share_ema:.4f} < target={self.target_geom_share:.4f} → "
                        f"scale {old_scale:.3f} → {self._geom_scale:.3f}"
                    )

        # ============================================================
        # OBJECT TEMPLATE CHAMFER (New - Step 3A)
        # ============================================================

        # Object Template Chamfer
        if self.global_step % 50 == 0:  # Log every 50 steps
            logger.info(f"[Object Chamfer DEBUG - Step {self.global_step}]")
            logger.info(
                f"  w_obj_chamfer: {getattr(self, 'current_obj_chamfer_weight', getattr(self.opt.loss, 'w_obj_chamfer', 'NOT SET'))}")
            logger.info(f"  start_step: {getattr(self.opt.loss, 'obj_chamfer_start_step', 'NOT SET')}")
            logger.info(
                f"  Should activate: {self.global_step >= getattr(self.opt.loss, 'obj_chamfer_start_step', 20000)}")

        # obj_chamfer_weight = getattr(self.opt.loss, 'w_obj_chamfer', 0.0)
        base_w = getattr(self.opt.loss, 'w_obj_chamfer', 0.2)
        step = self.global_step

        # NEW: Check for decay schedule in config
        decay_cfg = getattr(self.opt.loss, 'w_obj_chamfer_decay', {})
        logger.info(f"[Chamfer Decay] decay_cfg type: {type(decay_cfg)}, value: {decay_cfg}")
        decay_enabled = decay_cfg.get('enabled', False)
        logger.info(f"[Chamfer Decay] enabled={decay_enabled}, step={step}, cfg={decay_cfg.keys()}")
        if decay_enabled:
            # Config-driven decay schedule
            obj_chamfer_weight = base_w  # Default

            # Check each step threshold in order
            if step >= decay_cfg.get('step_25000', 25000):
                obj_chamfer_weight = decay_cfg.get('step_25000_weight', base_w * 0.25)
            elif step >= decay_cfg.get('step_20000', 20000):
                obj_chamfer_weight = decay_cfg.get('step_20000_weight', base_w * 0.5)
            elif step >= decay_cfg.get('step_15000', 15000):
                obj_chamfer_weight = decay_cfg.get('step_15000_weight', base_w * 0.75)
            elif step >= decay_cfg.get('step_10000', 10000):
                obj_chamfer_weight = decay_cfg.get('step_10000_weight', base_w * 0.9)
            elif step >= decay_cfg.get('step_3000', 3000):  # ADD THIS
                obj_chamfer_weight = decay_cfg.get('step_3000', base_w * 0.05)  # ADD THIS
            else:
                obj_chamfer_weight = base_w
        else:
            # Fallback to hardcoded schedule (original behavior)
            if step < 5000:
                obj_chamfer_weight = base_w                # 0–5k: 1.0×
            elif step < 10000:
                obj_chamfer_weight = base_w * 0.5          # 5–10k: 0.5× (50% decay)
            else:
                obj_chamfer_weight = base_w * 0.1          # ≥10k: 0.1× (90% decay)

        if step % 1000 == 0:
            logger.info(f"[Obj Chamfer] Step {step}: weight={obj_chamfer_weight:.4f}")
        obj_chamfer_start = getattr(self.opt.loss, 'obj_chamfer_start_step', 20000)
        # Check if Chamfer should be temporarily suppressed (e.g., after SDS collapse)
        suppress_chamfer = hasattr(self, "_obj_chamfer_suppressed_until") and \
                           (self.global_step < self._obj_chamfer_suppressed_until)

        if obj_chamfer_weight > 0 and self.global_step >= obj_chamfer_start and not suppress_chamfer:
            if len(self.object_templates) == 0:
                if self.global_step == obj_chamfer_start:
                    logger.warning(
                        "Object Chamfer enabled but no templates loaded. "
                        "Run scripts/prepare_templates.py first."
                    )
            else:
                # ============================================================
                # TEMPLATE-BASED SDF INITIALIZATION (ONE-TIME AT ACTIVATION)
                # ============================================================
                if self.global_step == obj_chamfer_start:
                    try:
                        # Extract category
                        init_category = batch.get('object_category') or batch.get('category')

                        # Unwrap nested structures
                        while isinstance(init_category, (list, tuple)) and len(init_category) > 0:
                            init_category = init_category[0]
                        if isinstance(init_category, torch.Tensor):
                            init_category = (init_category.item() if init_category.numel() == 1
                                           else str(init_category[0].item()))
                        if init_category:
                            init_category = str(init_category).strip().lower()

                        # Initialize from template if SDF is degenerate
                        if init_category and init_category in self.object_templates:
                            object_node = self.model.nodes.get('object')
                            if (object_node and hasattr(object_node, 'server') and
                                hasattr(object_node.server, 'object_model') and
                                hasattr(object_node.server.object_model, 'sdf_grid')):

                                sdf_grid = object_node.server.object_model.sdf_grid
                                sdf_std = sdf_grid.std().item()

                                # ALWAYS initialize at activation step (removed degenerate check)
                                logger.info(
                                    f"[SDF INIT] Initializing SDF teacher grid from template '{init_category}' "
                                    f"(current std={sdf_std:.6f})..."
                                )

                                # Only initialize if degenerate
                                template_verts = self.object_templates[init_category]

                                # Build a volumetric teacher grid from the template mesh
                                initialized_sdf = self._mesh_to_sdf_grid(
                                    template_verts,
                                    grid_resolution=sdf_grid.shape[-1],
                                    padding=0.1,
                                )

                                if initialized_sdf is not None:
                                    with torch.no_grad():
                                        # Treat as derived teacher: write detached values
                                        sdf_grid.data = initialized_sdf.to(sdf_grid.device).detach()

                                    logger.info(
                                        f"✅ [SDF INIT] Success! New stats: "
                                        f"mean={sdf_grid.mean().item():.4f}, "
                                        f"std={sdf_grid.std().item():.4f}, "
                                        f"range=[{sdf_grid.min().item():.4f}, {sdf_grid.max().item():.4f}]"
                                    )
                    except Exception as e:
                        logger.error(f"[SDF INIT] Failed: {e}")

                # ============================================================
                # INITIALIZE TRACKING VARIABLES (FIX: Define before use!)
                # ============================================================
                template_applied = False
                obj_verts_list = None
                obj_faces_list = None
                obj_category = None

                # ============================================================
                # STEP 1: Extract Object Category
                # ============================================================

                obj_category = None

                # Try multiple possible keys (dataset-dependent)
                category_keys = ['object_category', 'category', 'obj_name', 'object_name']
                for key in category_keys:
                    if key in batch:
                        obj_category = batch[key]
                        break

                # ============================================================
                # UNWRAP NESTED TUPLES/LISTS RECURSIVELY (FIX)
                # ============================================================
                # Dataset may return:
                # - ('cheez-it box',)           → single-element tuple
                # - [('cheez-it box',)]         → list of tuples
                # - [[('cheez-it box',)]]       → nested list
                # We need to unwrap ALL layers until we get a string

                while isinstance(obj_category, (list, tuple)) and len(obj_category) > 0:
                    obj_category = obj_category[0]

                # Handle tensor
                if isinstance(obj_category, torch.Tensor):
                    obj_category = obj_category.item() if obj_category.numel() == 1 else obj_category[0].item()

                # NOW convert to string (after unwrapping)
                if obj_category is not None:
                    # At this point obj_category should be a plain string like 'cheez-it box'
                    # or 'Cheez-It Box' (with capitals)
                    obj_category = str(obj_category).strip().lower()

                    # Remove any lingering tuple artifacts (paranoid check)
                    # "('cheez-it box',)" → "cheez-it box"
                    obj_category = obj_category.strip("()\"' ")

                    if self.global_step == obj_chamfer_start:
                        logger.info(f"[Object Chamfer] Cleaned category: '{obj_category}'")

                    # ============================================================
                    # FUZZY CATEGORY NORMALIZATION
                    # ============================================================
                    category_to_template = {
                        # Boxes
                        'cheez-it box': 'cracker_box',
                        'cheez-it cracker box': 'cracker_box',
                        'cracker box': 'cracker_box',

                        # Fruits
                        'banana': 'banana',

                        # Tools
                        'power drill': 'power_drill',
                        'drill': 'power_drill',

                        # Bottles
                        "french's mustard bottle": 'mustard_bottle',
                        'mustard bottle': 'mustard_bottle',
                        'mustard': 'mustard_bottle',

                        # Generic boxes (map to gelatin_box as closest match)
                        'game/media box': 'gelatin_box',
                        'game/product box': 'gelatin_box',
                        'game box': 'gelatin_box',
                        'product box': 'gelatin_box',
                        'small rectangular object': 'gelatin_box',
                        'package box': 'gelatin_box',

                        # Additional objects
                        'chips can': 'chips_can',
                        'pringles': 'chips_can',
                        'tomato soup can': 'tomato_soup_can',
                        'soup can': 'tomato_soup_can',
                        'scissors': 'scissors',
                        'hammer': 'hammer',
                        'foam brick': 'foam_brick',
                        'pear': 'pear',
                        'strawberry': 'strawberry',
                        'tennis ball': 'tennis_ball',
                    }

                    original_category = obj_category
                    if obj_category in category_to_template:
                        obj_category = category_to_template[obj_category]
                        if self.global_step % 1000 == 0:
                            logger.debug(f"[Object Chamfer] Mapped '{original_category}' → '{obj_category}'")

                # ============================================================
                # STEP 3: Check Template Availability & Compute Chamfer
                # ============================================================

                # Debugging at first step
                if self.global_step == obj_chamfer_start:
                    logger.info(f"[Object Chamfer] Checking template for: '{obj_category}'")
                    logger.info(f"[Object Chamfer] Available: {list(self.object_templates.keys())}")
                    if obj_category in self.object_templates:
                        logger.info(f"[Object Chamfer] ✅ Template FOUND for '{obj_category}'")
                    else:
                        logger.warning(f"[Object Chamfer] ⚠️  Template NOT FOUND for '{obj_category}'")

                if obj_category in self.object_templates:
                    # ============================================================
                    # VALIDATE SDF BEFORE EXTRACTION
                    # ============================================================
                    # Before line 2673
                    if self.global_step % 100 == 0:
                        logger.info(f"[PRE-VALIDATION] Step {self.global_step}")
                        logger.info(f"  Has model.nodes: {hasattr(self.model, 'nodes')}")
                        if hasattr(self.model, 'nodes'):
                            logger.info(f"  'object' in nodes: {'object' in self.model.nodes}")
                    # Initialize flag
                    sdf_is_valid = False

                    try:
                        # ============================================================
                        # FIX: ModuleDict doesn't support .get(), use 'in' check
                        # ============================================================
                        if 'object' in self.model.nodes:
                            object_node = self.model.nodes['object']
                            if hasattr(object_node, 'server'):
                                if hasattr(object_node.server, 'object_model'):
                                    # Check for SDF grid OR vertex representation
                                    if hasattr(object_node.server.object_model, 'sdf_grid'):
                                        sdf = object_node.server.object_model.sdf_grid
                                        sdf_std = sdf.std().item()
                                        sdf_min = sdf.min().item()
                                        sdf_max = sdf.max().item()
                                        has_zero_crossing = (sdf_min < 0.0) and (sdf_max > 0.0)

                                        # min_std_threshold = 1e-5 if self.global_step < 10000 else 1e-4
                                        # More lenient threshold to allow early learning
                                        min_std_threshold = 1e-6 if self.global_step < 10000 else 1e-5
                                        sdf_is_valid = has_zero_crossing and sdf_std > min_std_threshold

                                        # Log SDF stats every 100 steps (more frequent monitoring)
                                        if self.global_step % 100 == 0:
                                            logger.info(
                                                f"[SDF Stats] Step {self.global_step}: "
                                                f"std={sdf_std:.6f} (threshold={min_std_threshold:.6f}), "
                                                f"range=[{sdf_min:.4f}, {sdf_max:.4f}], "
                                                f"zero_crossing={has_zero_crossing}, valid={sdf_is_valid}"
                                            )

                                            # Log to tensorboard for plotting
                                            logger.info('sdf/std', sdf_std)
                                            logger.info('sdf/min', sdf_min)
                                            logger.info('sdf/max', sdf_max)
                                            logger.info('sdf/has_zero_crossing', float(has_zero_crossing))
                                            logger.info('sdf/is_valid', float(sdf_is_valid))

                                        if not sdf_is_valid and self.global_step % 500 == 0:
                                            logger.warning(
                                                f"[Object Chamfer] DEGENERATE SDF at step {self.global_step}:\n"
                                                f"  std={sdf_std:.6f} (need >1e-4), range=[{sdf_min:.4f}, {sdf_max:.4f}]\n"
                                                f"  zero_crossing={has_zero_crossing} → Skipping mesh extraction"
                                            )
                                    elif hasattr(object_node.server.object_model, 'v3d_cano'):
                                        # Vertex representation is available (for fallback/use later),
                                        # but do NOT flip sdf_is_valid here; it is strictly SDF-based.
                                        pass
                    except Exception as e:
                        if self.global_step % 100  == 0:
                            logger.warning(f"[Object Chamfer] SDF validation error: {e}")

                    # Only extract mesh if valid
                    if sdf_is_valid:
                        # Before extraction
                            if self.global_step % 100 == 0:
                                logger.info(f"[Pre-Extract Debug] Step {self.global_step}")

                                if hasattr(self.model, 'nodes') and 'object' in self.model.nodes:
                                    obj_node = self.model.nodes['object']
                                    if hasattr(obj_node, 'server') and hasattr(obj_node.server, 'object_model'):
                                        obj_model = obj_node.server.object_model

                                        logger.info(f"  Has v3d_cano: {hasattr(obj_model, 'v3d_cano')}")
                                        if hasattr(obj_model, 'v3d_cano'):
                                            logger.info(f"  v3d_cano shape: {obj_model.v3d_cano.shape}")
                                            logger.info(
                                                f"  v3d_cano range: [{obj_model.v3d_cano.min():.4f}, {obj_model.v3d_cano.max():.4f}]")

                                        logger.info(f"  Has f3d_cano: {hasattr(obj_model, 'f3d_cano')}")
                                        if hasattr(obj_model, 'f3d_cano'):
                                            logger.info(f"  f3d_cano shape: {obj_model.f3d_cano.shape}")

                                        logger.info(f"  Has faces: {hasattr(obj_model, 'faces')}")
                                        if hasattr(obj_model, 'faces') and obj_model.faces is not None:
                                            logger.info(f"  faces shape: {obj_model.faces.shape}")

                            # Try to extract object mesh from SDF (largest component only)
                            try:
                                obj_verts_list, obj_faces_list = self._extract_object_mesh_from_sdf(batch)
                            except Exception as e:
                                logger.error(f"[Mesh Extract EXCEPTION] {e}")
                                import traceback
                                logger.error(traceback.format_exc())
                                obj_verts_list = None
                                obj_faces_list = None

                            # NEW: Detailed extraction logging
                            if self.global_step % 100 == 0:
                                extraction_success = (
                                    isinstance(obj_verts_list, list)
                                    and len(obj_verts_list) > 0
                                    and obj_verts_list[0] is not None
                                    and obj_verts_list[0].numel() > 0
                                )
                                logger.info(
                                    f"[Object Mesh Extract] step={self.global_step}, "
                                    f"success={extraction_success}"
                                )

                            # ============================================================
                            # NEW: Object compactness loss (anti-floater / tight block)
                            # Uses extracted verts to encourage a single, compact block.
                            # ============================================================
                            w_obj_compact = getattr(self.opt.loss, "w_obj_compactness", 0.0)

                            if (
                                w_obj_compact > 0.0
                                and isinstance(obj_verts_list, list)
                                and len(obj_verts_list) > 0
                                and obj_verts_list[0] is not None
                                and obj_verts_list[0].numel() > 0
                            ):
                                compact_losses = []

                                for b_idx, verts_b in enumerate(obj_verts_list):
                                    # verts_b: [V, 3] in approximately [-1.5, 1.5]^3 after extraction
                                    if verts_b is None or verts_b.numel() == 0:
                                        continue

                                    # Center and radius
                                    center = verts_b.mean(dim=0, keepdim=True)           # [1, 3]
                                    dists = (verts_b - center).norm(dim=-1)             # [V]

                                    # Encourage most mass inside a target radius (e.g. 1.0 in [-1.5,1.5])
                                    target_radius = 1.0
                                    max_dist = dists.max()
                                    # Penalize outliers far from center
                                    radius_penalty = torch.nn.functional.relu(max_dist - target_radius)

                                    # Optional: encourage non-degenerate volume (avoid super-thin shells)
                                    bbox_min, _ = verts_b.min(dim=0)
                                    bbox_max, _ = verts_b.max(dim=0)
                                    bbox_size = bbox_max - bbox_min                   # [3]
                                    volume = bbox_size.prod()
                                    # Target a moderate volume; adjust 0.01–0.1 depending on scale
                                    target_volume = 0.02
                                    volume_penalty = torch.abs(volume - target_volume)

                                    compact_loss_b = 0.7 * radius_penalty + 0.3 * volume_penalty
                                    compact_losses.append(compact_loss_b)

                                    if self.global_step % 200 == 0:
                                        logger.info(
                                            f"[Obj Compactness] step={self.global_step}, "
                                            f"batch={b_idx}, max_dist={float(max_dist):.4f}, "
                                            f"radius_penalty={float(radius_penalty):.4f}, "
                                            f"volume={float(volume):.6f}, "
                                            f"volume_penalty={float(volume_penalty):.4f}"
                                        )

                                if len(compact_losses) > 0:
                                    obj_compactness_loss = torch.stack(compact_losses).mean()
                                    geometric_loss = geometric_loss + w_obj_compact * obj_compactness_loss
                                    loss_output["loss/obj_compactness"] = float(obj_compactness_loss.item())

                                    if self.global_step % 100 == 0:
                                        logger.info(
                                            f"[Obj Compactness] step={self.global_step}, "
                                            f"loss={obj_compactness_loss.item():.6f}, "
                                            f"w={w_obj_compact:.3f}"
                                        )

                            # ============================================================
                            # Object Template Chamfer (single, unified path)
                            # ============================================================
                            try:
                                # Check if template exists
                                if obj_category in self.object_templates:
                                    # Choose verts to use for Chamfer:
                                    # 1) Prefer extracted mesh from SDF/MC (obj_verts_list)
                                    # 2) Fallback to v3d_cano if extraction failed
                                    obj_verts_for_chamfer = None

                                    if (
                                        isinstance(obj_verts_list, list)
                                        and len(obj_verts_list) > 0
                                        and obj_verts_list[0] is not None
                                        and obj_verts_list[0].numel() > 0
                                    ):
                                        obj_verts = obj_verts_list[0]
                                        if obj_verts.shape[0] > 0:
                                            obj_verts_for_chamfer = obj_verts

                                    # Fallback: use v3d_cano if SDF/MC failed
                                    if obj_verts_for_chamfer is None:
                                        if hasattr(self.model, 'nodes') and 'object' in self.model.nodes:
                                            obj_node = self.model.nodes['object']
                                            if hasattr(obj_node, 'server') and hasattr(obj_node.server, 'object_model'):
                                                obj_model = obj_node.server.object_model
                                                if hasattr(obj_model, 'v3d_cano'):
                                                    v3d = obj_model.v3d_cano
                                                    if v3d is not None and v3d.numel() > 0:
                                                        obj_verts_for_chamfer = v3d
                                                        if self.global_step % 100 == 0:
                                                            logger.info(
                                                                f"[Object Chamfer] Using v3d_cano fallback for Chamfer at step {self.global_step}"
                                                            )

                                    if obj_verts_for_chamfer is not None and obj_verts_for_chamfer.shape[0] > 0:
                                        template_verts = self.object_templates[obj_category]

                                        # Normalize both to zero-mean, unit scale
                                        obj_verts_centered = obj_verts_for_chamfer - obj_verts_for_chamfer.mean(
                                            dim=0, keepdim=True
                                        )
                                        template_centered = template_verts - template_verts.mean(
                                            dim=0, keepdim=True
                                        )

                                        obj_scale = obj_verts_centered.norm(dim=1).max()
                                        template_scale = template_centered.norm(dim=1).max()

                                        obj_verts_norm = obj_verts_centered / (obj_scale + 1e-6)
                                        template_verts_norm = template_centered / (template_scale + 1e-6)

                                        from pytorch3d.loss import chamfer_distance

                                        chamfer_loss, _ = chamfer_distance(
                                            obj_verts_norm.unsqueeze(0),
                                            template_verts_norm.unsqueeze(0)
                                        )

                                        geometric_loss = geometric_loss + obj_chamfer_weight * chamfer_loss
                                        loss_output['loss/obj_chamfer'] = chamfer_loss.item()

                                        if self.global_step % 100 == 0:
                                            logger.info(
                                                f"[Object Chamfer] Step {self.global_step}: "
                                                f"loss={chamfer_loss.item():.6f}, "
                                                f"category={obj_category}, "
                                                f"pred_verts={obj_verts_for_chamfer.shape[0]}, "
                                                f"template_verts={template_verts.shape[0]}, "
                                                f"weight={obj_chamfer_weight:.3f}"
                                            )
                                    else:
                                        if self.global_step % 500 == 0:
                                            logger.warning("[Object Chamfer] No valid verts for Chamfer (extraction + v3d_cano failed)")

                                else:
                                    # Template not available - generic prior will be used later
                                    if self.global_step == obj_chamfer_start:
                                        logger.info(
                                            f"[Object Chamfer] No template for '{obj_category}', will use generic prior"
                                        )

                            except Exception as e:
                                # Error handling for Chamfer
                                if self.global_step % 500 == 0:
                                    logger.error(
                                        f"[Object Chamfer] Failed at step {self.global_step}: {e}"
                                    )
                                    import traceback
                                    logger.error(traceback.format_exc())

                # ============================================================
                # GENERIC SHAPE PRIOR (Fallback for sequences without templates)
                # ============================================================

                # FIX: Now template_applied, obj_verts_list are guaranteed to exist
                if not template_applied and obj_verts_list and len(obj_verts_list) > 0:
                    try:
                        obj_verts = obj_verts_list[0]
                        obj_faces = obj_faces_list[0] if obj_faces_list else None

                        if obj_verts.shape[0] > 10 and obj_faces is not None and obj_faces.shape[0] > 0:
                            from pytorch3d.loss import mesh_laplacian_smoothing

                            obj_mesh = Meshes(verts=[obj_verts], faces=[obj_faces])

                            # Smoothness loss (penalize wrinkled surfaces)
                            smoothness_loss = mesh_laplacian_smoothing(obj_mesh, method='uniform')

                            # Compactness loss (penalize unrealistic volumes)
                            obj_centered = obj_verts - obj_verts.mean(dim=0)
                            bbox_size = obj_centered.max(dim=0)[0] - obj_centered.min(dim=0)[0]
                            volume_loss = torch.abs(bbox_size.prod() - 0.001)  # Target ~10cm³

                            # Combined generic loss
                            generic_weight = getattr(self.opt.loss, 'w_obj_generic', 0.05)
                            generic_loss = generic_weight * (0.5 * smoothness_loss + 0.5 * volume_loss)

                            geometric_loss = geometric_loss + generic_loss
                            loss_output['loss/obj_generic'] = generic_loss.item()

                            if self.global_step % 100 == 0:
                                logger.info(
                                    f"[Object Prior] Generic regularization: "
                                    f"smoothness={smoothness_loss.item():.6f}, "
                                    f"volume={volume_loss.item():.6f}"
                                )

                    except Exception as e:
                        if self.global_step % 500 == 0:
                            logger.warning(f"[Object Generic] Failed: {e}")

        # ================================================================
        # ✅ OBJECT SMOOTHNESS REGULARIZATION
        # ================================================================
        smoothness_added = False

        # Debug logging (only first step to avoid spam)
        if self.global_step == 0:
            logger.info("[DEBUG] Checking for object smoothness regularization...")
            logger.info(f"[DEBUG] Model type: {type(self.model).__name__}")
            logger.info(f"[DEBUG] Has 'nodes'? {hasattr(self.model, 'nodes')}")
            if hasattr(self.model, 'nodes'):
                logger.info(f"[DEBUG] Node keys: {list(self.model.nodes.keys())}")

        # Attempt 1: Via model.nodes (most likely path)
        if hasattr(self.model, 'nodes') and 'object' in self.model.nodes:
            object_node = self.model.nodes['object']

            if hasattr(object_node, 'server'):
                object_server = object_node.server

                if hasattr(object_server, 'object_model'):
                    object_model = object_server.object_model

                    if hasattr(object_model, 'v3d_cano'):
                        v3d_cano = object_model.v3d_cano

                        # Check if it's a Parameter (learnable)
                        if isinstance(v3d_cano, nn.Parameter):
                            # Compute smoothness loss
                            smoothness_weight = getattr(self.opt.loss, 'w_obj_smoothness', 0.0)
                            if smoothness_weight > 0.0:
                                # Compute smoothness loss only if non-zero weight
                                smoothness_loss = get_smoothness_loss(v3d_cano)
                                geometric_loss = geometric_loss + smoothness_weight * smoothness_loss
                                smoothness_added = True

                                # Log every 100 steps
                                if self.global_step % 100 == 0:
                                    logger.info('loss/object_smoothness', smoothness_loss.item())

                                    # Calculate change rate
                                    current_smooth = smoothness_loss.item()
                                    if not hasattr(self, '_prev_smoothness'):
                                        self._prev_smoothness = current_smooth
                                        change_rate = 0.0
                                    elif abs(self._prev_smoothness) < 1e-8:
                                        # Avoid division by zero when previous smoothness is ~0
                                        change_rate = 0.0
                                        self._prev_smoothness = current_smooth
                                    else:
                                        change_rate = (current_smooth - self._prev_smoothness) / self._prev_smoothness * 100
                                        self._prev_smoothness = current_smooth

                                    logger.info(
                                        f"✅ [Object Smoothness] Step {self.global_step}: "
                                        f"loss={smoothness_loss.item():.6f}, "
                                        f"change={change_rate:+.2f}%"
                                    )

                                    loss_output['object_smoothness'] = smoothness_loss.item()
                        else:
                            if self.global_step == 0:
                                logger.warning("⚠️ v3d_cano exists but is NOT nn.Parameter (still a buffer?)")

        # Attempt 2: Via model.servers (fallback)
        if not smoothness_added and hasattr(self.model, 'servers') and 'object' in self.model.servers:
            object_server = self.model.servers['object']

            if hasattr(object_server, 'object_model'):
                object_model = object_server.object_model

                if hasattr(object_model, 'v3d_cano'):
                    v3d_cano = object_model.v3d_cano

                    if isinstance(v3d_cano, nn.Parameter):
                        smoothness_loss = get_smoothness_loss(v3d_cano)
                        geometric_loss = geometric_loss + 0.001 * smoothness_loss

                        smoothness_added = True

                        if self.global_step % 100 == 0:
                            logger.info('loss/object_smoothness', smoothness_loss.item())
                            logger.info(f"✅ Object smoothness loss: {smoothness_loss.item():.6f}")
                            loss_output['object_smoothness'] = smoothness_loss.item()

        # Final check: Log if smoothness wasn't added
        if self.global_step == 0 and not smoothness_added:
            logger.error("❌ Object smoothness regularization NOT ACTIVE")
            logger.error("   Possible reasons:")
            logger.error("   1. Object node doesn't exist")
            logger.error("   2. v3d_cano is still a buffer (not nn.Parameter)")
            logger.error("   3. Path to object_model is different")

        # ================================================================
        # SDF VALIDITY REGULARIZATION (Forces proper zero-crossings)
        # ================================================================
        if hasattr(self.model, "nodes") and "object" in self.model.nodes:
            object_node = self.model.nodes["object"]

            if hasattr(object_node, "server"):
                object_server = object_node.server

                if hasattr(object_server, "object_model") and hasattr(object_server.object_model, "sdf_grid"):
                    sdf_grid = object_server.object_model.sdf_grid
                    w_sdf_reg = getattr(self.opt.loss, "w_sdf_reg", 1.0)
                    w_eikonal = getattr(self.opt.loss, "w_eikonal", 1.0)
                    w_smooth = getattr(self.opt.loss, "w_smooth", 1.0)

                    # Stronger SDF early
                    if self.global_step < sdf_warmup_end:
                        # Strong SDF only in early warmup
                        w_sdf_reg *= 3.0
                        w_eikonal *= 3.0
                    elif getattr(self, "_sdf_is_valid", False):
                        # After warmup + valid SDF: decay SDF terms
                        decay = 0.2  # e.g. 20% of base strength
                        w_sdf_reg *= decay
                        w_eikonal *= decay
                        w_smooth *= decay

                    # ---------- 1) HEALTH + DIAGNOSTICS (ALWAYS) ----------
                    sdf_min = sdf_grid.min()
                    sdf_max = sdf_grid.max()
                    sdf_std = sdf_grid.std()
                    with torch.no_grad():
                        # Simple 1D zero-crossing count along flattened grid
                        flat = sdf_grid.view(-1)
                        zero_crossings = ((flat[:-1] * flat[1:]) < 0).sum().item()
                        zero_cross_ratio = zero_crossings / max(flat.numel() - 1, 1)

                    loss_output["debug/sdf_zero_crossings"] = float(zero_crossings)
                    loss_output["debug/sdf_zero_cross_ratio"] = float(zero_cross_ratio)

                    zc_x = ((sdf_grid[:, 1:, :, :] * sdf_grid[:, :-1, :, :]) < 0).sum()
                    zc_y = ((sdf_grid[:, :, 1:, :] * sdf_grid[:, :, :-1, :]) < 0).sum()
                    zc_z = ((sdf_grid[:, :, :, 1:] * sdf_grid[:, :, :, :-1]) < 0).sum()
                    zero_cross_count = (zc_x + zc_y + zc_z).item()
                    # Teacher/grid metric
                    loss_output["loss/sdf_zero_crossings"] = float(zero_cross_count)
                    # Student/implicit metric – mirror grid value for now
                    loss_output["loss/implicit_sdf_zero_crossings"] = float(zero_cross_count)
                    # NEW: choose thresholds for std and zero-cross count
                    # Use config-driven warm-up boundary
                    warmup_end = getattr(self.opt.training, "sdf_warmup_end", 2000)
                    has_zero = (sdf_min < 0.0) and (sdf_max > 0.0)

                    if self.global_step < warmup_end:
                        # Early: only require non-trivial std and sign span
                        std_thresh = 1e-6
                        is_degenerate = (not has_zero) or (sdf_std.item() < std_thresh)
                        min_zc = 0  # for loss/penalty only
                    else:
                        # Later: add zero-cross count constraint
                        std_thresh = 1e-5
                        min_zc = 5
                        is_degenerate = (
                                (not has_zero)
                                or (zero_cross_count < min_zc)
                                or (sdf_std.item() < std_thresh)
                        )

                    self._sdf_is_valid = not is_degenerate

                    zero_cross_count_tensor = torch.tensor(float(zero_cross_count), device=sdf_grid.device)
                    zc_target = float(min_zc)
                    lambda_zc = 0.1
                    zero_cross_count_loss = lambda_zc * F.relu(zc_target - zero_cross_count_tensor)

                    if self.global_step % 100 == 0:
                        logger.info("sdf/min", sdf_min.item())
                        logger.info("sdf/max", sdf_max.item())
                        logger.info("sdf/std", sdf_std.item())
                        logger.info("sdf/has_zero", float(has_zero))
                        logger.info("sdf/zero_cross_count", zero_cross_count)

                    # Distillation from SDF grid into implicit network (DISABLED for Option B)
                    use_sdf_distill = getattr(self.opt.training, "use_sdf_distill", False)
                    if use_sdf_distill:
                        min_distill_step = getattr(self.opt.training, "distill_start_step", 4000)
                        min_grid_zero_cross = getattr(self.opt.training, "distill_min_grid_zeros", 2000)
                        if has_zero and self.global_step > min_distill_step and zero_cross_count > min_grid_zero_cross:
                            implicit_zc = getattr(self, "last_implicit_zero_crossings", 1000)
                            if implicit_zc < 300 and zero_cross_count > 10000:
                                self.sdf_distill_boost_steps = getattr(self, "sdf_distill_boost_steps", 0) + 500

                            geometric_loss = self.addsdfdistillationloss(
                                object_model=object_server.object_model,
                                batch=batch,
                                step=self.global_step,
                                geometric_loss=geometric_loss,
                                loss_output=loss_output,
                            )

                    # SDS soft gate and degeneracy counter / re-init
                    if hasattr(self, "_sdf_is_valid") and not self._sdf_is_valid:
                        old_sds_mult = sds_mult
                        sds_mult *= 0.5
                        logger.warning(
                            f"[SDF-HEALTH] SDF invalid at step {self.global_step}: "
                            f"scaling sds_mult {old_sds_mult:.3f} → {sds_mult:.3f} (soft gate)"
                        )

                    if not hasattr(self, "_sdf_bad_count"):
                        self._sdf_bad_count = 0
                    self._sdf_bad_count = self._sdf_bad_count + 1 if is_degenerate else 0

                    if self._sdf_bad_count >= 5 and self.global_step > 1000:
                        try:
                            # You likely already have a helper like this; if not, keep your old re-init code here.
                            self._reinit_object_sdf_from_template(object_server.object_model, batch)
                            logger.warning(
                                f"[SDF Reg] Reinitialized SDF from template after {self._sdf_bad_count} bad steps."
                            )
                        except Exception as e:
                            logger.error(f"[SDF Reg] SDF re-init failed: {e}")
                        finally:
                            self._sdf_bad_count = 0

                    # --------------------------------------------------------
                    # 1. Zero-crossing encouragement (mean ≈ 0)
                    # --------------------------------------------------------
                    mean_sdf = sdf_grid.mean()
                    zero_crossing_loss = torch.abs(mean_sdf) + F.relu(1e-3 - sdf_std)

                    # 2. Gradient magnitude / eikonal / range
                    grad_x = sdf_grid[:, 1:, :, :] - sdf_grid[:, :-1, :, :]
                    grad_y = sdf_grid[:, :, 1:, :] - sdf_grid[:, :, :-1, :]
                    grad_z = sdf_grid[:, :, :, 1:] - sdf_grid[:, :, :, :-1]
                    gradient_magnitude = (grad_x.pow(2).mean() + grad_y.pow(2).mean() + grad_z.pow(2).mean()).sqrt()
                    gradient_loss = F.relu(0.1 - gradient_magnitude) * 0.5
                    smoothness_loss = gradient_magnitude  # NEW: define smoothness_loss for SDF block

                    # --------------------------------------------------------
                    # 3. Eikonal loss (|∇φ| ≈ 1)
                    # --------------------------------------------------------
                    eikonal_loss = (gradient_magnitude - 1.0).pow(2) * 0.1

                    # --------------------------------------------------------
                    # 4. Range encouragement (span positive/negative)
                    #    Push min < -0.1 and max > 0.1, softly
                    # --------------------------------------------------------
                    range_loss = (F.relu(0.1 + sdf_min) + F.relu(0.1 - sdf_max)) * 0.5

                # --------------------------------------------------------
                    # 5. SDF–v3d coupling: encourage SDF(v3d_cano) ≈ 0
                    # --------------------------------------------------------
                    sdf_on_verts_loss = 0.0
                    v_std = None
                    if hasattr(object_server, "object_model") and hasattr(object_server.object_model, "v3d_cano"):
                        v3d = object_server.object_model.v3d_cano
                        if isinstance(v3d, torch.Tensor) and v3d.numel() > 0:
                            v_std = v3d.std().item()
                            coords = v3d.unsqueeze(0).unsqueeze(0)[..., [2, 1, 0]]
                            sdf_samples = F.grid_sample(
                                sdf_grid.unsqueeze(0),
                                coords.view(1, 1, -1, 1, 3),
                                align_corners=True,
                                mode="bilinear",
                                padding_mode="border",
                            )
                            sdf_on_verts_loss = sdf_samples.abs().mean()

                    w_v2s_base = getattr(self.opt.loss, "w_sdf_on_verts", 5.0)
                    if self.global_step < sdf_warmup_end:
                        # Let SDF move more freely vs. noisy v3d at the very beginning
                        w_v2s = 0.1 * w_v2s_base
                    else:
                        # Softer coupling once SDF is at least somewhat healthy
                        if getattr(self, "_sdf_is_valid", False):
                            w_v2s = 5.0 * w_v2s_base    # was 20.0
                        else:
                            w_v2s = 1.0 * w_v2s_base    # light pull even when invalid

                    # Finally, kill coupling if v3d is clearly degenerate
                    if v_std is not None and v_std < 0.01:  # was 0.02
                        w_v2s = 0.0

                    # --------------------------------------------------------
                    # Combine with configurable weights
                    # --------------------------------------------------------
                    # ---------- 2) DECIDE IF WE BACKPROP THROUGH sdf_grid ----------
                    can_reg = isinstance(sdf_grid, nn.Parameter) and sdf_grid.requires_grad

                    if can_reg:
                        # zero_cross_count_loss, gradient_loss, eikonal_loss, range_loss, v2s coupling
                        sdf_regularization = (
                                w_sdf_reg * (zero_cross_count_loss + range_loss) +
                                w_eikonal * eikonal_loss +
                                w_smooth * smoothness_loss
                        )
                        geometric_loss = geometric_loss + sdf_regularization

                        if 'loss_components' in locals():
                            loss_components['object_shape'] += sdf_regularization.item()
                        loss_output['loss/sdf_regularization'] = sdf_regularization.item()
                        loss_output['loss/sdf_zero_cross'] = zero_crossing_loss.item()
                        loss_output['loss/sdf_gradient'] = gradient_loss.item()
                        loss_output['loss/eikonal'] = eikonal_loss.item()
                        loss_output['loss/sdf_range'] = range_loss.item()
                        loss_output["loss/sdf_on_verts"] = float(sdf_on_verts_loss)

                        if self.global_step % 100 == 0:
                            logger.info('loss/sdf_zero_crossing', zero_crossing_loss.item())
                            logger.info('loss/sdf_gradient', gradient_loss.item())
                            logger.info('loss/sdf_eikonal', eikonal_loss.item())
                            logger.info('loss/sdf_range', range_loss.item())

                    else:
                        # Non-learnable / derived SDF: only use as diagnostic, no gradients
                        if self.global_step % 500 == 0:
                            logger.info(
                                "[SDF Reg] Skipping SDF regularization; sdf_grid is treated as "
                                f"derived/cached (is_param={can_reg})."
                            )
                        # Skip regularization
                    # B) Vertex coupling that should still move v3d_cano even if sdf_grid is derived
                    if sdf_on_verts_loss != 0.0 and w_v2s > 0.0:
                        v2s_term = w_v2s * sdf_on_verts_loss
                        geometric_loss = geometric_loss + v2s_term
                        loss_output["loss/sdf_on_verts"] = float(v2s_term.item())
            # ================================================================
            # VERTEX-BASED REGULARIZATION (for v3d_cano approach)
            # ================================================================
            if hasattr(object_server, 'object_model') and hasattr(object_server.object_model, 'v3d_cano'):
                obj_model = object_server.object_model
                v3d_cano = obj_model.v3d_cano

                # ------------------------------------------------------------
                # Object geometry freeze / unfreeze schedule (config-driven)
                # ------------------------------------------------------------
                # Config keys (in training: section of YAML):
                #   training.object_freeze_at:    step to start freezing v3d_cano
                #   training.object_unfreeze_at:  step to re-enable gradients (optional)
                if isinstance(v3d_cano, nn.Parameter):
                    should_require_grad = True

                    if freeze_at is not None and self.global_step >= freeze_at:
                        # Within the freeze window until optional unfreeze_at
                        if unfreeze_at is None or self.global_step < unfreeze_at:
                            should_require_grad = False

                    if v3d_cano.requires_grad != should_require_grad:
                        v3d_cano.requires_grad = should_require_grad
                        state = "FROZEN" if not should_require_grad else "UNFROZEN"
                        logger.info(
                            f"[Object Freeze] Step {self.global_step}: "
                            f"set v3d_cano.requires_grad = {should_require_grad} ({state})"
                        )

                # Only apply vertex-based regularization when v3d_cano is trainable
                if isinstance(v3d_cano, nn.Parameter) and v3d_cano.requires_grad:
                    if v3d_cano.shape[0] > 100:  # Only for dense meshes
                        # --------------------------------------------------------
                        # Existing: vertex spread / anti-collapse regularization
                        # --------------------------------------------------------
                        # Average variance over XYZ
                        vertex_variance = v3d_cano.var(dim=0).mean()

                        # Hinge loss: penalize only if variance is TOO LOW
                        # Official HOLD v3d_cano std ≈ 0.29, so a floor like 0.05 is safe
                        min_var = 0.05
                        vertex_collapse_loss = torch.nn.functional.relu(min_var - vertex_variance)

                        # Stronger early, relaxed later
                        if self.global_step < 5000:
                            vertex_collapse_weight = 0.1
                        else:
                            vertex_collapse_weight = 0.02

                        # CHANGED: add into geometric_loss, not total_loss
                        geometric_loss = geometric_loss + vertex_collapse_weight * vertex_collapse_loss
                        loss_output["loss/vertex_collapse"] = float(vertex_collapse_loss.item())

                        if self.global_step % 100 == 0:
                            self.log('loss/vertex_collapse', vertex_collapse_loss.item())
                            self.log('debug/vertex_variance', vertex_variance.item())
                            logger.info(
                                f"✅ [Vertex spread] step={self.global_step}, "
                                f"loss={vertex_collapse_loss.item():.6f}, var={vertex_variance.item():.6f}"
                            )

                        # --------------------------------------------------------
                        # NEW: Vertex normal prior (template-aligned normals)
                        # --------------------------------------------------------
                        w_vn = getattr(self.opt.loss, "w_vertex_normal_prior", 0.0)
                        faces = getattr(obj_model, "faces", None)
                        template_normals = getattr(obj_model, "template_normals", None)

                        if (
                            w_vn > 0.0
                            and faces is not None
                            and template_normals is not None
                            and faces.numel() > 0
                        ):
                            try:
                                # Build a single-batch mesh and compute vertex normals
                                mesh = Meshes(verts=[v3d_cano], faces=[faces])
                                current_normals = mesh.verts_normals_padded()[0]  # [V, 3]

                                # Ensure shapes match; truncate to the minimum length
                                num_verts_current = current_normals.shape[0]
                                num_verts_template = template_normals.shape[0]
                                n = min(num_verts_current, num_verts_template)

                                current = current_normals[:n]
                                target = template_normals[:n].to(current.device)

                                cos_sim = torch.nn.functional.cosine_similarity(
                                    current, target, dim=-1, eps=1e-6
                                ).clamp(-1.0, 1.0)
                                normal_prior_loss = 1.0 - cos_sim.mean()

                                # CHANGED: add into geometric_loss, not total_loss
                                geometric_loss = geometric_loss + w_vn * normal_prior_loss
                                loss_output["loss/vertex_normal_prior"] = float(normal_prior_loss.item())

                                if self.global_step % 100 == 0:
                                    logger.info(
                                        f"[Vertex Normal Prior] step={self.global_step}, "
                                        f"loss={normal_prior_loss.item():.6f}, w={w_vn:.3f}"
                                    )

                            except Exception as e:
                                # Fail-safe: log but do not break training
                                if self.global_step % 500 == 0:
                                    logger.warning(f"[Vertex Normal Prior] Failed at step {self.global_step}: {e}")

        # In src/hold/hold.py, where losses are computed
        # Add L2 regularization on residual MLP to keep it near identity
        w_residual_l2 = getattr(self.opt.loss, 'w_residual_l2', 0.01)
        logger.info(
            f"[Residual L2] w_residual_l2={w_residual_l2}, has object_node={hasattr(self.model.nodes, 'object')}")
        if w_residual_l2 > 0 and hasattr(self.model, 'nodes'):
            object_node = self.model.nodes['object']
            if hasattr(object_node, 'server') and hasattr(object_node.server, 'object_model'):
                residual_mlp = object_node.server.object_model.residual_mlp
                residual_l2 = sum(p.pow(2).sum() for p in residual_mlp.parameters())
                geometric_loss += w_residual_l2 * residual_l2
                loss_output["loss/residual_l2"] = residual_l2.item()

        # Now that ALL geometry terms are in geometric_loss, apply scaling
        # self._geom_scale: auto controller (Phase 3)
        # geom_mult: warm-up phase multiplier
        # Effective geometry weight = self._geom_scale * geom_mult
        geometric_loss = geometric_loss * self._geom_scale

        # ================================================================
        # GEOMETRY CAP AROUND 4K TO AVOID BAD BASIN
        # ------------------------------------------------
        # Limit the effective geometry loss in the 3k–6k window so that
        # "Geometric Total" cannot spike into the ~0.4+ regime where the
        # optimizer falls into the bad hand-contact basin.
        # You can make geom_cap_max configurable via opt.training.geom_cap_max.
        # ================================================================
        if 3000 < self.global_step < 6000:
            # Max geometry loss (after _geom_scale, before geom_mult)
            geom_cap_max = getattr(self.opt.training, "geom_cap_max", 0.2)

            geo_before_cap = geometric_loss.item()
            if geo_before_cap > geom_cap_max:
                geometric_loss = torch.clamp(geometric_loss, max=geom_cap_max)

                # Optional: log only occasionally to avoid spam
                if self.global_step % 100 == 0:
                    logger.info(
                        f"[GEOM-CAP] Step {self.global_step}: "
                        f"clamped geometric_loss from {geo_before_cap:.6f} "
                        f"to {geom_cap_max:.6f}"
                    )

        # Rebuild total_loss from scaled base + scaled geometry
        total_loss = base_mult * base_loss + geom_mult * geometric_loss

        loss_output['geometric_loss'] = geometric_loss.item()
        loss_components['geometric'] = (geom_mult * geometric_loss).item()
        loss_components['base_raw'] = base_loss.item()
        loss_components['base'] = (base_mult * base_loss).item()

        logger.info(f"[GEOM-DEBUG] After geometry aggregation (scaled): {geometric_loss.item()}")

        # ========================================
        # MANO Joint Supervision (MODIFIED)
        # ========================================
        if False:
        # if hasattr(self.opt, 'training') and \
        #         getattr(self.opt.training, 'supervise_joints', False):
            from src.hold import loss_terms

            try:
                # ============================================================
                # Pre-compute GT joints using MANO (CORRECT VERSION)
                # ============================================================
                gt_joints_computed = False
                for hand_name in ['right', 'left']:
                    gt_pose_key = f'gt.{hand_name}.hand_pose'
                    if gt_pose_key in batch and hand_name in self.model.nodes:
                        node = self.model.nodes[hand_name]

                        # Extract GT params from batch
                        gt_pose_full = batch[gt_pose_key]  # [B, 48]
                        gt_trans = batch[f'gt.{hand_name}.hand_trans']  # [B, 3]
                        gt_shape = batch[f'gt.{hand_name}.hand_shape']  # [10] or [B, 10]

                        # Ensure gt_shape is batched [B, 10]
                        if gt_shape.ndim == 1:
                            gt_shape = gt_shape.unsqueeze(0).expand(gt_pose_full.shape[0], -1)

                        try:
                            with torch.no_grad():
                                # ========================================================
                                # Get scene_scale - CRITICAL MISSING PARAMETER!
                                # ========================================================
                                batch_size = gt_pose_full.shape[0]

                                # Option 1: Use model's scale attribute
                                if hasattr(node, 'scale'):
                                    scene_scale = torch.full(
                                        (batch_size,),
                                        node.scale,
                                        device=gt_pose_full.device
                                    )
                                # Option 2: Use default scale (usually 1.0)
                                else:
                                    scene_scale = torch.ones(
                                        batch_size,
                                        device=gt_pose_full.device
                                    )

                                # ========================================================
                                # Call MANO server with POSITIONAL arguments (CORRECT!)
                                # ========================================================
                                gt_output = node.server(
                                    scene_scale,  # arg1: [B] scene scaling
                                    gt_trans,  # arg2: [B, 3] translation
                                    gt_pose_full,  # arg3: [B, 48] full pose
                                    gt_shape  # arg4: [B, 10] shape
                                )

                                # Extract joints from output
                                if isinstance(gt_output, dict) and 'jnts' in gt_output:
                                    gt_joints = gt_output['jnts'] # [B, 21, 3]

                                    # Add to batch for loss computation
                                    batch[f'gt.j3d.{hand_name}'] = gt_joints
                                    gt_joints_computed = True

                                    if self.global_step % 100 == 0:
                                        logger.info(
                                            f"[GT Joints] Computed for {hand_name}: "
                                            f"shape={gt_joints.shape}, "
                                            f"range=[{gt_joints.min().item():.3f}, {gt_joints.max().item():.3f}]"
                                        )
                                else:
                                    logger.warning(
                                        f"[GT Joints] MANO output missing 'jnts' key. "
                                        f"Keys: {gt_output.keys() if isinstance(gt_output, dict) else type(gt_output)}"
                                    )

                        except Exception as e:
                            if self.global_step % 100 == 0:
                                logger.error(f"[GT Joints] Failed to compute for {hand_name}: {e}")
                                import traceback

                                logger.error(traceback.format_exc())

                # Compute joint supervision loss if GT joints available
                if gt_joints_computed:
                    joint_loss = loss_terms.get_joint_supervision_loss_v2(
                        model_outputs, batch, hand_node_name='right'
                    )

                    joint_loss_weight = getattr(self.opt.training, 'joint_loss_weight', 0.1)

                    if joint_loss > 0:
                        loss_output["loss"] = loss_output["loss"] + joint_loss * joint_loss_weight
                        loss_output["loss/joint_supervision"] = joint_loss

                        if self.global_step % 100 == 0:
                            logger.info(
                                f"[Joint Loss] {joint_loss.item():.4f} mm "
                                f"(weight: {joint_loss_weight}, "
                                f"contribution: {(joint_loss * joint_loss_weight).item():.4f})"
                            )
                else:
                    if self.global_step % 100 == 0:
                        logger.debug("[Joint Loss] GT joints not computed, skipping supervision")
            except Exception as e:
                logger.warning(f"[Joint Loss] Failed: {e}")

        if should_profile:
            self.memory_profiler.checkpoint("after_loss")

        # ================================================================
        # ADD: First-step diagnostic logging
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
        # PHASE 3 & PHASE 5: SDS COMPUTATION WITH MUTUAL EXCLUSIVITY
        if self.phase3_enabled and self.ghop_enabled:
            # Ensure Phase 3 bounds are initialized and long enough
            if not hasattr(self, "phase3_start_iter"):
                self.phase3_start_iter = 5000
            if not hasattr(self, "phase3_end_iter") or self.phase3_end_iter < 50000:
                self.phase3_end_iter = 50000

            # Log activation status at first step
            if self.global_step == 0:
                logger.info("="*70)
                logger.info("[GHOP] Phase 3 Configuration:")
                logger.info(f"  phase3_enabled: {self.phase3_enabled}")
                logger.info(f"  ghop_enabled: {self.ghop_enabled}")
                logger.info(f"  phase3_start_iter: {self.phase3_start_iter}")
                logger.info(f"  phase3_end_iter: {self.phase3_end_iter}")
                logger.info(f"  warmup_iters: {getattr(self, 'warmup_iters', 'NOT SET')}")
                logger.info(f"  w_sds: {getattr(self, 'w_sds', 'NOT SET')}")
                logger.info(f"  sds_iters: {getattr(self, 'sds_iters', 'NOT SET')}")
                logger.info(f"  Has ghop_manager: {hasattr(self, 'ghop_manager')}")
                logger.info("="*70)

            # CRITICAL: Check if GHOP should activate at this step
            should_compute_ghop = (
                self.global_step >= getattr(self, 'phase3_start_iter', 0) and
                self.global_step < getattr(self, 'phase3_end_iter', 99999) and  # ← ADD THIS
                self.global_step >= getattr(self, 'warmup_iters', 0) and
                (self.global_step % getattr(self, 'sds_iters', 1)) == 0
            )

            # Log phase transitions
            if self.global_step == self.phase3_start_iter:
                logger.info(f"[Phase 3] SDS diffusion ACTIVATED at step {self.global_step}")

            if self.global_step == self.phase3_end_iter:  # ← ADD THIS
                logger.info(f"[Phase 3] SDS diffusion DEACTIVATED at step {self.global_step}")
                logger.info(f"[Phase 5] Temporal consistency will activate at step {self.phase5_start_iter}")

            # Log every 50 steps
            if self.global_step % 50 == 0:
                logger.info(f"[GHOP] Step {self.global_step}: should_compute={should_compute_ghop}")

            # INITIALIZE: Default values
            ghop_losses = {}
            ghop_info = {"stage": "unknown", "stage_progress": 0.0, "error": None}
            weighted_ghop = torch.tensor(0.0, device=loss_output["loss"].device, requires_grad=True)

            # Force SDS off during geometry & rendering warm-ups
            if sds_disabled:
                should_compute_ghop = False

            if should_compute_ghop:
                logger.debug(f"[GHOP] ✅ Computing SDS loss at step {self.global_step}")
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
                    object_sdf = self._extract_sdf_grid_from_nodes(
                        batch,
                        resolution=self.grid_resolution
                    )

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
                    std_thresh = 1e-5 if self.global_step < self.opt.training.sdf_warmup_end else 5e-6
                    if sdf_std < std_thresh:
                        logger.warning(
                            f"Phase 3 objectsdf is weak/flat std={sdf_std:.6f} "
                            f"(thresh={std_thresh:.1e}) – skipping SDS this step."
                        )
                        # Soft‑skip: set a flag or just return 0 SDS, but DO NOT throw.
                        return 0.0, {}  # or equivalently: ghoplosses['sds'] = 0


                    logger.debug(
                        f"[Phase 3] ✓ Object SDF valid: shape={object_sdf.shape}, "
                        f"std={sdf_std:.4f}, range=[{object_sdf.min():.4f}, {object_sdf.max():.4f}]"
                    )

                    # ============================================================
                    # STEP 3: Get text prompt
                    # ============================================================
                    # The dataset already provides properly formatted prompts
                    # Priority: use pre-formatted prompts > manual construction

                    # Check if batch has pre-formatted prompts
                    if 'text_prompt' in batch and batch['text_prompt'] is not None:
                        # Use dataset's pre-formatted prompt
                        text_prompts = batch['text_prompt']
                        logger.debug(f"[TEXT-PROMPT] Using pre-formatted prompt from dataset")

                    elif 'text_prompt_detailed' in batch and batch['text_prompt_detailed'] is not None:
                        # Use detailed prompts if available
                        text_prompts = batch['text_prompt_detailed']
                        logger.debug(f"[TEXT-PROMPT] Using detailed prompt from dataset")

                    else:
                        # Fallback: construct manually from category
                        category = batch.get('category', batch.get('object_category', 'Object'))

                        logger.debug(f"[TEXT-PROMPT] Raw category from batch: {repr(category)}")

                        # Handle list of tuples: [('Cheez-It Box',), ('Cheez-It Box',)]
                        if isinstance(category, list):
                            # Extract first element if list
                            if len(category) > 0:
                                category = category[0]
                            else:
                                category = 'Object'

                        # Handle single tuple: ('Cheez-It Box',)
                        if isinstance(category, tuple):
                            category = category[0] if len(category) > 0 else 'Object'

                        # Clean string
                        category = str(category).strip()

                        # Remove any remaining artifacts
                        category = category.strip("()[]'\"")

                        # Remove duplicate prefix if present
                        if "a hand grasping a" in category.lower():
                            category = category.split("a hand grasping a", 1)[-1].strip()

                        text_prompts = [f"a hand grasping a {category}"] * B
                        logger.debug(f"[TEXT-PROMPT] Manually constructed prompt from category")

                    # Ensure text_prompts is a list
                    if isinstance(text_prompts, str):
                        text_prompts = [text_prompts] * B
                    elif not isinstance(text_prompts, list):
                        text_prompts = list(text_prompts)

                    # Verify batch size matches
                    if len(text_prompts) != B:
                        logger.warning(f"[TEXT-PROMPT] Prompt count ({len(text_prompts)}) != batch size ({B}), repeating first")
                        text_prompts = [text_prompts[0]] * B

                    logger.debug(f"[TEXT-PROMPT] Final prompts (raw): {text_prompts}")  # ← RENAMED

                    # ============================================================
                    # CRITICAL FIX: Unwrap tuple prompts from dataset
                    # ============================================================
                    # Dataset returns: [('prompt text',), ('prompt text',)]
                    # CLIP needs: ['prompt text', 'prompt text']

                    cleaned_prompts = []
                    for prompt in text_prompts:
                        if isinstance(prompt, tuple):
                            # Unwrap tuple: ('text',) -> 'text'
                            prompt = prompt[0] if len(prompt) > 0 else ''
                        cleaned_prompts.append(str(prompt))

                    text_prompts = cleaned_prompts
                    logger.debug(f"[TEXT-PROMPT] Cleaned prompts (after unwrapping): {text_prompts}")  # ← MOVED HERE

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

                    # Call ghop_manager with correct signature
                    ghop_losses, ghop_info = self.ghop_manager.compute_losses(
                        object_sdf=object_sdf,        # [B, 1, 64, 64, 64]
                        hand_params=hand_params,      # ✅ FIXED: Pass dict directly
                        text_prompts=text_prompts,    # List[str]
                        iteration=self.global_step    # int
                    )

                    # After line 2259, ADD:
                    logger.info(f"[GHOP-DEBUG] compute_losses returned: ghop_losses keys={list(ghop_losses.keys())}")
                    logger.info(f"[GHOP-DEBUG] SDS loss value: {ghop_losses.get('sds', 'KEY NOT FOUND')}")
                    logger.info(f"[GHOP-DEBUG] Info dict: {ghop_info}")

                    # Simple SDS-collapse warning: very low SDS loss can indicate degenerate geometry
                    sds_raw = float(ghop_losses.get('sds', 0.0))
                    if sds_raw < 0.01 and self.global_step > 5000:
                        logger.warning(
                            f"[SDS COLLAPSE WARNING] Step {self.global_step}: "
                            f"SDS loss is very low ({sds_raw:.6f}) – check geometry metrics/meshes for collapse."
                        )

                        # Reactive curriculum: gently perturb object geometry to escape bad basin
                        try:
                            obj_node = self.model.nodes.get("object", None)
                            if obj_node and hasattr(obj_node, "server") and hasattr(obj_node.server, "object_model"):
                                obj_model = obj_node.server.object_model
                                if hasattr(obj_model, "v3d_cano"):
                                    with torch.no_grad():
                                        noise_scale = 0.01
                                        obj_model.v3d_cano.add_(noise_scale * torch.randn_like(obj_model.v3d_cano))
                                    logger.warning(
                                        "[CURRICULUM] Injected small noise into v3d_cano due to SDS collapse."
                                    )
                        except Exception as e:
                            logger.error(f"[CURRICULUM] Failed to perturb v3d_cano on SDS collapse: {e}")

                        # Temporarily suppress object template Chamfer to loosen prior
                        self._obj_chamfer_suppressed_until = self.global_step + 1000
                        logger.warning(
                            f"[CURRICULUM] Temporarily suppressing object Chamfer until step "
                            f"{self._obj_chamfer_suppressed_until}"
                        )


                    # ============================================================
                    # STEP 5: Get SDS weight from scheduler (HIGHEST PRIORITY)
                    # ============================================================
                    # Priority order:
                    # 1. Scheduler weight (if enabled) - NEW
                    # 2. Phase 5 dynamic weights (if enabled)
                    # 3. Default weight = 1.0

                    # Get base SDS weight from scheduler
                    current_sds_weight = self.sds_weight_scheduler.get_weight(self.global_step)

                    # Apply Phase 5 multiplier ONLY if Phase 5 is actually active (after phase5_start_iter)
                    # Phase 5 scheduler provides loss_weights from step 0, but we only want to use them
                    # when Phase 5 temporal consistency is active (step >= 1950)
                    if (self.phase5_enabled and
                        'sds' in loss_weights and
                        self.global_step >= self.phase5_start_iter):

                        phase5_multiplier = loss_weights['sds']
                        final_sds_weight = current_sds_weight * phase5_multiplier
                        logger.debug(
                            f"[SDS-WEIGHT] Scheduler: {current_sds_weight:.4f} × "
                            f"Phase5: {phase5_multiplier:.4f} = {final_sds_weight:.4f}"
                        )
                    else:
                        # Before Phase 5 starts: use scheduler weight directly
                        final_sds_weight = current_sds_weight
                        if self.global_step % self.log_sds_weight_every == 0:
                            logger.debug(f"[SDS-WEIGHT] Using scheduler weight only: {final_sds_weight:.4f}")

                    # Apply weight to raw SDS loss
                    if 'sds' in ghop_losses:
                        raw_sds_loss = ghop_losses['sds']
                        weighted_ghop = raw_sds_loss * final_sds_weight

                        # Log at specified frequency
                        if self.global_step % self.log_sds_weight_every == 0:
                            logger.info(
                                f"[SDS-WEIGHT] Step {self.global_step}: "
                                f"weight={final_sds_weight:.4f}, "
                                f"raw_loss={raw_sds_loss.item():.6f}, "
                                f"weighted_loss={weighted_ghop.item():.6f}"
                            )

                        # Store components for logging
                        loss_output["loss/sds_raw"] = raw_sds_loss
                        loss_output["loss/sds_weighted"] = weighted_ghop
                        loss_output["sds_weight"] = final_sds_weight

                        logger.debug(
                            f"[Phase 3] SDS loss computed: "
                            f"raw={raw_sds_loss.item():.6f}, "
                            f"weight={final_sds_weight:.4f}, "
                            f"weighted={weighted_ghop.item():.6f}"
                        )
                    else:
                        weighted_ghop = torch.tensor(0.0, device=device, requires_grad=True)
                        logger.warning(f"[Phase 3] No 'sds' key in ghop_losses at step {self.global_step}")

                    # ============================================================
                    # STEP 6: Add GHOP losses to total loss
                    # ============================================================
                    # GHOP FIX: Conditional detach to prevent double backward
                    # Only detach if Phase 4 or Phase 5 are also active (multiple loss graphs)
                    phase4_active = (
                            hasattr(self, 'phase4_enabled') and
                            self.phase4_enabled and
                            self.contact_start_iter <= self.global_step < self.contact_end_iter
                    )
                    phase5_active = (hasattr(self, 'phase5_enabled') and self.phase5_enabled and
                                     self.global_step >= self.phase5_start_iter)

                    # Always detach GHOP loss to ensure consistent graph structure across all phases
                    scaled_ghop = sds_mult * weighted_ghop
                    total_loss = total_loss + scaled_ghop  # keep gradients
                    logger.debug(f"[Phase 3] Added GHOP loss WITH GRADIENTS")
                    loss_output["loss_sds"] = scaled_ghop
                    loss_output["loss/sds"] = scaled_ghop
                    loss_components['sds'] = scaled_ghop.item()
                    loss_output['ghop_loss'] = weighted_ghop.detach()  # Detach only for logging
                    logger.warning(f"[DEBUG] Line 1978 executed: ghop_loss = {loss_output['ghop_loss'].item():.6f}")


                    # ============================================================
                    # STEP 7: Log GHOP metrics
                    # ============================================================
                    stage_map = {'sds': 1, 'contact': 2, 'contact_only': 3, 'unknown': 0}
                    stage_numeric = stage_map.get(ghop_info.get('stage', 'unknown'), 0)

                    logger.info('ghop/stage_numeric', float(stage_numeric), prog_bar=False)
                    logger.info('ghop/stage_progress', ghop_info.get('stage_progress', 0.0), prog_bar=True)
                    logger.info('ghop/total_loss', weighted_ghop.item() if isinstance(weighted_ghop, torch.Tensor) else float(weighted_ghop), prog_bar=True)

                    # Log SDS-specific metrics
                    if 'sds' in ghop_losses:
                        sds_value = ghop_losses['sds']
                        if isinstance(sds_value, torch.Tensor):
                            logger.info('ghop/sds_loss', sds_value.item(), prog_bar=True)

                    # Console logging
                    if self.global_step % self.log_ghop_every == 0:
                        logger.info(
                            f"\n[Phase 3 - Step {self.global_step}] GHOP SDS:\n"
                            f"  Stage:            {ghop_info.get('stage', 'unknown')}\n"
                            f"  Stage progress:   {ghop_info.get('stage_progress', 0.0):.3f}\n"
                            f"  Total loss:       {weighted_ghop.item() if isinstance(weighted_ghop, torch.Tensor) else weighted_ghop:.4f}\n"
                            f"  SDS weight:       {sds_weight:.3f}"
                            f"  SDS weight:       {final_sds_weight:.3f}"
                        )

                    logger.debug(f"[Phase 3] ✓ GHOP losses added to total loss")

                except ValueError as e:
                    # Expected extraction failures
                    logger.warning(f"[GHOP] ❌ Extraction failed at step {self.global_step}: {e}")
                    ghop_info["error"] = str(e)
                    zero_loss = torch.tensor(0.0, device=loss_output["loss"].device, requires_grad=True)
                    logger.warning(
                        f"[DEBUG] Line 2013 about to overwrite ghop_loss (current: {loss_output.get('ghop_loss', 'NOT SET')})")
                    loss_output["ghop_loss"] = zero_loss

                except Exception as e:
                    # Unexpected errors
                    logger.error(f"[GHOP] ❌ CRITICAL ERROR at step {self.global_step}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    ghop_info["error"] = str(e)
                    zero_loss = torch.tensor(0.0, device=loss_output["loss"].device, requires_grad=True)
                    loss_output["ghop_loss"] = zero_loss

            # ============================================================
            # DIAGNOSTIC 1: RGB Convergence Check
            # ============================================================
            if self.sds_diagnostics.get('rgb_convergence_check', {}).get('enabled', False):
                check_config = self.sds_diagnostics['rgb_convergence_check']
                check_step = check_config.get('step', 1000)
                threshold = check_config.get('threshold', 0.08)
                warning_only = check_config.get('warning_only', True)

                if self.global_step == check_step:
                    # Extract RGB loss from loss_output
                    rgb_loss_value = loss_output.get('loss/rgb', loss_output.get('loss', 0.0))

                    if isinstance(rgb_loss_value, torch.Tensor):
                        rgb_loss_value = rgb_loss_value.item()

                    if rgb_loss_value > threshold:
                        msg = (
                            f"\n{'=' * 70}\n"
                            f"[RGB-CONVERGENCE-CHECK] FAILED at step {check_step}!\n"
                            f"{'=' * 70}\n"
                            f"  RGB loss:     {rgb_loss_value:.6f}\n"
                            f"  Threshold:    {threshold:.6f}\n"
                            f"  Gap:          +{(rgb_loss_value - threshold):.6f} ({((rgb_loss_value / threshold - 1) * 100):.1f}% over)\n"
                            f"\n"
                            f"  ⚠️  INTERPRETATION:\n"
                            f"  RGB loss should converge to < {threshold} by step {check_step}.\n"
                            f"  Current value indicates SDS weight may still be too high,\n"
                            f"  preventing proper geometric learning.\n"
                            f"\n"
                            f"  📋 RECOMMENDATION:\n"
                            f"  - Option 1: Reduce all SDS weights by 50% in config\n"
                            f"  - Option 2: Increase RGB weight to 2.0 or 5.0\n"
                            f"  - Option 3: Extend Phase 3 by 50% more steps\n"
                            f"{'=' * 70}\n"
                        )
                        if warning_only:
                            logger.warning(msg)
                        else:
                            logger.error(msg)
                            raise RuntimeError("RGB convergence check failed - stopping training")
                    else:
                        logger.info(
                            f"\n{'=' * 70}\n"
                            f"[RGB-CONVERGENCE-CHECK] ✅ PASSED at step {check_step}!\n"
                            f"{'=' * 70}\n"
                            f"  RGB loss: {rgb_loss_value:.6f} < {threshold:.6f}\n"
                            f"  Geometric learning is progressing well.\n"
                            f"  SDS weight balance appears appropriate.\n"
                            f"{'=' * 70}\n"
                        )

            # ============================================================
            # DIAGNOSTIC 2: SDS Trend Check (Detect Divergence)
            # ============================================================
            if self.sds_diagnostics.get('sds_trend_check', {}).get('enabled', False):
                # Initialize history dict if not exists
                if not hasattr(self, '_sds_history'):
                    self._sds_history = {}  # step -> raw_sds_loss

                check_config = self.sds_diagnostics['sds_trend_check']
                check_every = check_config.get('check_every', 200)
                window = check_config.get('window', 400)
                max_increase = check_config.get('max_increase', 0.2)

                # Store current SDS loss (raw, not weighted)
                if 'loss/sds_raw' in loss_output:
                    raw_sds = loss_output['loss/sds_raw']
                    if isinstance(raw_sds, torch.Tensor):
                        self._sds_history[self.global_step] = raw_sds.item()

                # Check trend at specified frequency
                if self.global_step % check_every == 0 and self.global_step >= window:
                    prev_step = self.global_step - window

                    if prev_step in self._sds_history and self.global_step in self._sds_history:
                        current_sds = self._sds_history[self.global_step]
                        prev_sds = self._sds_history[prev_step]

                        if prev_sds > 0:
                            increase_ratio = (current_sds - prev_sds) / prev_sds

                            if increase_ratio > max_increase:
                                logger.warning(
                                    f"\n{'=' * 70}\n"
                                    f"[SDS-TREND-CHECK] ⚠️  SDS LOSS INCREASING!\n"
                                    f"{'=' * 70}\n"
                                    f"  Step {prev_step}:        {prev_sds:.6f}\n"
                                    f"  Step {self.global_step}: {current_sds:.6f}\n"
                                    f"  Change:                  +{increase_ratio * 100:.1f}%\n"
                                    f"  Threshold:               +{max_increase * 100:.1f}%\n"
                                    f"\n"
                                    f"  ⚠️  INTERPRETATION:\n"
                                    f"  SDS loss should decrease or stay stable as training progresses.\n"
                                    f"  Increasing SDS indicates the diffusion model is rejecting\n"
                                    f"  the current geometry as implausible - object shape may be\n"
                                    f"  diverging from correct 3D structure.\n"
                                    f"\n"
                                    f"  📋 THIS IS THE EXACT PATTERN OBSERVED IN FAILED TRAINING:\n"
                                    f"  - Step 20000: SDS = 6.32\n"
                                    f"  - Step 39900: SDS = 6.59 (+4.3%)\n"
                                    f"  - Result: Wrong canonical mesh (3× height, 2.4× volume)\n"
                                    f"{'=' * 70}\n"
                                )
                            elif increase_ratio < -0.1:
                                # SDS decreasing is good
                                logger.info(
                                    f"[SDS-TREND-CHECK] ✅ SDS loss decreasing properly: "
                                    f"{increase_ratio * 100:.1f}% over {window} steps "
                                    f"({prev_sds:.4f} → {current_sds:.4f})"
                                )
                            # Clean up old history to prevent memory buildup
                            if len(self._sds_history) > 100:
                                old_steps = sorted(self._sds_history.keys())[:-50]
                                for old_step in old_steps:
                                    del self._sds_history[old_step]


        else:
            # Log why GHOP was skipped
            if self.global_step % 50 == 0:
                logger.info(f"[GHOP] ⏭️  Skipped at step {self.global_step}:")
                logger.info(f"    global_step ({self.global_step}) >= phase3_start_iter ({getattr(self, 'phase3_start_iter', 0)}): {self.global_step >= getattr(self, 'phase3_start_iter', 0)}")
                logger.info(f"    global_step ({self.global_step}) >= warmup_iters ({getattr(self, 'warmup_iters', 0)}): {self.global_step >= getattr(self, 'warmup_iters', 0)}")
                logger.info(f"    global_step % sds_iters == 0: {(self.global_step % getattr(self, 'sds_iters', 1)) == 0}")

            zero_loss = torch.tensor(0.0, device=loss_output["loss"].device, requires_grad=True)
            loss_output["loss/sds"] = zero_loss  # ← Add this
            loss_output["ghop_loss"] = zero_loss

        if should_profile and self.phase3_enabled:
            self.memory_profiler.checkpoint("after_ghop")

        # ================================================================
        # GEOMETRY FREEZE CHECK (Prevent Phase 4/5 from eroding object)
        # ================================================================
        phase4_cfg = getattr(self.opt, 'phase4', {})
        freeze_enabled = phase4_cfg.get('freeze_object_geometry', False)
        freeze_at = phase4_cfg.get('freeze_at_iter', 35000)

        if freeze_enabled and self.global_step == freeze_at:
            logger.info("=" * 70)
            logger.info(f"[GEOMETRY FREEZE] Freezing object parameters at step {freeze_at}")
            logger.info("=" * 70)

            # 1) Require SDF to be valid
            if hasattr(self, "_sdf_is_valid") and not self._sdf_is_valid:
                logger.warning("[GEOMETRY FREEZE] SDF not valid; postponing freeze")
                return

            # 2) Require object Chamfer to be sufficiently low
            chamfer_ok = True
            if hasattr(self, "_last_obj_chamfer"):
                if self._last_obj_chamfer > 0.05:
                    chamfer_ok = False
                    logger.warning(
                        f"[GEOMETRY FREEZE] Chamfer(obj)={self._last_obj_chamfer:.6f} > 0.05; "
                        "postponing freeze"
                    )

            if not chamfer_ok:
                return

            # If we reach here, both SDF and Chamfer(obj) look good → safe to freeze

            # Freeze v3d_cano
            if hasattr(self.model.nodes.object.server.object_model, 'v3d_cano'):
                self.model.nodes.object.server.object_model.v3d_cano.requires_grad = False
                logger.info("  - v3d_cano: requires_grad = False")

            # Freeze sdf_grid
            if hasattr(self.model.nodes.object.server.object_model, 'sdf_grid'):
                self.model.nodes.object.server.object_model.sdf_grid.requires_grad = False
                logger.info("  - sdf_grid: requires_grad = False")

            # IMPORTANT: Remove from optimizer to prevent gradients from accumulating
            opt = self.optimizers()

            # Step 1: Collect IDs of frozen parameters
            frozen_param_ids = set()
            if hasattr(self.model.nodes.object.server.object_model, 'v3d_cano'):
                frozen_param_ids.add(id(self.model.nodes.object.server.object_model.v3d_cano))
            if hasattr(self.model.nodes.object.server.object_model, 'sdf_grid'):
                frozen_param_ids.add(id(self.model.nodes.object.server.object_model.sdf_grid))

            # Step 2: Clean optimizer state dict (remove Adam momentum buffers)
            state_keys_to_remove = []

            for key in list(opt.state.keys()):
                # In PyTorch, keys can be Tensors or already IDs
                if isinstance(key, torch.Tensor):
                    key_id = id(key)
                else:
                    key_id = key

                if key_id in frozen_param_ids:
                    state_keys_to_remove.append(key)

            for key in state_keys_to_remove:
                del opt.state[key]

            logger.info(f"  - Cleaned {len(state_keys_to_remove)} frozen parameter states from optimizer")

            # Step 3: Clean param_groups (remove frozen params from update list)
            total_removed = 0
            for param_group in opt.param_groups:
                original_count = len(param_group['params'])
                param_group['params'] = [p for p in param_group['params'] if p.requires_grad]
                total_removed += original_count - len(param_group['params'])

            logger.info(f"  - Removed {total_removed} frozen parameters from optimizer param_groups")
            logger.info("=" * 70)

        # ====================================================================
        # PHASE 4: Contact Refinement (OPTIMIZED - Skip if weight=0)
        # ====================================================================
        # OPTIMIZATION: Calculate weight BEFORE expensive operations
        WEIGHT_THRESHOLD = 1e-6
        if hasattr(self, 'phase5_enabled') and self.phase5_enabled and hasattr(self, 'phase5_scheduler'):
            # Use scheduler weight (includes ramp-up/decay)
            effective_contact_weight = loss_weights.get('contact', 0.0)
            weight_source = "scheduler"
        else:
            # Fallback: Manual warmup from config
            # Ensure attributes exist with safe defaults
            contact_start = getattr(self, 'contact_start_iter', float('inf'))
            contact_warmup = getattr(self, 'contact_warmup_iters', 1000)

            if self.global_step >= contact_start:
                contact_progress = min(
                    (self.global_step - contact_start) / max(contact_warmup, 1),
                    1.0
                )
                effective_contact_weight = self.w_contact * contact_progress
            else:
                effective_contact_weight = 0.0
            weight_source = "config"
        # ← INSERT DEBUG LOG HERE (AFTER WEIGHT CALCULATION)
        if self.global_step % 100 == 0:
            logger.info(
                f"[Phase 4 Debug] effective_contact_weight={effective_contact_weight:.6f}, "
                f"scheduler_weight={loss_weights.get('contact', 'N/A')}, "
                f"config_weight={getattr(self.opt.loss, 'w_contact', 'N/A')}, "
                f"source={weight_source}"  # ← ADD weight_source for clarity
            )

        # Check if Phase 4 should run (boundaries + weight)
        phase4_active = (
            getattr(self, 'phase4_enabled', False) and
            self.contact_refiner is not None and
            self.contact_start_iter <= self.global_step < self.contact_end_iter and
            effective_contact_weight > WEIGHT_THRESHOLD  # ← OPTIMIZATION: Skip if weight=0
        )

        # Log skip reason occasionally (avoid spam)
        if not phase4_active and self.global_step % 500 == 0:
            if not getattr(self, 'phase4_enabled', False):
                skip_reason = "Phase 4 disabled"
            elif self.contact_refiner is None:
                skip_reason = "Contact refiner not initialized"
            elif self.global_step < self.contact_start_iter:
                skip_reason = f"Before start ({self.global_step} < {self.contact_start_iter})"
            elif self.global_step >= self.contact_end_iter:
                skip_reason = f"After end ({self.global_step} >= {self.contact_end_iter})"
            elif effective_contact_weight <= WEIGHT_THRESHOLD:
                skip_reason = f"Zero weight ({effective_contact_weight:.6f} from {weight_source})"
            else:
                skip_reason = "Unknown"
            logger.debug(f"[Phase 4] SKIPPED - {skip_reason}")

        if phase4_active:
            try:
                # Activation log
                if not hasattr(self, 'phase4_activated') or not self.phase4_activated:
                    logger.info(f"[Phase 4] Contact loss ACTIVATED at step {self.global_step}")
                    self.phase4_activated = True

                logger.debug(f"[Phase 4] Running with weight {effective_contact_weight:.4f}")

                # ============================================================
                # Extract hand mesh (always needed for validation)
                # ============================================================
                hand_verts, hand_faces = self._extract_hand_mesh(batch)

                # Around line 3465 (BEFORE the skip checks), ADD:
                if self.global_step % 100 == 0:
                    logger.info(
                        f"[Phase 4 Debug] effective_contact_weight={effective_contact_weight:.6f}, "
                        f"scheduler_weight={loss_weights.get('contact', 'N/A')}, "
                        f"config_weight={getattr(self.opt.loss, 'w_contact', 'N/A')}"
                    )

                # ============================================================
                # Validate SDF before proceeding
                # ============================================================
                self._sdf_is_valid = self._validate_sdf_for_extraction()

                if not self._sdf_is_valid:
                    # Log warning and skip entire Phase 4
                    if self.global_step % 100 == 0:
                        logger.warning(
                            f"[Phase 4] Degenerate SDF at step {self.global_step}, "
                            f"skipping contact loss computation"
                        )
                else:
                    # ============================================================
                    # ONLY execute contact loss if SDF is valid
                    # ============================================================
                    obj_verts_list, obj_faces_list = self._extract_object_mesh_from_sdf(batch)

                    # Debug logging (every 50 steps)
                    if self.global_step % 50 == 0:
                        logger.info(f"[Phase 4 - Step {self.global_step}] Mesh Extraction:")
                        logger.info(f"  Hand verts: {hand_verts.shape}")
                        logger.info(f"  Hand faces: {hand_faces.shape if hand_faces is not None else 'None'}")

                        if isinstance(obj_verts_list, list):
                            for b_idx, obj_v in enumerate(obj_verts_list):
                                obj_f = obj_faces_list[b_idx] if isinstance(obj_faces_list, list) else obj_faces_list[b_idx]
                                logger.info(
                                    f"  Batch {b_idx}: Object verts={obj_v.shape[0]}, "
                                    f"faces={obj_f.shape[0] if obj_f is not None else 0}"
                                )
                                if obj_v.shape[0] == 0:
                                    logger.error(f"    ❌ Empty object mesh for batch {b_idx}!")

                    # ============================================================
                    # Adaptive Contact Zone Detection
                    # ============================================================
                    contact_zones = None

                    if (self.phase5_enabled and
                        hasattr(self, 'adaptive_contacts') and
                        hasattr(self, 'phase5_scheduler') and
                        self.phase5_scheduler is not None and
                        self.phase5_scheduler.should_update_contact_zones(self.global_step)):

                        logger.debug(f"[Phase 5] Detecting adaptive contact zones at step {self.global_step}")

                        try:
                            contact_zones = self.adaptive_contacts(
                                hand_verts=hand_verts,
                                obj_verts_list=obj_verts_list,
                                iteration=self.global_step,
                                batch_indices=list(range(hand_verts.shape[0]))
                            )

                            # Log contact statistics
                            if self.global_step % self.log_phase5_every == 0:
                                contact_stats = self.adaptive_contacts.get_contact_statistics(contact_zones)

                                logger.info('phase5/contact_mean', contact_stats['mean'], prog_bar=False)
                                logger.info('phase5/contact_min', contact_stats['min'], prog_bar=False)
                                logger.info('phase5/contact_max', contact_stats['max'], prog_bar=False)
                                logger.info('phase5/contact_std', contact_stats['std'], prog_bar=False)

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
                    # Contact Refinement Computationf
                    # ============================================================
                    batch_size = hand_verts.shape[0]
                    total_contact_loss = torch.tensor(0.0, device=self.device)
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
                        h_verts = hand_verts[b]
                        h_faces = hand_faces if hand_faces.dim() == 2 else hand_faces[b]
                        o_verts = obj_verts_list[b] if isinstance(obj_verts_list, list) else obj_verts_list[b]
                        o_faces = obj_faces_list[b] if isinstance(obj_faces_list, list) else obj_faces_list[b]

                        # Skip empty meshes
                        if o_verts.shape[0] == 0:
                            logger.warning(f"[Phase 4] Empty object mesh for batch {b}, skipping")
                            continue

                        # Get contact zones for this sample
                        zones_b = contact_zones[b] if contact_zones is not None else None

                        try:
                            # Call contact refiner
                            if hasattr(self.contact_refiner, 'forward_with_faces'):
                                contact_loss_b, contact_metrics_b = self.contact_refiner.forward_with_faces(
                                    hand_verts=h_verts,
                                    hand_faces=h_faces,
                                    obj_verts=o_verts,
                                    obj_faces=o_faces,
                                    contact_zones=zones_b,
                                )
                            else:
                                contact_loss_b, contact_metrics_b = self.contact_refiner(
                                    hand_verts=h_verts,
                                    hand_faces=h_faces,
                                    obj_verts=o_verts,
                                    obj_faces=o_faces
                                )

                            if contact_loss_b is not None:
                                total_contact_loss = total_contact_loss + contact_loss_b
                                num_valid_samples += 1

                                # Accumulate metrics
                                for key in contact_metrics_accum:
                                    if key in contact_metrics_b:
                                        contact_metrics_accum[key] += contact_metrics_b[key]

                        except Exception as e:
                            logger.error(f"[Phase 4] Contact refiner failed for batch {b}: {e}")
                            continue

                    # ============================================================
                    # Apply Weight and Add to Total Loss
                    # ============================================================
                    if num_valid_samples > 0:
                        # Average over valid samples
                        total_contact_loss = total_contact_loss / num_valid_samples
                        for key in contact_metrics_accum:
                            contact_metrics_accum[key] /= num_valid_samples

                        # Apply pre-calculated effective weight
                        weighted_contact_loss = total_contact_loss * effective_contact_weight

                        # Apply warm-up/contact multiplier
                        scaled_contact = contact_mult * weighted_contact_loss

                        # Add to total loss accumulator
                        total_loss = total_loss + scaled_contact
                        loss_output['loss_contact'] = scaled_contact
                        loss_output['contact_loss'] = scaled_contact.detach()

                        if 'loss_components' in locals():
                            loss_components['contact'] = scaled_contact.item()

                        # Logging
                        logger.info('phase4/contact_loss', scaled_contact.detach().item(), prog_bar=True)
                        logger.info('phase4/contact_weight', effective_contact_weight)
                        logger.info('phase4/penetration',
                            contact_metrics_accum['penetration'].detach().item() if isinstance(contact_metrics_accum['penetration'], torch.Tensor)
                            else float(contact_metrics_accum['penetration']))
                        logger.info('phase4/attraction',
                            contact_metrics_accum['attraction'].detach().item() if isinstance(contact_metrics_accum['attraction'], torch.Tensor)
                            else float(contact_metrics_accum['attraction']))
                        logger.info('phase4/dist_mean',
                            contact_metrics_accum['dist_mean'].detach().item() if isinstance(contact_metrics_accum['dist_mean'], torch.Tensor)
                            else float(contact_metrics_accum['dist_mean']))
                        logger.info('phase4/num_contacts', float(contact_metrics_accum['num_contacts']))
                        logger.info('phase4/num_penetrations', float(contact_metrics_accum['num_penetrations']))

                        # Console logging
                        if self.global_step % self.log_contact_every == 0:
                            contact_type = "Adaptive" if contact_zones is not None else "Fixed"
                            logger.info(
                                f"\n[Phase 4 - Step {self.global_step}] Contact Refinement ({contact_type}):\n"
                                f"  Applied weight:   {effective_contact_weight:.4f} (from {weight_source})\n"
                                f"  Valid samples:    {num_valid_samples}/{batch_size}\n"
                                f"  Penetration loss: {contact_metrics_accum['penetration']:.4f}\n"
                                f"  Attraction loss:  {contact_metrics_accum['attraction']:.4f}\n"
                                f"  Mean distance:    {contact_metrics_accum['dist_mean']:.4f}m\n"
                                f"  Num contacts:     {int(contact_metrics_accum['num_contacts'])}\n"
                                f"  Num penetrations: {int(contact_metrics_accum['num_penetrations'])}\n"
                                f"  Weighted loss:    {scaled_contact:.4f}\n"
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

        if should_profile and phase4_active:
            self.memory_profiler.checkpoint("after_contact")

        # ====================================================================
        # PHASE 2: Backward Compatibility
        # ====================================================================
        if self.phase2_enabled and self.hold_loss_module is not None and not self.phase3_enabled:
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
                        hand_pose=hand_params['pose'],
                        category=category,
                        iteration=self.global_step
                    )

                    # PHASE 2 FIX: Conditional detach if other phases active
                    phase3_active = hasattr(self, 'phase3_enabled') and self.phase3_enabled and self.ghop_enabled
                    phase4_active = (
                            hasattr(self, 'phase4_enabled') and
                            self.phase4_enabled and
                            self.contact_start_iter <= self.global_step < self.contact_end_iter
                    )
                    phase5_active = (hasattr(self, 'phase5_enabled') and self.phase5_enabled and
                                    self.global_step >= self.phase5_start_iter)

                    if phase3_active or phase4_active or phase5_active:
                        # Other phases active: detach to prevent graph conflicts
                        # loss_output["loss"] = loss_output["loss"] + loss_sds.detach()
                        loss_output["loss"] = total_loss
                        final_loss = total_loss
                        logger.debug(f"[Phase 2] Detached HOLD SDS loss (multi-phase mode)")
                    else:
                        # Only Phase 2: keep gradients
                        # loss_output["loss"] = loss_output["loss"] + loss_sds
                        loss_output["loss"] = total_loss
                        final_loss = total_loss

                    loss_output["sds_loss"] = loss_sds

                    if loss_sds.item() > 0:
                        logger.info('train/sds_loss', loss_sds.detach().item(), prog_bar=True)

            except Exception as e:
                logger.error(f"[Phase 2] SDS loss failed: {e}")

        # ====================================================================
        # PHASE 5: TEMPORAL CONSISTENCY FOR VIDEO SEQUENCES
        # ====================================================================
        # ✅ FIX: Ensure Phase 3 has ended before Phase 5 starts
        should_compute_phase5 = (
            self.phase5_enabled and
            self.global_step >= self.phase5_start_iter and
            self.global_step < getattr(self, 'phase5_end_iter', float('inf'))
        )
        if should_compute_phase5:
            try:
                # Log phase transition
                if self.global_step == self.phase5_start_iter:
                    logger.info(f"[Phase 5] Temporal consistency ACTIVATED at step {self.global_step}")
                    logger.info(f"[Phase 3] SDS diffusion fully DISABLED")
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

                    # ================================================================
                    # Step 4 Get sequence ID for history tracking (IMPROVED)
                    # ================================================================
                    # Try multiple potential sequence identifier fields
                    sequence_id = None

                    # Priority order: video_id > sequence_id > scene_id > idx
                    for key in ['video_id', 'sequence_id', 'scene_id', 'idx']:
                        if key in batch:
                            seq_val = batch[key]

                            # Handle tensor
                            if isinstance(seq_val, torch.Tensor):
                                if seq_val.numel() == 1:
                                    sequence_id = f"{key}_{seq_val.item()}"
                                else:
                                    sequence_id = f"{key}_{seq_val[0].item()}"
                                break

                            # Handle list/tuple
                            elif isinstance(seq_val, (list, tuple)):
                                if len(seq_val) > 0:
                                    sequence_id = f"{key}_{seq_val[0]}"
                                break

                            # Handle string/int
                            elif isinstance(seq_val, (str, int, float)):
                                sequence_id = f"{key}_{seq_val}"
                                break

                    # Fallback to global step-based ID if nothing found
                    if sequence_id is None:
                        # Use global_step // batch_size as a proxy for video frame number
                        batch_size = batch['hA'].shape[0] if 'hA' in batch else 1
                        approx_frame_id = self.global_step // max(batch_size, 1)
                        sequence_id = f"default_frame_{approx_frame_id}"

                        logger.debug(
                            f"[Phase 5] No sequence ID found in batch. "
                            f"Using fallback: {sequence_id}"
                        )

                    logger.debug(f"[Phase 5] Temporal tracking with sequence_id: {sequence_id}")

                    # =============================================================
                    # Step 5: Compute temporal consistency losses
                    # =============================================================
                    temporal_loss, temporal_metrics = self.temporal_module(
                        sample=batch,  # Contains hA, hA_n, c2w, c2w_n from hoi.py
                        predicted_hand_pose=predicted_pose,
                        sequence_id=sequence_id,
                    )
                    # Add BEFORE the if statement
                    logger.warning(f"[DEBUG Phase 5] temporal_loss: {temporal_loss}")
                    logger.warning(f"[DEBUG Phase 5] temporal_loss is None: {temporal_loss is None}")
                    if temporal_loss is not None:
                        logger.warning(f"[DEBUG Phase 5] temporal_loss.item(): {temporal_loss.item()}")
                    else:
                        logger.warning(f"[DEBUG Phase 5] temporal_loss IS NONE - no loss to log")
                    # Add logging
                    if temporal_loss is not None:
                        logger.info('phase5/temporal_loss_raw', temporal_loss.detach().item(), prog_bar=True)
                        if self.global_step % self.log_phase5_every == 0:
                            logger.info(
                                f"[Phase 5 - Step {self.global_step}] Temporal loss: {temporal_loss.item():.6f}")
                            if temporal_metrics:
                                logger.info(f"[Phase 5] Metrics: {temporal_metrics}")

                    # =============================================================
                    # Step 6: Apply Phase 5 dynamic weighting
                    # =============================================================
                    # Scheduler already provides complete weight (no config multiplication needed)
                    if hasattr(self, 'phase5_scheduler') and self.phase5_scheduler is not None:
                        # Use scheduler weight only
                        weighted_temporal = temporal_loss * loss_weights['temporal']
                        logger.debug(f"[Phase 5] Using scheduler temporal weight: {loss_weights['temporal']:.4f}")
                    else:
                        # Fallback: Use config weight
                        weighted_temporal = temporal_loss * self.w_temporal
                        logger.debug(f"[Phase 5] Using config temporal weight: {self.w_temporal:.4f}")

                    # NEW: apply warm-up temporal multiplier
                    scaled_temporal = temporal_mult * weighted_temporal

                    # =============================================================
                    # Step 7: Add to total loss (OPTION A: with gradients)
                    # =============================================================
                    # OPTION A: Remove detachment for direct gradient optimization
                    # Temporal loss needs to directly optimize hand pose smoothness
                    # Line ~2553 - Use total_loss accumulator
                    total_loss = total_loss + scaled_temporal                  # keep gradients
                    loss_output["loss_temporal"] = scaled_temporal             # debug: grad-tracking
                    loss_output['temporal_loss'] = scaled_temporal.detach()    # logging-only
                    if 'loss_components' in locals():
                        loss_components['temporal'] = scaled_temporal.item()

                    logger.debug(f"[Phase 5] Added temporal loss WITH GRADIENTS: {scaled_temporal.item():.6f}")

                    # Add gradient monitoring for validation
                    if self.global_step % 100 == 0:
                        temporal_grad_norm = weighted_temporal.grad.norm() if weighted_temporal.grad is not None else 0.0
                        logger.info(f"[Phase 5 Grad Check] temporal_loss grad_norm: {temporal_grad_norm:.6f}")

                    # =============================================================
                    # Step 8: Log temporal metrics
                    # =============================================================
                    logger.info('phase5/temporal_loss', scaled_temporal.item(), prog_bar=True)
                    v = temporal_metrics.get('velocity', 0.0)
                    logger.info('phase5/velocity_loss', v.detach().item() if isinstance(v, torch.Tensor) else float(v), prog_bar=False)
                    a = temporal_metrics.get('acceleration', 0.0)
                    logger.info('phase5/acceleration_loss', a.detach().item() if isinstance(a, torch.Tensor) else float(a), prog_bar=False)
                    c = temporal_metrics.get('camera_motion', 0.0)
                    logger.info('phase5/camera_motion_loss', c.detach().item() if isinstance(c, torch.Tensor) else float(c), prog_bar=False)
                    w = temporal_metrics.get('adaptive_weight', 1.0)
                    logger.info('phase5/temporal_adaptive_weight', w.detach().item() if isinstance(w, torch.Tensor) else float(w), prog_bar=False)

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
        if should_profile and self.phase5_enabled:
            self.memory_profiler.checkpoint("after_temporal")

        # ====================================================================
        # ✅ CRITICAL FIX: Verify Loss is Not Zero Before Backward
        # ====================================================================
        # final_loss = loss_output.get("loss",
        #                              torch.tensor(0.0, device=next(self.parameters()).device, requires_grad=True))

        # ====================================================================
        # ✅ OPTION A VALIDATION: Monitor all gradient components
        # ====================================================================
        if self.global_step % 100 == 0 and self.global_step > 0:
            grad_components = {
                'rgb': loss_output.get('loss/rgb', None),
                'ghop': loss_output.get('ghop_loss', None),
                'contact': loss_output.get('contact_loss', None),
                'temporal': loss_output.get('temporal_loss', None),
            }

            logger.info(f"\n[OPTION A GRAD CHECK - Step {self.global_step}]")
            for name, loss_val in grad_components.items():
                if loss_val is not None and isinstance(loss_val, torch.Tensor):
                    has_grad = loss_val.requires_grad
                    grad_norm = loss_val.grad.norm().item() if loss_val.grad is not None else 0.0
                    logger.info(f"  {name:10s}: requires_grad={has_grad}, grad_norm={grad_norm:.6f}")

            # Check for gradient imbalance (contact >> temporal)
            contact_grad = grad_components.get('contact')
            temporal_grad = grad_components.get('temporal')

            if contact_grad is not None and temporal_grad is not None:
                if contact_grad.grad is not None and temporal_grad.grad is not None:
                    ratio = contact_grad.grad.norm() / max(temporal_grad.grad.norm(), 1e-8)
                    if ratio > 100:
                        logger.warning(
                            f"  ⚠️  GRADIENT IMBALANCE: contact/temporal = {ratio:.1f}x. "
                            f"Consider reducing w_contact or increasing w_temporal."
                        )

        # Safety check: Ensure loss_output['loss'] exists and is valid
        if "loss" not in loss_output:
            logger.error(
                f"[Step {self.global_step}] CRITICAL: loss_output has no 'loss' key! Keys: {list(loss_output.keys())}")
            # Instead of unconditionally zeroing:
            if "loss" not in loss_output or not torch.isfinite(loss_output["loss"]):
                logger.warning(
                    f"[Step {self.global_step}] loss_output['loss'] missing or non-finite. "
                    "Setting loss to zero for this batch."
                )
                loss_output["loss"] = torch.tensor(0.0, device=next(self.parameters()).device, requires_grad=True)
            # In all cases, final_loss uses loss_output["loss"]
            # final_loss = loss_output["loss"]

        # Ensure loss has requires_grad

        # Get phase activity status for diagnostics
        phase3_active = (
                hasattr(self, 'phase3_enabled') and self.phase3_enabled and
                self.global_step >= getattr(self, 'phase3_start_iter', 0) and
                self.global_step < getattr(self, 'phase3_end_iter', 99999)
        )
        phase4_active = (
                hasattr(self, 'phase4_enabled') and
                self.phase4_enabled and
                self.contact_start_iter <= self.global_step < self.contact_end_iter
        )
        phase5_active = (hasattr(self, 'phase5_enabled') and self.phase5_enabled and
                         self.global_step >= self.phase5_start_iter)

        # Convert to float for logging
        def to_float(x):
            if isinstance(x, torch.Tensor):
                return x.item() if x.numel() > 0 else 0.0
            return float(x) if x is not None else 0.0

        # ====================================================================
        # CRITICAL: Check for Zero Loss at Phase Transitions
        # ====================================================================
        loss_value = total_loss.item() if total_loss.numel() > 0 else 0.0

        if abs(loss_value) < 1e-8:  # Effectively zero
            logger.warning(
                f"\n[Step {self.global_step}] ⚠️  ZERO LOSS DETECTED!\n"
                f"  Loss value:       {loss_value:.10f}\n"
                f"  Phase 3 active:   {phase3_active}\n"
                f"  Phase 4 active:   {phase4_active}\n"
                f"  Phase 5 active:   {phase5_active}\n"
                f"  Loss components:\n"
                f"    - Base RGB:     {to_float(loss_output.get('loss', 0.0)):.6f}\n"
                f"    - GHOP:         {to_float(loss_output.get('ghop_loss', 0.0)):.6f}\n"
                f"    - Contact:      {to_float(loss_output.get('contact_loss', 0.0)):.6f}\n"
                f"  Skipping backward for this iteration (preventing crash)\n"
            )

            # Clear gradients and skip this iteration
            opt = self.optimizers()
            opt.zero_grad(set_to_none=True)

            # Return dummy loss for PyTorch Lightning
            dummy_loss = torch.tensor(0.0, device=next(self.parameters()).device, requires_grad=False)
            logger.error(
                f"[DEBUG] About to hit dummy_loss return at step {self.global_step}, loss={total_loss.item()}")
            loss_to_return = dummy_loss

        # ================================================================
        # GHOP FIX: Manual optimization to prevent double backward
        # ================================================================
        # At the END of all phase computations, before backward:
        loss_output["loss"] = total_loss  # Write accumulator to dict
        final_loss = total_loss  # Use accumulator for backward

        # Verify this is the FINAL value used
        logger.debug(f"[TRAIN STEP] step={self.global_step} final_loss={final_loss.item():.6f}")
        if torch.isnan(final_loss) or torch.isinf(final_loss):
            logger.warning(
                f"[Step {self.global_step}] final_loss is NaN/Inf. "
                "Skipping optimization for this batch."
            )
            # Preserve the idea of “no learning on bad step”:
            # - Do not call backward/step.
            # - Return a zero loss tensor (so Lightning's machinery still works).
            safe_zero = torch.tensor(0.0, device=final_loss.device, requires_grad=False)
            logger.error(f"[DEBUG] About to hit GHOP safe_zero return at step {self.global_step}")
            loss_to_return = safe_zero

        # ================================================================
        # UNIFIED LOSS LOGGING - Place at END of training_step (before return)
        # ================================================================
        LOG_INTERVAL = 100  # Log comprehensive breakdown every 100 steps
        CONSOLE_INTERVAL = 10  # Log key metrics to console every 10 steps

        # -------------------------------------------------------------------
        # 1. TENSORBOARD LOGGING (every step for all metrics)
        # -------------------------------------------------------------------
        logger.info('train/loss_total', total_loss.item(), prog_bar=True)

        # Base losses
        if 'loss/rgb' in loss_output:
            logger.info('train/loss_rgb', loss_output['loss/rgb'], prog_bar=False)
        if 'loss/mask' in loss_output:
            logger.info('train/loss_mask', loss_output['loss/mask'], prog_bar=False)
        if 'loss/eikonal' in loss_output:
            logger.info('train/loss_eikonal', loss_output['loss/eikonal'], prog_bar=False)

        # Geometric losses (Step 2)
        if 'geometric_loss' in loss_output:
            logger.info('train/geometric_loss', loss_output['geometric_loss'], prog_bar=False)
        if 'loss/normal_consistency' in loss_output:
            logger.info('train/normal_consistency', loss_output['loss/normal_consistency'], prog_bar=False)
        if 'loss/depth_smoothness' in loss_output:
            logger.info('train/depth_smoothness', loss_output['loss/depth_smoothness'], prog_bar=False)
        if 'loss/chamfer' in loss_output:  # ← ADD THIS
            logger.info('train/chamfer', loss_output['loss/chamfer'], prog_bar=False)
        if 'loss/obj_chamfer' in loss_output:  # ← ADD THIS (Object Chamfer)
            logger.info('train/obj_chamfer', loss_output['loss/obj_chamfer'], prog_bar=False)

        # Object smoothness (if active)
        if 'object_smoothness' in loss_output:
            logger.info('train/object_smoothness', loss_output['object_smoothness'], prog_bar=False)

        # Phase 3: SDS losses
        if 'loss/sds_weighted' in loss_output:
            logger.info('train/sds_weighted', loss_output['loss/sds_weighted'], prog_bar=True)
            logger.info('train/sds_raw', loss_output.get('loss/sds_raw', 0.0), prog_bar=False)
            logger.info('train/sds_weight', loss_output.get('sds_weight', 0.0), prog_bar=False)

        # Phase 4: Contact loss
        if 'contact_loss' in loss_output:
            logger.info('train/contact_loss', loss_output['contact_loss'], prog_bar=False)

        # Phase 5: Temporal loss
        if 'temporal_loss' in loss_output:
            logger.info('train/temporal_loss', loss_output['temporal_loss'], prog_bar=False)

        # -------------------------------------------------------------------
        # 2. CONSOLE LOGGING - Key Metrics (every 10 steps)
        # -------------------------------------------------------------------
        if self.global_step % CONSOLE_INTERVAL == 0:
            # Extract values
            rgb_val = loss_output.get('loss/rgb', 0.0)
            rgb_val = rgb_val.item() if torch.is_tensor(rgb_val) else rgb_val

            total_val = total_loss.item()

            logger.info(
                f"[Step {self.global_step:6d}] "
                f"Total: {total_val:.6f} | "
                f"RGB: {rgb_val:.6f}"
            )

        # -------------------------------------------------------------------
        # 3. COMPREHENSIVE BREAKDOWN (every 100 steps)
        # -------------------------------------------------------------------
        if self.global_step % LOG_INTERVAL == 0:
            # Helper function
            def to_float(x):
                if isinstance(x, torch.Tensor):
                    return x.item() if x.numel() > 0 else 0.0
                return float(x) if x is not None else 0.0

            # Extract all components
            total_f = to_float(total_loss)
            rgb_f = to_float(loss_output.get('loss/rgb', 0.0))
            mask_f = to_float(loss_output.get('loss/sem', 0.0)) # Semantic loss
            mask_binary_f = to_float(loss_output.get("loss/mask_binary", 0.0))  # New binary mask loss
            eikonal_f = to_float(loss_output.get('loss/eikonal', 0.0))
            geometric_f = to_float(loss_output.get('geometric_loss', 0.0))
            normal_f = to_float(loss_output.get('loss/normal_consistency', 0.0))
            depth_smooth_f = to_float(loss_output.get('loss/depth_smoothness', 0.0))
            chamfer_f = to_float(loss_output.get('loss/chamfer', 0.0))  # ← ADD THIS
            obj_chamfer_f = to_float(loss_output.get('loss/obj_chamfer', 0.0))
            obj_smooth_f = to_float(loss_output.get('object_smoothness', 0.0))
            # Cache latest object Chamfer for geometry-freeze logic
            self._last_obj_chamfer = obj_chamfer_f

            # Reactive curriculum: track how long Chamfer(obj) stays high
            if not hasattr(self, "high_chamfer_steps"):
                self.high_chamfer_steps = 0

            if obj_chamfer_f > 0.05:
                self.high_chamfer_steps += 1
            else:
                self.high_chamfer_steps = 0

            # If Chamfer(obj) has been >0.05 for a long window, boost SDS
            if self.high_chamfer_steps >= 2000:
                old_sds_dyn = self.dynamic_sds_mult
                self.dynamic_sds_mult = min(self.dynamic_sds_mult * 1.5, 5.0)
                logger.warning(
                    f"[CURRICULUM] Chamfer(obj) stuck >0.05 for "
                    f"{self.high_chamfer_steps} checks – "
                    f"dynamic_sds_mult {old_sds_dyn:.2f} → {self.dynamic_sds_mult:.2f}"
                )
                self.high_chamfer_steps = 0
            sds_raw_f = to_float(loss_output.get('loss/sds_raw', 0.0))
            sds_weighted_f = to_float(loss_output.get('loss/sds_weighted', 0.0))
            sds_weight_f = to_float(loss_output.get('sds_weight', 0.0))
            contact_f = to_float(loss_output.get('contact_loss', 0.0))
            temporal_f = to_float(loss_output.get('temporal_loss', 0.0))
            sdf_zero_f = to_float(loss_output.get('loss/sdf_zero_cross', 0.0))
            sdf_grad_f = to_float(loss_output.get('loss/sdf_gradient', 0.0))
            sdf_range_f = to_float(loss_output.get('loss/sdf_range', 0.0))
            sdf_on_verts_f = to_float(loss_output.get('loss/sdf_on_verts', 0.0))
            # New: separate teacher vs student zero-cross counts
            grid_zero_crossings_f = to_float(loss_output.get('loss/sdf_zero_crossings', 0.0))
            implicit_zero_crossings_f = to_float(
                loss_output.get('loss/implicit_sdf_zero_crossings', 0.0)
            )
            # Store for reactive monitoring in next step
            self._last_implicit_zero_crossings = implicit_zero_crossings_f

            # Calculate percentages (avoid division by zero)
            if total_f > 1e-8:
                rgb_pct = (rgb_f / total_f) * 100
                sds_pct = (sds_weighted_f / total_f) * 100
                contact_pct = (contact_f / total_f) * 100
                temporal_pct = (temporal_f / total_f) * 100
                geometric_pct = (geometric_f / total_f) * 100
            else:
                rgb_pct = sds_pct = contact_pct = temporal_pct = geometric_pct = 0.0

            # Phase status
            phase3_active = (
                    hasattr(self, 'phase3_enabled') and self.phase3_enabled and
                    self.global_step >= getattr(self, 'phase3_start_iter', 0) and
                    self.global_step < getattr(self, 'phase3_end_iter', 99999)
            )
            phase4_active = (
                    hasattr(self, 'phase4_enabled') and self.phase4_enabled and
                    self.contact_start_iter <= self.global_step < self.contact_end_iter
            )
            phase5_active = (
                    hasattr(self, 'phase5_enabled') and self.phase5_enabled and
                    self.global_step >= self.phase5_start_iter
            )

            # Build phase indicator
            active_phases = []
            if phase3_active:
                active_phases.append(f"P3[SDS]")
            if phase4_active:
                active_phases.append(f"P4[Contact]")
            if phase5_active:
                active_phases.append(f"P5[Temporal]")
            phase_str = " + ".join(active_phases) if active_phases else "P1[Base]"

            # Print comprehensive breakdown
            logger.info("=" * 80)
            logger.info(f"LOSS BREAKDOWN - Step {self.global_step} - {phase_str}")
            logger.info("=" * 80)
            logger.info(f"  TOTAL LOSS:          {total_f:.6f}")
            logger.info("-" * 80)
            logger.info(f"  Base Components:")
            logger.info(f"    RGB:               {rgb_f:.6f}  ({rgb_pct:5.1f}%)")
            logger.info(f"    Mask (semantic):   {mask_f:.6f}")
            logger.info(f"    Mask (binary):     {mask_binary_f:.6f}")
            logger.info(f"    Eikonal:           {eikonal_f:.6f}")
            logger.info(f"    SDF Zero-cross (loss, grid):  {sdf_zero_f:.6f}")
            logger.info(f"    SDF Gradient (grid):          {sdf_grad_f:.6f}")
            logger.info(f"    SDF Range   (grid):           {sdf_range_f:.6f}")
            logger.info(f"    SDF on_verts (grid):          {sdf_on_verts_f:.6f}")
            logger.info(f"    Implicit Zero-crossings:      {implicit_zero_crossings_f:.6f}")
            logger.info(f"    Grid Zero-crossings:          {grid_zero_crossings_f:.6f}")
            logger.info("-" * 80)
            logger.info(f"  Step 2 - Geometric Regularization:")
            logger.info(f"    Geometric Total:   {geometric_f:.6f}  ({geometric_pct:5.1f}%)")
            logger.info(f"      Normal Consist:  {normal_f:.6f}")
            logger.info(f"      Depth Smooth:    {depth_smooth_f:.6f}")
            logger.info(f"      Chamfer (hand):  {chamfer_f:.6f}")  # ← ADD THIS
            logger.info(f"      Chamfer (obj):   {obj_chamfer_f:.6f}")
            logger.info(f"    Object Smooth:     {obj_smooth_f:.6f}")

            if phase3_active:
                logger.info("-" * 80)
                logger.info(f"  Phase 3 - SDS Diffusion:")
                logger.info(f"    SDS (raw):         {sds_raw_f:.6f}")
                logger.info(f"    SDS weight:        {sds_weight_f:.4f}")
                logger.info(f"    SDS (weighted):    {sds_weighted_f:.6f}  ({sds_pct:5.1f}%)")

            if phase4_active:
                logger.info("-" * 80)
                logger.info(f"  Phase 4 - Contact:")
                logger.info(f"    Contact Loss:      {contact_f:.6f}  ({contact_pct:5.1f}%)")

            if phase5_active:
                logger.info("-" * 80)
                logger.info(f"  Phase 5 - Temporal:")
                logger.info(f"    Temporal Loss:     {temporal_f:.6f}  ({temporal_pct:5.1f}%)")

            logger.info("=" * 80)

            # Gradient balance check (only if SDS active)
            if sds_weighted_f > 0 and rgb_f > 0:
                gradient_ratio = sds_weighted_f / rgb_f
                logger.info(f"  Gradient Balance:    SDS/RGB = {gradient_ratio:.2f}x")
                if gradient_ratio > 5.0:
                    logger.warning("    ⚠️  SDS dominates - may prevent RGB convergence")
                elif gradient_ratio < 0.2:
                    logger.warning("    ⚠️  RGB dominates - SDS guidance may be weak")

                # Track how long RGB dominates (low SDS/RGB)
                if not hasattr(self, "rgb_dom_steps"):
                    self.rgb_dom_steps = 0

                if gradient_ratio < 0.3:
                    self.rgb_dom_steps += 1
                else:
                    self.rgb_dom_steps = 0

                if self.rgb_dom_steps >= 1000:
                    old_geom_dyn = self.dynamic_geom_mult
                    self.dynamic_geom_mult = max(self.dynamic_geom_mult * 0.9, 0.5)
                    logger.warning(
                        f"[CURRICULUM] RGB-dominant regime for {self.rgb_dom_steps} checks "
                        f"(ratio={gradient_ratio:.2f}) – "
                        f"dynamic_geom_mult {old_geom_dyn:.2f} → {self.dynamic_geom_mult:.2f}"
                    )
                    self.rgb_dom_steps = 0

                # Existing closed-loop SDS boosting: SDS collapse + RGB dominance
                if (
                    self.global_step > 5000
                    and sds_raw_f < 0.01
                    and gradient_ratio < 0.5
                ):
                    old_sds_dyn = self.dynamic_sds_mult
                    old_geom_dyn = self.dynamic_geom_mult

                    # Boost SDS by ~2–3×, cap at 5×
                    self.dynamic_sds_mult = min(self.dynamic_sds_mult * 2.0, 5.0)
                    # Soften geometry by 0.5×, floor at 0.1
                    self.dynamic_geom_mult = max(self.dynamic_geom_mult * 0.5, 0.5)

                    logger.warning(
                        f"[SDS-BOOST] Step {self.global_step}: "
                        f"sds_raw={sds_raw_f:.6f}, ratio={gradient_ratio:.2f} → "
                        f"dynamic_sds_mult {old_sds_dyn:.2f} → {self.dynamic_sds_mult:.2f}, "
                        f"dynamic_geom_mult {old_geom_dyn:.2f} → {self.dynamic_geom_mult:.2f}"
                    )

            logger.info("=" * 80)

        # -------------------------------------------------------------------
        # 4. ZERO LOSS WARNING (critical safety check)
        # -------------------------------------------------------------------
        loss_value = total_loss.item() if total_loss.numel() > 0 else 0.0

        if abs(loss_value) < 1e-8:  # Effectively zero
            logger.error("!" * 80)
            logger.error(f"CRITICAL: ZERO LOSS DETECTED at step {self.global_step}")
            logger.error("!" * 80)
            logger.error(f"  Phase 3 active: {phase3_active}")
            logger.error(f"  Phase 4 active: {phase4_active}")
            logger.error(f"  Phase 5 active: {phase5_active}")
            logger.error(f"  Loss components:")
            logger.error(f"    - Base RGB: {to_float(loss_output.get('loss/rgb', 0.0)):.6f}")
            logger.error(f"    - GHOP:     {to_float(loss_output.get('ghop_loss', 0.0)):.6f}")
            logger.error(f"    - Contact:  {to_float(loss_output.get('contact_loss', 0.0)):.6f}")
            logger.error(f"    - Temporal: {to_float(loss_output.get('temporal_loss', 0.0)):.6f}")
            logger.error("!" * 80)

        # Get optimizer
        opt = self.optimizers()

        # ✅ Zero gradients with aggressive cleanup
        opt.zero_grad(set_to_none=True)

        if should_profile:
            self.memory_profiler.checkpoint("after_zero_grad")

        # Manual backward - SINGLE call only
        # ================================================================
        # ✅ FIX: Add phase transition detection
        # ================================================================
        is_phase_transition = (
                hasattr(self, 'phase3_end_iter') and
                hasattr(self, 'phase5_start_iter') and
                (self.phase3_end_iter - 1) <= self.global_step <= (self.phase5_start_iter + 1)
        )

        if is_phase_transition:
            logger.warning(
                f"[Step {self.global_step}] Phase transition detected. "
                f"Using retain_graph=True"
            )

        # Single backward call with appropriate retain_graph setting
        # self.manual_backward(final_loss, retain_graph=is_phase_transition or True)
        # self.manual_backward(final_loss, retain_graph=is_phase_transition)
        self.manual_backward(final_loss, retain_graph=True)
        # self.manual_backward(final_loss, retain_graph=False) # 02/15 test: temporary disable
        # distable retaining graph failed:pytorch_lightning.utilities.exceptions.MisconfigurationException: In manual optimization, `training_step` must either return a Tensor, a dict with extras to pass to `training_epoch_end` or have no return.

        # ================================================================
        # ✅ VERIFICATION: Check if gradients actually reached MANO params
        # ================================================================
        if self.global_step % 100 == 0:
            transl_param = self.model.nodes['right'].params.transl.weight
            try:
                pose_param = self.model.nodes['right'].params.pose.weight
                if pose_param.grad is not None:
                    grad_mean = pose_param.grad.abs().mean().item()
                    grad_max = pose_param.grad.abs().max().item()
                    logger.warning(f"[GRAD FLOW - Step {self.global_step}]")
                    logger.warning(f"  pose.grad EXISTS: mean={grad_mean:.10f}, max={grad_max:.10f}")
                else:
                    logger.warning(f"[GRAD FLOW - Step {self.global_step}]")
                    logger.warning(f"  ❌ pose.grad is None - GRADIENTS NOT FLOWING!")
                # ADD: Check object vertex gradients
                # obj_model = self.model.servers['object'].object_model
                obj_model = self.model.nodes["object"].server.object_model
                if hasattr(obj_model, 'v3d_cano'):
                    obj_vertices = obj_model.v3d_cano

                    logger.warning(f"[OBJECT GRAD CHECK - Step {self.global_step}]")
                    logger.warning(f"  v3d_cano.requires_grad = {obj_vertices.requires_grad}")
                    logger.warning(f"  v3d_cano.is_leaf = {obj_vertices.is_leaf}")
                    logger.warning(f"  v3d_cano.shape = {obj_vertices.shape}")

                    if obj_vertices.grad is not None:
                        grad_mean = obj_vertices.grad.abs().mean().item()
                        grad_max = obj_vertices.grad.abs().max().item()
                        logger.warning(f"  ✅ v3d_cano.grad EXISTS: mean={grad_mean:.10f}, max={grad_max:.10f}")
                    else:
                        logger.warning(f"  ❌ v3d_cano.grad is None - GRADIENTS NOT FLOWING!")
                # ADD: Check SDF gradients as well
                if hasattr(obj_model, 'sdf_grid'):
                    sdf_param = obj_model.sdf_grid
                    logger.warning(f"  sdf_grid.requires_grad = {sdf_param.requires_grad}")
                    logger.warning(f"  sdf_grid.is_leaf = {sdf_param.is_leaf}")
                    if sdf_param.grad is not None:
                        sdf_grad_mean = sdf_param.grad.abs().mean().item()
                        sdf_grad_max = sdf_param.grad.abs().max().item()
                        logger.warning(f"  ✅ sdf_grid.grad EXISTS: mean={sdf_grad_mean:.10f}, max={sdf_grad_max:.10f}")
                    else:
                        logger.warning(f"  ❌ sdf_grid.grad is None - SDF NOT RECEIVING GRADIENTS!")
            except Exception as e:
                logger.warning(f"[GRAD FLOW - Step {self.global_step}] Error checking gradients: {e}")

            if transl_param.grad is not None:
                grad_mean = transl_param.grad.abs().mean().item()
                logger.warning(f"  ✅ transl.grad mean={grad_mean:.10f}")
            else:
                logger.warning(f"  ❌ transl.grad is None!")
        # ================================================================

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        if should_profile:
            self.memory_profiler.checkpoint("after_backward")

        # Gradient clipping
        if hasattr(self.args, 'gradient_clip_val') and self.args.gradient_clip_val:
            torch.nn.utils.clip_grad_norm_(
                self.parameters(),
                self.args.gradient_clip_val
            )

        # ================================================================
        # PHASE 5 MEMORY MANAGEMENT: Per-step clearing during temporal phase
        # Addresses OOM from temporal history and adaptive contact accumulation
        # ================================================================
        if hasattr(self, 'phase5_enabled') and self.phase5_enabled:
            should_compute_phase5 = (
                    self.global_step >= getattr(self, 'phase5_start_iter', float('inf')) and
                    self.global_step < getattr(self, 'phase5_end_iter', float('inf'))
            )
            if should_compute_phase5:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                import gc
                gc.collect()

                if self.global_step % 10 == 0:
                    allocated = torch.cuda.memory_allocated() / 1024 ** 2
                    reserved = torch.cuda.memory_reserved() / 1024 ** 2
                    logger.debug(
                        f"[Phase 5 Memory] Step {self.global_step}: "
                        f"Allocated={allocated:.1f}MB, Reserved={reserved:.1f}MB"
                    )

        # ================================================================
        # ✅ OPTIMIZED: Single cache clear after backward
        # REMOVED: Multiple redundant synchronize() and empty_cache() calls
        # ================================================================
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Optimizer step
        opt.step()

        if should_profile:
            self.memory_profiler.checkpoint("after_optimizer")
        # ================================================================
        # ✅ FRAGMENTATION FIX: Periodic defragmentation every 10 steps
        # Forces CUDA allocator to return fragmented memory to OS
        # Addresses confirmed 23.3 GB GPU growth with stable PyTorch memory
        # ================================================================
        if batch_idx % 10 == 0 and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            # Force Python garbage collection
            import gc
            gc.collect()

            # Second cache clear after gc (catches freed tensors)
            torch.cuda.empty_cache()

            # Optional: Log defragmentation action
            if should_profile or (batch_idx % 100 == 0):
                allocated = torch.cuda.memory_allocated() / 1024 ** 2
                reserved = torch.cuda.memory_reserved() / 1024 ** 2
                logger.debug(
                    f"[Defrag Step {batch_idx}] "
                    f"Allocated: {allocated:.1f} MB, Reserved: {reserved:.1f} MB")

        # Learning rate scheduler
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()
        # ================================================================
        # ✅ FIX: Define is_phase5_active before diagnostic check
        # ================================================================
        is_phase5_active = (
            hasattr(self, 'temporal_module') and
            self.temporal_module is not None and
            hasattr(self, 'hparams') and
            self.global_step >= self.hparams.get('phase5_start_iter', 1100)
        )
        # ================================================================
        # ✅ OPTIMIZED: Final cleanup - reduced frequency (every 50 steps)
        # REMOVED: Every-step gc.collect() and multiple cache clears
        # ================================================================
        opt.zero_grad(set_to_none=True)

        if batch_idx % 10 == 0 and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()

        # ================================================================
        # ✅ DIAGNOSTIC: Add memory_summary for fragmentation detection
        # ================================================================
        if batch_idx % 10 == 0 and torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated() / 1024 ** 2
                reserved = torch.cuda.memory_reserved() / 1024 ** 2
                gap = reserved - allocated

                # Log every 10 steps to capture growth pattern
                logger.info(
                    f"\n[Memory Summary Step {batch_idx}]\n"
                    f"  Allocated: {allocated:.1f} MB\n"
                    f"  Reserved:  {reserved:.1f} MB\n"
                    f"  Gap:       {gap:.1f} MB ({gap / max(reserved, 1) * 100:.1f}%)\n"
                )

                # Save full summary if gap > 500MB
                if gap > 500:
                    memory_summary = torch.cuda.memory_summary(device=0, abbreviated=False)  # Changed to False for detail
                    summary_file = f"../ghop_production_chunked_results/memory_summary_step_{batch_idx}.txt"
                    with open(summary_file, 'w') as f:
                        f.write(memory_summary)
                    logger.warning(f"Memory gap {gap:.1f} MB exceeds threshold. Full summary: {summary_file}")

            except Exception as e:
                logger.debug(f"Memory summary failed: {e}")

        if should_profile:
            self.memory_profiler.checkpoint("after_final_clear")
            self.memory_profiler.report()

        # ================================================================
        # DIAGNOSTIC (MUST BE BEFORE RETURN)
        # ================================================================
        if (self._diagnostic_enabled and is_phase5_active and
                self.global_step % 100 == 0):

            if hasattr(self, 'temporal_module') and self.temporal_module is not None:
                mem_info = self.temporal_diagnostic.log_temporal_state(
                    self.temporal_module,
                    epoch=self.current_epoch,
                    step=self.global_step,
                    tag="[PERIODIC CHECK]"
                )

        # Detailed memory report every 200 steps
        if self.global_step % 200 == 0 and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 ** 2
            reserved = torch.cuda.memory_reserved() / 1024 ** 2
            peak = torch.cuda.max_memory_allocated() / 1024 ** 2

            msg = (
                f"\n{'=' * 70}\n"
                f"[Training Step {self.global_step}] Memory Status\n"
                f"  Allocated: {allocated:.2f} MB\n"
                f"  Reserved:  {reserved:.2f} MB\n"
                f"  Peak:      {peak:.2f} MB\n"
                f"{'=' * 70}\n"
            )
            logger.info(msg)

        # ----------------------------------------------------------------------
        # Loss diagnostics & logging  (THIS IS YOUR CANONICAL BLOCK)
        # ----------------------------------------------------------------------
        if (self.global_step % 50 == 0) or (batch_idx % 200 == 0):
            rgb_val = loss_output.get("loss/rgb", None)
            sds_val = loss_output.get("loss/sds", None)
            contact_val = loss_output.get("loss/contact", None)
            temporal_val = loss_output.get("loss/temporal", None)

            def fmt(x):
                return f"{x.item():.6f}" if x is not None and torch.is_tensor(x) else "None"

            logger.debug(
                f"[Loss Debug] step={self.global_step} "
                f"RGB={fmt(rgb_val)} SDS={fmt(sds_val)} "
                f"Contact={fmt(contact_val)} Temporal={fmt(temporal_val)} "
                f"TOTAL={final_loss.item():.6f}"
            )

        # Log to Lightning metrics
        logger.info('train/loss', final_loss.detach().item(), prog_bar=True)
        for key, value in loss_output.items():
            if isinstance(value, torch.Tensor) and key != 'loss':
                logger.info(f'train/{key}', value.detach().item(), prog_bar=False)

        # Explicit per-step scalar loss logging
        logger.debug(f"[TRAIN STEP] step={self.global_step} loss={final_loss.item():.6f}")

        # ================================================================
        # VERTEX MOVEMENT TRACKING (End of training_step)
        # ================================================================
        if self.global_step % 100 == 0:
            try:
                if hasattr(self.model, 'nodes') and 'object' in self.model.nodes:
                    obj_node = self.model.nodes['object']
                    if hasattr(obj_node, 'server') and hasattr(obj_node.server, 'object_model'):
                        obj_model = obj_node.server.object_model
                        if hasattr(obj_model, 'v3d_cano'):
                            v3d = obj_model.v3d_cano

                            # Movement metrics
                            vertex_std = v3d.std().item()
                            vertex_range = (v3d.max() - v3d.min()).item()
                            vertex_mean = v3d.mean().item()

                            logger.info(
                                f"[Object Vertices] Step {self.global_step}: "
                                f"std={vertex_std:.6f}, range={vertex_range:.4f}, mean={vertex_mean:.4f}"
                            )

                            # Gradient check
                            if v3d.grad is not None:
                                grad_norm = v3d.grad.norm().item()
                                logger.info(f"[Object Gradient] norm={grad_norm:.6e}")
                            else:
                                logger.warning("[Object Gradient] No gradients!")
            except Exception as e:
                logger.error(f"[Vertex Tracking] Failed: {e}")

        # STEP-LEVEL misc export: every save_misc_every steps
        if getattr(self, "save_misc_every", -1) > 0 and self.global_step > 0:
            if self.global_step % self.save_misc_every == 0 and not getattr(self.args, "no_meshing", False):
                # Use a consistent tag for meshing; "misc" is fine
                self.meshing_cano("misc")
                self.save_misc()

        return {'loss': final_loss.detach()}

    # ====================================================================
    # HELPER METHODS
    # ====================================================================

    def _reinit_object_sdf_from_template(self, object_model, batch):
        # 1. Extract category (same pattern as lines 2578–2587)
        init_category = batch.get('object_category') or batch.get('category')

        while isinstance(init_category, (list, tuple)) and len(init_category) > 0:
            init_category = init_category[0]
        if isinstance(init_category, torch.Tensor):
            init_category = (init_category.item() if init_category.numel() == 1
                             else str(init_category[0].item()))
        if init_category:
            init_category = str(init_category).strip().lower()

        if not (init_category and init_category in self.object_templates):
            logger.warning(f"[SDF REINIT] No template for '{init_category}', aborting.")
            return

        if not hasattr(object_model, 'sdf_grid'):
            # In mesh-first mode we may not have a persistent grid; just warn and return.
            logger.warning("[SDF REINIT] object_model has no sdf_grid attribute; "
                           "skipping SDF teacher reinit.")
            return

        sdf_grid = object_model.sdf_grid
        template_verts = self.object_templates[init_category]

        # Build a fresh volumetric teacher grid from the template mesh
        initialized_sdf = self._mesh_to_sdf_grid(
            template_verts,
            grid_resolution=sdf_grid.shape[-1],
            padding=0.1,
        )

        if initialized_sdf is None:
            logger.error(f"[SDF REINIT] mesh_to_sdf_grid failed for '{init_category}'.")
            return

        with torch.no_grad():
            # Treat sdf_grid as a derived / teacher field
            sdf_grid.data = initialized_sdf.to(sdf_grid.device).detach()

        logger.info(
            f"[SDF REINIT] Success for '{init_category}'. "
            f"mean={sdf_grid.mean().item():.4f}, "
            f"std={sdf_grid.std().item():.6f}, "
            f"range=[{sdf_grid.min().item():.4f}, {sdf_grid.max().item():.4f}]"
        )

    def _extract_sdf_grid_from_nodes(self, batch, resolution=32):
        """Extract SDF values on regular grid from object node."""
        # ================================================================
        # ✅ MEMORY OPTIMIZATION: Clear cache before expensive operation
        # ================================================================
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ✅ FORCE no gradient tracking
        original_grad_mode = torch.is_grad_enabled()
        if not self.training:
            torch.set_grad_enabled(False)

        try:  # ✅ FIX: Wrap in try-finally to ensure gradient mode is restored
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
            # (kept consistent with existing marching-cubes / mesh code)
            x = torch.linspace(-1.5, 1.5, H, device=device, requires_grad=False)
            # try:
            #     grid = torch.stack(torch.meshgrid(x, x, x, indexing='ij'), dim=-1)
            # except TypeError:
            #     grid = torch.stack(torch.meshgrid(x, x, x), dim=-1)
            grid = torch.stack(torch.meshgrid(x, x, x), dim=-1)
            grid = grid.detach()
            grid = grid.unsqueeze(0).expand(B, -1, -1, -1, -1)  # (B, H, H, H, 3)

            # Step 3: Find object node
            object_node = None
            for node in self.model.nodes.values():
                if "object" in node.node_id.lower():
                    object_node = node
                    break

            if object_node is None:
                logger.warning("[Helper] No object node found, returning zero SDF")
                object_sdf = torch.zeros(B, 1, resolution, resolution, resolution, device=device)
                return object_sdf

            # ============================================================
            # NEW: Preferred path (Option B) – implicit decoder + FiLM
            # ============================================================
            if hasattr(object_node, "implicit_network"):
                try:
                    # Flatten grid to [B, H³, 3]
                    grid_flat = grid.reshape(B, -1, 3)

                    sdf_batches = []
                    for b in range(B):
                        # Query implicit SDF for this batch element at canonical points
                        x_c = grid_flat[b]  # [H³, 3]
                        sdf_b = self._query_object_implicit_sdf(object_node, x_c, batch)  # [H³, 1] (FiLM‑conditioned)

                        if sdf_b.dim() == 1:
                            sdf_b = sdf_b.unsqueeze(-1)
                        # Reshape to [H, H, H, 1]
                        sdf_b = sdf_b.reshape(H, H, H, 1)
                        sdf_batches.append(sdf_b)

                    # Stack to [B, H, H, H, 1] → [B, 1, H, H, H]
                    object_sdf = torch.stack(sdf_batches, dim=0).permute(0, 4, 1, 2, 3).contiguous()

                    logger.debug(
                        f"[Helper] IMPLICIT SUCCESS: implicit_network + FiLM, "
                        f"object_sdf.shape={object_sdf.shape}"
                    )
                    return object_sdf
                except Exception as e:
                    logger.debug(f"[Helper] IMPLICIT FAILED: implicit_network query: {e}")
                    # Fall through to legacy helpers below

            # >>> LEGACY / FALLBACK PATHS (Option A or old models) <<<

            # Step 3.5: mesh → SDF from v3d_cano (initialization / fallback)
            obj_model = getattr(getattr(object_node, "server", None), "object_model", None)
            if obj_model is not None and hasattr(obj_model, "v3d_cano"):
                verts = obj_model.v3d_cano
                if isinstance(verts, torch.Tensor) and verts.numel() > 0:
                    sdf = self._mesh_to_sdf_grid(
                        verts,
                        grid_resolution=resolution,
                        padding=0.1,
                    )  # returns [1, D, H, W] or [D,H,W]

                    if sdf is not None:
                        # Normalize shape to [1,1,D,H,W]
                        if sdf.dim() == 3:
                            sdf = sdf.unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
                        elif sdf.dim() == 4:
                            sdf = sdf.unsqueeze(0)              # [1,1,D,H,W]

                        object_sdf = sdf.expand(B, -1, -1, -1, -1).contiguous()
                        return object_sdf
                # if verts empty or mesh_to_sdf_grid failed, fall through

            # Legacy methods using server / node outputs
            grid_flat = grid.reshape(B, -1, 3)

            # Step 4: Try multiple SDF extraction methods
            sdf_values = None

            # METHOD 1: server.forward_sdf (preferred)
            if hasattr(object_node, "server") and hasattr(object_node.server, "forward_sdf"):
                try:
                    idx = batch.get('idx', torch.zeros(B, dtype=torch.long, device=device))
                    if idx.ndim > 1:
                        idx = idx.squeeze(-1)

                    # Expand idx to match grid_flat points [B, H³]
                    num_points = grid_flat.shape[1]
                    idx_expanded = idx.unsqueeze(1).expand(-1, num_points)  # [B, H³]

                    # Pass both points AND indices
                    sdf_output = object_node.server.forward_sdf(
                        grid_flat,  # [B, H³, 3] - spatial coordinates
                        idx_expanded  # [B, H³] - frame indices for each point
                    )
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

            # METHOD 4: Query via model's render_core / implicit_network (legacy)
            if sdf_values is None and hasattr(object_node, "implicit_network"):
                try:
                    logger.debug("[Helper] Attempting METHOD 4: model render_core query")

                    # Get batch indices
                    idx = batch.get('idx', torch.zeros(B, dtype=torch.long, device=device))

                    # FIX: Ensure idx is 1D [B] before expand
                    if idx.ndim > 1:
                        idx = idx.squeeze()
                        if idx.ndim == 0:
                            idx = idx.unsqueeze(0)

                    logger.debug(f"[Helper] idx shape after squeeze: {idx.shape}")

                    # Now safe to expand
                    num_points = grid_flat.shape[1]  # H³
                    idx_expanded = idx.unsqueeze(1).expand(-1, num_points)

                    logger.debug(f"[Helper] idx_expanded shape: {idx_expanded.shape}")

                    model_input = {
                        'points': grid_flat,  # [B, H³, 3]
                        'indices': idx_expanded,  # [B, H³]
                    }

                    # Forward through model
                    if hasattr(object_node, 'implicit_network'):
                        # Get feature vector for this sample
                        if hasattr(object_node, 'frame_latent_encoder'):
                            features = object_node.frame_latent_encoder(idx)  # [B, feature_dim]
                        elif hasattr(object_node, 'feature_vector'):
                            features = object_node.feature_vector.weight[idx]  # [B, feature_dim]
                        else:
                            features = None

                        # Query implicit network point-by-point
                        sdf_list = []
                        for b in range(B):
                            points_b = grid_flat[b]  # [H³, 3]

                            if features is not None:
                                feat_b = features[b].unsqueeze(0).expand(points_b.shape[0], -1)
                                input_b = torch.cat([points_b, feat_b], dim=-1)
                            else:
                                input_b = points_b

                            try:
                                # ✅ FIX: Create cond dict for implicit_network
                                # ImplicitNet expects: forward(input, cond) where cond[self.cond] = features
                                if features is not None:
                                    # Use 'pose' as the conditioning key (standard for object networks)
                                    cond = {'pose': feat_b}  # [feature_dim]
                                else:
                                    # No features - use zeros as fallback
                                    cond = {'pose': torch.zeros(32, device=device)}

                                output_b = object_node.implicit_network(points_b.unsqueeze(0), cond)

                                if isinstance(output_b, dict):
                                    sdf_b = output_b.get('sdf', output_b.get('model_out', output_b.get('output')))
                                else:
                                    sdf_b = output_b

                                if sdf_b.dim() == 2:
                                    sdf_b = sdf_b.unsqueeze(-1)
                                if sdf_b.shape[-1] != 1:
                                    sdf_b = sdf_b[..., :1]

                                sdf_list.append(sdf_b.squeeze(0))  # [H³, 1]
                            except Exception as e_inner:
                                logger.debug(f"[Helper] Batch {b} failed: {e_inner}, using zeros")
                                sdf_list.append(torch.zeros(1, points_b.shape[0], 1, device=device))

                        if len(sdf_list) == B:
                            sdf_values = torch.stack(sdf_list, dim=0)  # [B, H³, 1]
                            logger.debug(f"[Helper] METHOD 4 SUCCESS: Per-batch query: {sdf_values.shape}")

                    if sdf_values is not None:
                        logger.debug(f"[Helper] METHOD 4 SUCCESS: implicit_network: {sdf_values.shape}")
                except Exception as e:
                    logger.debug(f"[Helper] METHOD 4 FAILED: implicit_network: {e}")
                    import traceback
                    logger.debug(f"[Helper] Traceback: {traceback.format_exc()}")

            # FALLBACK: Return zero grid
            if sdf_values is None:
                # ADD DETAILED DIAGNOSTICS:
                logger.error(f"[SDF Debug] All methods failed for node: {object_node.node_id}")
                logger.error(f"  Has server: {hasattr(object_node, 'server')}")
                if hasattr(object_node, 'server'):
                    logger.error(f"  Has forward_sdf: {hasattr(object_node.server, 'forward_sdf')}")
                    logger.error(f"  Has shape_net: {hasattr(object_node.server, 'shape_net')}")
                    logger.error(f"  Has object_model: {hasattr(object_node.server, 'object_model')}")
                logger.error(f"  Has forward: {hasattr(object_node, 'forward')}")
                logger.error(f"  Has implicit_network: {hasattr(object_node, 'implicit_network')}")

                # ALSO check if all SDF methods returned None
                logger.warning("[Helper] Listing all node attributes:")
                logger.warning(f"  {dir(object_node)}")

                logger.warning(
                    "[Helper] All SDF extraction methods failed, using zero grid. "
                    "This is expected in early training before object geometry is initialized. "
                    f"Attempted methods: forward_sdf, shape_net, forward, implicit_network on node '{object_node.node_id}'"
                )
                object_sdf = torch.zeros(B, 1, resolution, resolution, resolution, device=device)
                return object_sdf

            # Step 5: Reshape to (B, 1, H, H, H) format
            if sdf_values.dim() == 2:
                sdf_values = sdf_values.unsqueeze(-1)
            if sdf_values.shape[-1] != 1:
                sdf_values = sdf_values[..., :1]

            object_sdf = sdf_values.reshape(B, resolution, resolution, resolution, 1).permute(
                0, 4, 1, 2, 3
            ).contiguous()  # [B,1,H,H,H]

            return object_sdf

        finally:  # ✅ CRITICAL: This ALWAYS executes, even on exception
            # Restore gradient mode
            torch.set_grad_enabled(original_grad_mode)

            # Clear cache after expensive operation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return object_sdf

    def _add_sdf_distillation_loss(self, object_model, batch, step, geometric_loss, loss_output):
        """
        Option B: encoder-based geometry – disable grid→implicit SDF distillation.

        In this regime, geometry is provided by a pretrained frozen
        GeometryEncoder3D + ResidualMLP latent prior, so we no longer
        pull the implicit SDF toward object_model.sdf_grid.

        This function is kept for API compatibility with the training
        loop, but returns geometric_loss unchanged.
        """

        # Read training mode flags (optional, but makes behavior explicit)
        training_cfg = getattr(self.opt, "training", None)
        geometry_mode = getattr(training_cfg, "geometry_mode", "encoder") if training_cfg is not None else "encoder"
        use_sdf_distill = getattr(training_cfg, "use_sdf_distill", False) if training_cfg is not None else False

        # For encoder-based geometry (default in Option B), always skip distillation
        if geometry_mode == "encoder" or not use_sdf_distill:
            if step == 0:
                logger.info(
                    f"[SDF-DISTILL] Disabled "
                    f"(geometry_mode={geometry_mode}, use_sdf_distill={use_sdf_distill})."
                )

            # Keep loss keys for logging / downstream code, but force them to zero
            loss_output["loss/sdf_distill_raw"] = 0.0
            loss_output["loss/sdf_distill"] = 0.0
            loss_output["loss_sdf_distill_raw"] = 0.0
            loss_output["loss_sdf_distill"] = 0.0

            return geometric_loss

        # If you ever re-enable grid-based geometry (Option A), you can
        # reintroduce the previous grid→implicit distillation implementation here.
        return geometric_loss

    def _query_object_implicit_sdf(self, object_node, x_c, batch):
        """
        Query the object's implicit_network at canonical points x_c: [N, 3].

        In the encoder + FiLM setup (Option B), we condition the implicit SDF
        on the frozen geometry latent z_geo_refined, which the network uses
        internally to produce FiLM (gamma/beta) parameters instead of simple
        concatenation with the input.
        """
        # x_c: [N, 3] -> [1, N, 3]
        x_c_batched = x_c.unsqueeze(0)

        # Get implicit network
        implicit_net = object_node.implicit_network
        cond = {}

        # The network's self.cond attribute tells us which key to use
        if hasattr(implicit_net, "cond") and implicit_net.cond != "none":
            cond_key = implicit_net.cond  # e.g., "geo", "pose", etc.

            # Preferred: use geometry latent from encoder + residual MLP
            z_geo = getattr(object_node, "z_geo_refined", None)

            if z_geo is not None:
                # Ensure shape [1, D] and correct device
                if z_geo.dim() == 1:
                    z_geo = z_geo.unsqueeze(0)
                z_geo = z_geo.to(x_c.device)

                # Use z_geo_refined as conditioning vector for FiLM
                cond[cond_key] = z_geo  # [1, D]
            else:
                logger.warning(f"[SDF Query] z_geo_refined is None on {object_node.node_id}")
                if hasattr(object_node, 'server'):
                    if hasattr(object_node.server, 'object_model'):
                        obj_model = object_node.server.object_model
                        if hasattr(obj_model, 'z_geo_refined'):
                            z_geo = obj_model.z_geo_refined
                        elif hasattr(obj_model, 'get_z_geo_refined'):
                            z_geo = obj_model.get_z_geo_refined()
                # Fallback: zero latent with correct dim for backward compatibility
                if hasattr(implicit_net, "num_cond"):
                    cond_dim = implicit_net.num_cond
                else:
                    # Conservative default; should be overridden by num_cond
                    cond_dim = 128
                cond[cond_key] = torch.zeros(1, cond_dim, device=x_c.device)
        # Add debug in _query_object_implicit_sdf
        logger.info(
            f"[SDF Query] z_geo_refined stats: min={z_geo.min():.4f}, max={z_geo.max():.4f}, mean={z_geo.mean():.4f}")

        # Forward through FiLM-conditioned implicit network
        out = implicit_net(x_c_batched, cond)  # [1, N, C] or dict

        # Extract SDF channel
        if isinstance(out, dict):
            sdf_student = out.get("sdf", out.get("model_out", out.get("output")))
        else:
            sdf_student = out

        # Ensure shape [N, 1]
        if sdf_student.dim() == 3:
            # [1, N, C] -> [N, C]
            sdf_student = sdf_student.squeeze(0)
        if sdf_student.dim() == 1:
            sdf_student = sdf_student.unsqueeze(-1)
        if sdf_student.shape[-1] != 1:
            sdf_student = sdf_student[..., :1]

        return sdf_student  # [N, 1]

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
            global_orient = batch[global_orient_key]  # [B, 3] or [B, T, 3]
            pose = batch[pose_key]                     # [B, 45] or [B, T, 45]
            full_pose = torch.cat([global_orient, pose], dim=-1)  # [B, 48] or [B, T, 48]
            logger.debug(f"[Phase 4] Concatenated {global_orient_key} + {pose_key}: {full_pose.shape}")

            # ✅ ADD THIS FIX HERE:
            # Fix: Ensure full_pose is 2D [B, 48] for MANO
            if full_pose.ndim == 3:
                # Reshape from [B, T, 48] to [B*T, 48]
                B_orig, T_orig, num_params = full_pose.shape
                full_pose = full_pose.reshape(B_orig * T_orig, num_params)
                logger.debug(f"[Phase 4] Reshaped full_pose from 3D to 2D: {full_pose.shape}")

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

        # ✅ ADD THIS FIX HERE:
        # Fix: Ensure mano_shape is 2D [B, 10] for MANO
        if mano_shape.ndim == 3:
            # Reshape from [B, T, 10] to [B*T, 10]
            B_orig, T_orig, num_betas = mano_shape.shape
            mano_shape = mano_shape.reshape(B_orig * T_orig, num_betas)
            logger.debug(f"[Phase 4] Reshaped mano_shape from 3D to 2D: {mano_shape.shape}")
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

        # ✅ ADD THIS FIX HERE:
        # Fix: Ensure mano_trans is 2D [B, 3] for MANO
        if mano_trans.ndim == 3:
            # Reshape from [B, T, 3] to [B*T, 3]
            B_orig, T_orig, coord_dim = mano_trans.shape
            mano_trans = mano_trans.reshape(B_orig * T_orig, coord_dim)
            logger.debug(f"[Phase 4] Reshaped mano_trans from 3D to 2D: {mano_trans.shape}")
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
                hand_verts = mano_output['verts'].detach()  # [B, 778, 3]
            else:
                hand_verts = mano_output.detach()  # [B, 778, 3]

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

    def _get_template_mesh(self, batch):
        """
        Generate canonical MANO template for shape regularization.

        This is self-supervised (no GT dependency) and prevents:
        - Mesh collapse
        - Wild deformations
        - Non-hand-like geometries

        Returns:
            template_verts: [B, 778, 3] canonical MANO vertices
        """
        device = next(self.parameters()).device
        # Try multiple RGB key variants
        rgb_key = None
        for key in ['gt.rgb', 'rgb', 'target_rgb', 'image']:
            if key in batch:
                rgb_key = key
                break

        if rgb_key is None:
            raise ValueError(f"Cannot find RGB in batch. Available keys: {list(batch.keys())[:20]}")

        batch_size = batch[rgb_key].shape[0]

        # Canonical MANO: zero pose, mean shape, zero translation
        zero_pose = torch.zeros(batch_size, 48, device=device)
        mean_shape = torch.zeros(batch_size, 10, device=device)
        zero_trans = torch.zeros(batch_size, 3, device=device)
        scene_scale = torch.ones(batch_size, device=device)

        # Get hand node's MANO server (match the iteration pattern)
        hand_node = None
        for node in self.model.nodes.values():
            if 'right' in node.node_id.lower():
                hand_node = node
                break

        if hand_node is None:
            raise RuntimeError("Hand node 'right' not found for template generation")

        # Generate template (no gradients needed - it's a constant reference)
        with torch.no_grad():
            mano_output = hand_node.server(
                scene_scale,
                zero_trans,
                zero_pose,
                mean_shape,
                absolute=False
            )

        template_verts = mano_output['verts'] if isinstance(mano_output, dict) else mano_output

        logger.debug(f"[Template] Generated canonical MANO: {template_verts.shape}")
        return template_verts.detach()  # [B, 778, 3]

    def _extract_hand_mesh_for_chamfer(self, batch):
        """
        Extract current hand mesh WITH GRADIENTS for Chamfer loss.

        Key difference from _extract_hand_mesh():
        - NO torch.no_grad() wrapper
        - NO .detach() on vertices
        - Returns only vertices (faces not needed for Chamfer)

        Returns:
            hand_verts: [B, 778, 3] with gradients attached
        """
        hand_node = None
        for node in self.model.nodes.values():
            if 'right' in node.node_id.lower():
                hand_node = node
                node_id = node.node_id
                break

        if hand_node is None:
            raise ValueError("[Chamfer] No hand node found")

        # Extract MANO parameters
        fullpose_key = f"{node_id}.fullpose"
        pose_key = f"{node_id}.pose"
        global_orient_key = f"{node_id}.global_orient"
        shape_key = f"{node_id}.betas"
        trans_key = f"{node_id}.transl"

        # ================================================================
        # COPY FROM LINES 4328-4445 STARTS HERE
        # ================================================================

        # Extract pose (48-dim full pose)
        full_pose = None

        if fullpose_key in batch:
            full_pose = batch[fullpose_key]
            logger.debug(f"[Chamfer] Using {fullpose_key}: {full_pose.shape}")

        elif global_orient_key in batch and pose_key in batch:
            global_orient = batch[global_orient_key]
            pose = batch[pose_key]
            full_pose = torch.cat([global_orient, pose], dim=-1)
            logger.debug(f"[Chamfer] Concatenated {global_orient_key} + {pose_key}: {full_pose.shape}")

            # Fix: Ensure full_pose is 2D [B, 48] for MANO
            if full_pose.ndim == 3:
                B_orig, T_orig, num_params = full_pose.shape
                full_pose = full_pose.reshape(B_orig * T_orig, num_params)
                logger.debug(f"[Chamfer] Reshaped full_pose from 3D to 2D: {full_pose.shape}")

        elif pose_key in batch:
            pose = batch[pose_key]
            batch_size = pose.shape[0]
            device = pose.device
            zero_global = torch.zeros(batch_size, 3, device=device)
            full_pose = torch.cat([zero_global, pose], dim=-1)
            logger.warning(f"[Chamfer] No global_orient found, using zeros")

        else:
            available_keys = [k for k in batch.keys() if node_id in k]
            raise ValueError(
                f"[Chamfer] Cannot find MANO pose in batch.\n"
                f"  Tried keys: {fullpose_key}, {pose_key}, {global_orient_key}\n"
                f"  Available keys for '{node_id}': {available_keys}"
            )

        # Extract shape parameters (betas)
        mano_shape = batch.get(shape_key, None)

        if mano_shape is None:
            batch_size = full_pose.shape[0]
            device = full_pose.device
            mano_shape = torch.zeros(batch_size, 10, device=device)
            logger.warning(f"[Chamfer] No shape parameters at {shape_key}, using mean shape")
        else:
            logger.debug(f"[Chamfer] Using {shape_key}: {mano_shape.shape}")

        # Fix: Ensure mano_shape is 2D [B, 10]
        if mano_shape.ndim == 3:
            B_orig, T_orig, num_betas = mano_shape.shape
            mano_shape = mano_shape.reshape(B_orig * T_orig, num_betas)
            logger.debug(f"[Chamfer] Reshaped mano_shape from 3D to 2D: {mano_shape.shape}")

        # Extract translation
        mano_trans = batch.get(trans_key, None)

        if mano_trans is None:
            batch_size = full_pose.shape[0]
            device = full_pose.device
            mano_trans = torch.zeros(batch_size, 3, device=device)
            logger.warning(f"[Chamfer] No translation at {trans_key}, using zero translation")
        else:
            logger.debug(f"[Chamfer] Using {trans_key}: {mano_trans.shape}")

        # Fix: Ensure mano_trans is 2D [B, 3]
        if mano_trans.ndim == 3:
            B_orig, T_orig, coord_dim = mano_trans.shape
            mano_trans = mano_trans.reshape(B_orig * T_orig, coord_dim)
            logger.debug(f"[Chamfer] Reshaped mano_trans from 3D to 2D: {mano_trans.shape}")

        # Extract scene_scale (required by MANOServer)
        scene_scale = None

        # Try to get from batch
        scale_keys = [
            f"{node_id}.scene_scale",
            "scene_scale",
            f"{node_id}.scale"
        ]

        for scale_key in scale_keys:
            if scale_key in batch:
                scene_scale = batch[scale_key]
                logger.debug(f"[Chamfer] Using {scale_key}: {scene_scale.shape}")
                break

        # Try node attributes
        if scene_scale is None and hasattr(hand_node, 'scene_scale'):
            scene_scale = hand_node.scene_scale
            logger.debug(f"[Chamfer] Using hand_node.scene_scale: {scene_scale}")

        # Default to 1.0
        if scene_scale is None:
            batch_size = full_pose.shape[0]
            device = full_pose.device
            scene_scale = torch.ones(batch_size, device=device)
            logger.warning("[Chamfer] No scene_scale found, using 1.0")

        # Ensure scene_scale is correct shape
        if scene_scale.dim() == 0:
            batch_size = full_pose.shape[0]
            scene_scale = scene_scale.unsqueeze(0).expand(batch_size)
        elif scene_scale.shape[0] != full_pose.shape[0]:
            batch_size = full_pose.shape[0]
            scene_scale = scene_scale.view(-1)[0].unsqueeze(0).expand(batch_size)

        # ================================================================
        # COPY ENDS HERE
        # ================================================================

        # CRITICAL: No torch.no_grad() here! We need gradients!
        mano_output = hand_node.server(
            scene_scale,
            mano_trans,
            full_pose,
            mano_shape,
            absolute=False
        )

        # Extract vertices - keep gradients!
        hand_verts = mano_output['verts'] if isinstance(mano_output, dict) else mano_output
        # DON'T call .detach() - we need gradients to flow back to MANO params

        return hand_verts  # [B, 778, 3] with requires_grad=True

    def _get_teacher_mesh_from_grid(self, object_model, max_verts=50000):
        """
        Extract (and cache) a low-res mesh from the teacher SDF grid using marching cubes.
        Used only for distillation; no gradients needed.
        """
        import numpy as np
        from skimage import measure

        if not hasattr(object_model, "sdf_grid"):
            return None, None

        # Cache so we don't run marching cubes every step
        if hasattr(object_model, "_distill_teacher_verts") and hasattr(object_model, "_distill_teacher_normals"):
            return object_model._distill_teacher_verts, object_model._distill_teacher_normals

        sdf_grid = object_model.sdf_grid.detach().cpu().numpy()  # [1, D, H, W] or [D,H,W]
        if sdf_grid.ndim == 4:
            sdf_grid = sdf_grid[0, 0]  # assume [1,1,D,H,W]

        try:
            verts, faces, normals, values = measure.marching_cubes(
                sdf_grid, level=0.0,
            )
        except Exception:
            return None, None

        # Normalize verts from voxel index space [0, D/H/W) to [-1,1]^3
        D, H, W = sdf_grid.shape
        z = verts[:, 0] / (D - 1) * 2.0 - 1.0
        y = verts[:, 1] / (H - 1) * 2.0 - 1.0
        x = verts[:, 2] / (W - 1) * 2.0 - 1.0
        verts_n = np.stack([x, y, z], axis=-1)  # [N,3] in [-1,1]

        # Optional subsample for efficiency
        if verts_n.shape[0] > max_verts:
            idx = np.random.choice(verts_n.shape[0], max_verts, replace=False)
            verts_n = verts_n[idx]
            normals = normals[idx]

        device = object_model.sdf_grid.device
        verts_t = torch.from_numpy(verts_n).to(device=device, dtype=torch.float32)
        normals_t = torch.from_numpy(normals).to(device=device, dtype=torch.float32)

        # Cache
        object_model._distill_teacher_verts = verts_t
        object_model._distill_teacher_normals = normals_t

        return verts_t, normals_t

    def _extract_object_mesh_from_sdf(self, batch):
        """
        Extract object mesh from implicit SDF via Marching Cubes and
        remove small disconnected floaters by keeping only the largest component.

        Returns:
            obj_verts_list: list of [Ni, 3] vertex tensors in canonical space (≈[-1.5, 1.5]³),
                            one per batch element – use these as instance meshes.
            obj_faces_list: list of [Mi, 3] face index tensors, one per batch element.

        These instance meshes can be voxelized for the geometry encoder via:
            voxel_sdf = self._mesh_to_sdf_grid(
                obj_verts_list[b],
                grid_resolution=32,
                padding=0.1,
                for_encoder=True,
            )  # → [1, 1, 64, 64, 64]
        """
        from skimage import measure

        # Extract SDF grid from object node (implicit decoder + FiLM or fallbacks)
        resolution = self.mesh_resolution
        object_sdf = self._extract_sdf_grid_from_nodes(batch, resolution=resolution).detach()

        batch_size = object_sdf.shape[0]
        obj_verts_list = []
        obj_faces_list = []

        for b in range(batch_size):
            # [H, H, H] grid for this batch element
            sdf_grid = object_sdf[b, 0].cpu().numpy().copy()
            logger.debug(
                f"[Stride Debug] sdf_grid strides: {sdf_grid.strides}, "
                f"negative: {any(s < 0 for s in sdf_grid.strides)}"
            )

            try:
                # Apply Marching Cubes in canonical volume [0, 3]³ with spacing 3 / H
                verts, faces, normals, values = measure.marching_cubes(
                    sdf_grid,
                    level=0.0,
                    spacing=(3.0 / resolution, 3.0 / resolution, 3.0 / resolution),
                )
                logger.debug(
                    f"[MC Debug] verts shape: {verts.shape}, strides: {verts.strides}, "
                    f"negative: {any(s < 0 for s in verts.strides)}"
                )
                logger.debug(f"[MC Debug] faces shape: {faces.shape}, strides: {faces.strides}")

                # ------------------------------------------------------------------
                # Compactness / anti-floater: keep only largest connected component
                # ------------------------------------------------------------------
                num_verts = verts.shape[0]
                num_faces = faces.shape[0]

                if num_verts > 0 and num_faces > 0:
                    # Build vertex adjacency from faces
                    neighbors = [[] for _ in range(num_verts)]
                    for f in faces:
                        i, j, k = int(f[0]), int(f[1]), int(f[2])
                        neighbors[i].append(j); neighbors[j].append(i)
                        neighbors[j].append(k); neighbors[k].append(j)
                        neighbors[k].append(i); neighbors[i].append(k)

                    labels = -np.ones(num_verts, dtype=np.int32)
                    comp_id = 0

                    for v in range(num_verts):
                        if labels[v] != -1:
                            continue
                        # BFS/DFS to label this component
                        stack = [v]
                        labels[v] = comp_id
                        while stack:
                            curr = stack.pop()
                            for nb in neighbors[curr]:
                                if labels[nb] == -1:
                                    labels[nb] = comp_id
                                    stack.append(nb)
                        comp_id += 1

                    if comp_id > 1:
                        comp_sizes = np.bincount(labels)
                        main_comp = int(comp_sizes.argmax())
                        keep_mask = (labels == main_comp)
                        kept_indices = np.nonzero(keep_mask)[0]

                        # Map old vertex indices to new compact indices
                        new_index = -np.ones(num_verts, dtype=np.int64)
                        new_index[kept_indices] = np.arange(kept_indices.shape[0], dtype=np.int64)

                        # Keep only faces fully inside the main component
                        face_keep_mask = keep_mask[faces].all(axis=1)
                        faces_main = faces[face_keep_mask]
                        faces_main = new_index[faces_main]

                        verts_main = verts[kept_indices]

                        logger.debug(
                            f"[Compactness] Batch {b}: {comp_id} components, "
                            f"keeping {verts_main.shape[0]} verts / {faces_main.shape[0]} faces "
                            f"(dropped {num_verts - verts_main.shape[0]} floater verts)."
                        )

                        verts = verts_main
                        faces = faces_main
                    else:
                        logger.debug(f"[Compactness] Batch {b}: single component, no floater removal.")
                else:
                    logger.debug(f"[Compactness] Batch {b}: empty mesh from marching cubes.")

                # Shift to canonical [-1.5, 1.5]³ coordinate system
                verts = verts - 1.5

                # Convert to tensors on the original device
                obj_verts_list.append(torch.from_numpy(verts.copy()).float().to(object_sdf.device))
                obj_faces_list.append(torch.from_numpy(faces.copy()).long().to(object_sdf.device))

                logger.debug(
                    f"[Phase 4] Extracted object mesh {b}: {verts.shape[0]} verts, {faces.shape[0]} faces"
                )

            except Exception as e:
                logger.warning(f"[Phase 4] Marching Cubes failed for batch {b}: {e}")
                # Fallback to empty mesh
                obj_verts_list.append(torch.zeros((0, 3), device=object_sdf.device))
                obj_faces_list.append(torch.zeros((0, 3), dtype=torch.long, device=object_sdf.device))

        return obj_verts_list, obj_faces_list
    # ====================================================================

    def _mesh_to_sdf_grid(self, vertices, grid_resolution=16, padding=0.1, for_encoder: bool = False):
        """
        Convert a single instance mesh (vertices) to an approximate SDF grid.

        This is used both to:
          - initialize / re-init SDF teacher grids from templates, and
          - build a 64³ voxelized SDF volume as encoder input (Option B).

        Args:
            vertices: [N, 3] canonical-space vertices for one instance.
            grid_resolution: SDF grid resolution (default 32 for encoder).
            padding: Extra scale around the mesh bounding box.
            for_encoder: If True, return [1, 1, R, R, R] ready for a 3D CNN.

        Returns:
            If for_encoder=False (default): [1, R, R, R] float32 SDF grid.
            If for_encoder=True:         [1, 1, R, R, R] float32 SDF grid.
        """
        if vertices is None or vertices.shape[0] == 0:
            logger.error("[SDF Init] Cannot initialize from empty mesh")
            return None

        device = vertices.device

        # Normalize vertices to canonical box [-1, 1]³ (instance-specific)
        v_min = vertices.min(dim=0)[0]
        v_max = vertices.max(dim=0)[0]
        v_center = (v_min + v_max) / 2
        v_scale = (v_max - v_min).max() * (1 + padding)
        v_scale = torch.clamp(v_scale, min=1e-6)  # Prevent division by zero

        vertices_norm = (vertices - v_center) / (v_scale / 2)

        # Create 3D grid in canonical space [-1, 1]^3
        grid_coords = torch.linspace(-1, 1, grid_resolution, device=device)

        # Safe for old PyTorch versions (no indexing kwarg)
        grid_x, grid_y, grid_z = torch.meshgrid(
            grid_coords, grid_coords, grid_coords
        )

        # Flatten grid to list of points
        grid_points = torch.stack(
            [grid_x, grid_y, grid_z], dim=-1
        ).reshape(-1, 3)

        # Compute unsigned distances (batched for memory)
        batch_size = 5000
        distances = []

        for i in range(0, len(grid_points), batch_size):
            batch = grid_points[i:i + batch_size]
            dists = torch.cdist(batch, vertices_norm)
            min_dists = dists.min(dim=1)[0]
            distances.append(min_dists)

        distances = torch.cat(distances, dim=0)
        unsigned_sdf = distances.reshape(grid_resolution, grid_resolution, grid_resolution)

        # Rough sign estimation (inside vs outside) using centroid heuristic
        centroid = vertices_norm.mean(dim=0)
        grid_to_centroid = torch.norm(grid_points - centroid, dim=1)
        grid_to_centroid = grid_to_centroid.reshape(grid_resolution, grid_resolution, grid_resolution)

        median_dist = grid_to_centroid.median()
        is_inside = grid_to_centroid < median_dist * 0.7

        signed_sdf = unsigned_sdf.clone()
        signed_sdf[is_inside] = -signed_sdf[is_inside]

        # Default behavior (backwards compatible): [1, R, R, R]
        sdf_grid = signed_sdf.unsqueeze(0).contiguous()  # [1, R, R, R]

        # For encoder input, add channel dimension → [1, 1, R, R, R]
        if for_encoder:
            sdf_grid = sdf_grid.unsqueeze(1)  # [1, 1, R, R, R]

        return sdf_grid

    def _validate_sdf_for_extraction(self) -> bool:
        try:
            # FIX: Use 'in' check instead of .get() for ModuleDict
            if 'object' not in self.model.nodes:
                return False
            object_node = self.model.nodes['object']
            if not hasattr(object_node, 'server'):
                return False
            object_server = object_node.server
            if not hasattr(object_server, 'object_model'):
                return False

            obj_model = object_server.object_model

            # If we already ran SDF-HEALTH this step, reuse it
            if hasattr(obj_model, 'sdf_grid'):
                sdf = obj_model.sdf_grid
                sdf_std = sdf.std().item()
                sdf_min = sdf.min().item()
                sdf_max = sdf.max().item()
                has_zero_crossing = (sdf_min < 0.0) and (sdf_max > 0.0)
                std_thresh = 1e-5  # keep simple here

                sdf_valid = has_zero_crossing and sdf_std > std_thresh

                # If SDF is bad but v3d exists, still allow extraction via v3d
                if not sdf_valid and hasattr(obj_model, 'v3d_cano'):
                    return True

                return sdf_valid

            # No SDF grid – fall back to v3d
            if hasattr(obj_model, 'v3d_cano'):
                return True

            return False

        except Exception as e:
            if self.global_step % 500 == 0:
                logger.warning(f"[SDF Validation] Error: {e}")
            return False

    def _is_video_batch(self, batch: Dict) -> bool:
        """
        Check if batch contains video sequence data from GHOP HOI4D dataset.

        SIMPLIFIED VERSION: Only requires hA_n and c2w_n (sufficient for Phase 5).
        The sequence_id check is made OPTIONAL for better compatibility.

        Video batches (GHOP HOI4D) contain:
        - hA_n: Hand pose at frame t+1 (REQUIRED)
        - c2w_n: Camera at frame t+1 (REQUIRED)
        - sequence_id/frame_idx: Metadata (OPTIONAL but helpful)

        Args:
            batch: Training batch dictionary

        Returns:
            bool: True if batch contains video data with temporal fields
        """
        # ================================================================
        # CRITICAL CHECKS: Must have both temporal fields
        # ================================================================
        has_next_hand = 'hA_n' in batch
        has_next_camera = 'c2w_n' in batch

        # ✅ SIMPLIFIED: Only require temporal fields
        # sequence_id is nice-to-have but not required for Phase 5
        is_video = has_next_hand and has_next_camera

        # ================================================================
        # ONE-TIME LOGGING: Log detection result once per training run
        # ================================================================
        if not hasattr(self, '_video_detection_logged'):
            self._video_detection_logged = True

            if is_video:
                # Additional metadata checks (for logging only)
                has_sequence_id = any(k in batch for k in ['video_id', 'sequence_id', 'scene_id'])
                has_temporal_idx = 'frame_idx' in batch or 'temporal_idx' in batch

                logger.info("=" * 70)
                logger.info("VIDEO BATCH DETECTED (Phase 5 will activate)")
                logger.info("=" * 70)
                logger.info("  Required temporal fields:")
                logger.info(f"    ✓ hA_n (next hand pose): {has_next_hand}")
                logger.info(f"    ✓ c2w_n (next camera): {has_next_camera}")
                logger.info("  Optional metadata:")
                logger.info(f"    {'✓' if has_sequence_id else '✗'} sequence_id: {has_sequence_id}")
                logger.info(f"    {'✓' if has_temporal_idx else '✗'} frame_idx: {has_temporal_idx}")
                logger.info("  Phase 5 Temporal Consistency: ENABLED")
                logger.info("=" * 70)
            else:
                logger.info("=" * 70)
                logger.info("SINGLE-IMAGE BATCH DETECTED")
                logger.info("=" * 70)
                logger.info("  Required temporal fields:")
                logger.info(f"    ✗ hA_n (next hand pose): {has_next_hand}")
                logger.info(f"    ✗ c2w_n (next camera): {has_next_camera}")
                logger.info("  Phase 5 Temporal Consistency: SKIPPED")
                logger.info("  This is expected for HOLD single-image dataset")
                logger.info("=" * 70)

        # ================================================================
        # DEBUG LOGGING: Only if still not video after checks
        # ================================================================
        elif not is_video:
            logger.debug(
                f"[Phase 5] Non-video batch: "
                f"has_next_hand={has_next_hand}, has_next_camera={has_next_camera}"
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
                params = node.params(batch['idx'])  # May return xdict or tensor

                # ================================================================
                # FIX: Handle xdict (common.xdict.xdict object)
                # ================================================================
                if hasattr(params, '__class__') and 'xdict' in params.__class__.__name__.lower():
                    # params is xdict - extract tensor from it
                    logger.debug(f"[Phase 5] node.params returned xdict, extracting tensor")

                    # Try common xdict field names
                    if hasattr(params, 'full_pose') and params.full_pose is not None:
                        params = params.full_pose
                    elif hasattr(params, 'pose') and params.pose is not None:
                        params = params.pose
                    elif hasattr(params, 'right') and params.right is not None:
                        # Nested structure: params.right.full_pose or params.right.pose
                        if hasattr(params.right, 'full_pose'):
                            params = params.right.full_pose
                        elif hasattr(params.right, 'pose'):
                            params = params.right.pose
                        else:
                            # Convert entire xdict to tensor (fallback)
                            logger.warning(f"[Phase 5] xdict structure unknown, using batch fallback")
                            params = None
                    else:
                        # Unable to extract from xdict
                        logger.warning(
                            f"[Phase 5] Could not extract tensor from xdict. "
                            f"Available fields: {list(params.keys()) if hasattr(params, 'keys') else 'unknown'}"
                        )
                        params = None

                # Now params should be a tensor (or None if extraction failed)
                if params is not None and isinstance(params, torch.Tensor):
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

            elif 'right.full_pose' in batch:  # Fixed: was 'right.fullpose'
                # MANO fullpose [B, 48]: strip global_orient
                full_pose = batch['right.full_pose']  # [B, 48]
                hand_pose = full_pose[..., 3:]  # [B, 45]
                logger.debug(
                    f"[Phase 5] Using batch 'right.full_pose', stripped to [B, 45]: "
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

        # ================================================================
        # ✅ CRITICAL: Clear PyTorch Lightning's output collection
        # This prevents accumulation of loss dicts from training_step
        # ================================================================
        if outputs is not None and len(outputs) > 0:
            try:
                # Convert to scalars for logging if needed
                avg_loss = torch.stack([x['loss'] for x in outputs]).mean().item()
                logger.debug(f"[Epoch {current_epoch}] Avg loss: {avg_loss:.6f} "
                             f"(num_steps={len(outputs)})")
                # Clear the list
                outputs.clear()
                logger.debug(f"[Epoch {current_epoch}] Cleared outputs collection")
            except Exception as e:
                logger.debug(f"[Epoch {current_epoch}] Could not process outputs: {e}")
                try:
                    outputs.clear()
                except:
                    pass

        # ================================================================
        # ✅ NUCLEAR CLEANUP: Clear ALL model state
        # ================================================================
        try:
            logger.info(f"[Epoch {current_epoch}] Nuclear cleanup starting...")

            # 1. Clear ALL gradients in model
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad = None

            # 2. Clear ALL buffers' gradients
            for name, buffer in self.model.named_buffers():
                if buffer.grad is not None:
                    buffer.grad = None

            # 3. Clear ALL submodule buffers
            for module in self.model.modules():
                for name, buffer in module.named_buffers(recurse=False):
                    if buffer.grad is not None:
                        buffer.grad = None

            # 4. Force zero_grad on ALL optimizers
            for opt in self.trainer.optimizers:
                opt.zero_grad(set_to_none=True)

            # 5. Clear PyTorch internal caches
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            # 6. Python garbage collection
            gc.collect()
            torch.cuda.empty_cache()  # Second pass
            gc.collect()  # Third pass

            mem_after = torch.cuda.memory_allocated() / 1024 ** 2
            logger.info(f"[Epoch {current_epoch}] Nuclear cleanup complete: {mem_after:.2f} MB")

            # ✅ NEW: Immediate cache clear after nuclear cleanup
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"[Epoch {current_epoch}] Nuclear cleanup failed: {e}")

        # ================================================================
        # ✅ CRITICAL: Epoch-End CUDA Cache Clearing
        # Problem: PyTorch caches freed memory across epochs → 490 MB/epoch
        # Solution: Explicitly return cached memory to GPU after cleanup
        # Expected: Prevents reserved memory from growing to 24 GB
        # ================================================================
        if torch.cuda.is_available():
            # ✅ NEW: Immediate aggressive cleanup BEFORE diagnostics
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

            try:
                # Get memory state BEFORE cache clear
                mem_before_allocated = torch.cuda.memory_allocated() / 1024 ** 2
                mem_before_reserved = torch.cuda.memory_reserved() / 1024 ** 2
                cache_before = mem_before_reserved - mem_before_allocated

                logger.info(f"[Epoch {current_epoch}] ════════════════════════════════════════")
                logger.info(f"[Epoch {current_epoch}] BEFORE Epoch-End Cache Clear:")
                logger.info(f"[Epoch {current_epoch}]   Allocated: {mem_before_allocated:.2f} MB")
                logger.info(f"[Epoch {current_epoch}]   Reserved:  {mem_before_reserved:.2f} MB")
                logger.info(f"[Epoch {current_epoch}]   Cache:     {cache_before:.2f} MB")

                # CRITICAL: Clear cache to return memory to GPU
                torch.cuda.synchronize()  # Wait for all GPU operations
                torch.cuda.empty_cache()  # Return cached memory to GPU
                torch.cuda.synchronize()  # Ensure clearing completed

                # Force garbage collection after cache clear
                gc.collect()

                # Get memory state AFTER cache clear
                mem_after_allocated = torch.cuda.memory_allocated() / 1024 ** 2
                mem_after_reserved = torch.cuda.memory_reserved() / 1024 ** 2
                cache_after = mem_after_reserved - mem_after_allocated
                cache_freed = cache_before - cache_after

                logger.info(f"[Epoch {current_epoch}] AFTER Epoch-End Cache Clear:")
                logger.info(f"[Epoch {current_epoch}]   Allocated: {mem_after_allocated:.2f} MB")
                logger.info(f"[Epoch {current_epoch}]   Reserved:  {mem_after_reserved:.2f} MB")
                logger.info(f"[Epoch {current_epoch}]   Cache:     {cache_after:.2f} MB")
                logger.info(f"[Epoch {current_epoch}]   Freed:     {cache_freed:.2f} MB ✓")
                logger.info(f"[Epoch {current_epoch}] ════════════════════════════════════════")

                # Sanity check: Warn if cache still large (> 1 GB)
                if cache_after > 1000:
                    logger.warning(
                        f"[Epoch {current_epoch}] ⚠️  Cache still large after clearing: "
                        f"{cache_after:.2f} MB (expected < 1000 MB)"
                    )
                    logger.warning(
                        f"[Epoch {current_epoch}] This may indicate fragmentation or "
                        f"cached allocations that cannot be freed"
                    )

                # Success metric: Log cumulative cache cleared
                if not hasattr(self, '_total_cache_freed'):
                    self._total_cache_freed = 0.0
                self._total_cache_freed += cache_freed

                logger.info(
                    f"[Epoch {current_epoch}] Cumulative cache freed: "
                    f"{self._total_cache_freed:.2f} MB across {current_epoch + 1} epochs"
                )

            except Exception as e:
                logger.error(f"[Epoch {current_epoch}] Epoch-end cache clearing failed: {e}")

        # ================================================================
        # FIX 1 ENHANCED: Clear Phase 5 temporal history (UNCONDITIONAL)
        # ================================================================
        if hasattr(self, 'temporal_module') and self.temporal_module is not None:
            # ✅ NEW: Log detection status ALWAYS
            phase5_start = getattr(self, 'phase5_start_iter', None)
            logger.info(
                f"[Epoch {current_epoch}] Temporal module status:\n"
                f"  - Module exists: True\n"
                f"  - Phase 5 start iter: {phase5_start}\n"
                f"  - Current global step: {self.global_step}\n"
                f"  - Would clear: {phase5_start is not None and self.global_step >= phase5_start}"
            )

            # ✅ CHANGE: Clear UNCONDITIONALLY if epoch >= 20 (when Phase 5 could be active)
            # Original version only cleared if global_step >= phase5_start_iter
            # But step counter might be wrong, so use epoch as backup
            should_clear = False

            # Method 1: Check step (original)
            if phase5_start is not None and self.global_step >= phase5_start:
                should_clear = True
                logger.info(f"[Epoch {current_epoch}] Clearing temporal (step-based): step {self.global_step} >= {phase5_start}")

            # Method 2: Check epoch (backup - Phase 5 starts at step 1100 = epoch 20)
            elif current_epoch >= 20:
                should_clear = True
                logger.info(f"[Epoch {current_epoch}] Clearing temporal (epoch-based): epoch {current_epoch} >= 20")

            # ✅ EMERGENCY: Clear EVERY epoch if memory is high (> 15 GB)
            if torch.cuda.is_available():
                mem_allocated = torch.cuda.memory_allocated() / 1024**2
                if mem_allocated > 15000:  # 15 GB
                    should_clear = True
                    logger.warning(
                        f"[Epoch {current_epoch}] ⚠️  Forcing temporal clear due to high memory: "
                        f"{mem_allocated:.0f} MB"
                    )

            if should_clear:
                logger.info(f"[Epoch {current_epoch}] Phase 5 cleanup starting...")

                # Get stats before cleanup
                if hasattr(self.temporal_module, 'get_memory_stats'):
                    stats_before = self.temporal_module.get_memory_stats()
                    logger.info(
                        f"[Epoch {current_epoch}] Temporal state BEFORE cleanup:\n"
                        f"  - Sequences: {stats_before['num_sequences']}\n"
                        f"  - Total frames: {stats_before['total_frames']}\n"
                        f"  - Memory: {stats_before['estimated_memory_mb']:.2f} MB"
                    )

                # Clear history
                self.temporal_module.clear_epoch_history()

                # Verify cleanup worked
                if hasattr(self.temporal_module, 'get_memory_stats'):
                    stats_after = self.temporal_module.get_memory_stats()
                    if stats_after['num_sequences'] > 0:
                        logger.error(
                            f"[Epoch {current_epoch}] ❌ Cleanup FAILED! "
                            f"Still have {stats_after['num_sequences']} sequences"
                        )
                        # Force nuclear option
                        logger.info(f"[Epoch {current_epoch}] Forcing full history reset...")
                        self.temporal_module.reset_history(sequence_id=None)

                        # Triple-check
                        stats_final = self.temporal_module.get_memory_stats()
                        if stats_final['num_sequences'] > 0:
                            logger.critical(
                                f"[Epoch {current_epoch}] ❌❌ CRITICAL: Reset also failed! "
                                f"Still have {stats_final['num_sequences']} sequences. "
                                f"Memory leak confirmed."
                            )
                    else:
                        logger.info(
                            f"[Epoch {current_epoch}] ✅ Temporal cleanup successful\n"
                            f"  - Sequences: {stats_after['num_sequences']}\n"
                            f"  - Memory freed: {stats_before.get('estimated_memory_mb', 0) - stats_after.get('estimated_memory_mb', 0):.2f} MB"
                        )

                # Additional CUDA cleanup after temporal clearing
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    gc.collect()
                    torch.cuda.empty_cache()

                    mem_after = torch.cuda.memory_allocated() / 1024**2
                    logger.info(f"[Epoch {current_epoch}] GPU memory after temporal cleanup: {mem_after:.2f} MB")
            else:
                logger.info(
                    f"[Epoch {current_epoch}] Temporal cleanup skipped "
                    f"(not yet active or memory OK)"
                )

        # ================================================================
        # FIX 2: Clear Phase 4 contact caches if they exist
        # ================================================================
        if self.phase4_enabled and hasattr(self, 'contact_refiner') and self.contact_refiner is not None:
            if hasattr(self.contact_refiner, 'clear_cache'):
                logger.info(f"[Epoch {current_epoch}] Clearing contact refiner cache...")
                self.contact_refiner.clear_cache()

        # ================================================================
        # FIX 3: AGGRESSIVE full memory cleanup at epoch end
        # ================================================================
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2

            msg = (
                f"\n{'='*70}\n"
                f"[Epoch {current_epoch} END] Aggressive Memory Cleanup\n"
                f"  Allocated: {allocated:.2f} MB\n"
                f"  Reserved: {reserved:.2f} MB\n"
                f"{'='*70}\n"
            )
            print(msg)
            logger.info(msg)

        # ================================================================
        # ✅ FIX 4: Clear Model Buffers and Gradients (SAFE VERSION)
        # ================================================================
        try:
            # Clear all model buffer gradients
            for name, buffer in self.model.named_buffers():
                if buffer.grad is not None:
                    buffer.grad = None

            # Clear ALL parameter gradients (SAFE - always do this)
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad = None

            # ✅ STRENGTHENED: Clean unused optimizer states EVERY epoch
            # Changed from every 10 to every 1 for more aggressive cleanup
            if current_epoch % 1 == 0:  # Every epoch
                logger.info(f"[Epoch {current_epoch}] Cleaning unused optimizer states...")
                for optimizer in self.trainer.optimizers:
                    # Get current parameter IDs
                    current_params = set()
                    for param_group in optimizer.param_groups:
                        for param in param_group['params']:
                            current_params.add(id(param))

                    # Find and remove states for parameters not in model
                    state_keys_to_remove = []
                    for param_key in list(optimizer.state.keys()):
                        if id(param_key) not in current_params:
                            state_keys_to_remove.append(param_key)

                    # Remove unused states
                    for key in state_keys_to_remove:
                        del optimizer.state[key]

                    if state_keys_to_remove:
                        logger.info(f"[Epoch {current_epoch}] ✅ Removed {len(state_keys_to_remove)} unused optimizer states")
                    else:
                        logger.debug(f"[Epoch {current_epoch}] No unused optimizer states found")

        except Exception as e:
            logger.warning(f"[Epoch {current_epoch}] Could not clear buffers/optimizer: {e}")

        # ================================================================
        # ✅ FIX 5: Clear PyTorch Lightning Callback States
        # ================================================================
        try:
            if hasattr(self.trainer, 'callback_metrics'):
                metrics_to_keep = {}
                for key, value in self.trainer.callback_metrics.items():
                    if isinstance(value, torch.Tensor):
                        metrics_to_keep[key] = value.detach().cpu().item()
                    else:
                        metrics_to_keep[key] = value

                self.trainer.callback_metrics = metrics_to_keep
                logger.debug(f"[Epoch {current_epoch}] Cleaned callback metrics")

        except Exception as e:
            logger.warning(f"[Epoch {current_epoch}] Could not clean callback metrics: {e}")

        # ================================================================
        # ✅ FIX 6: Force Synchronization Before Final Cleanup
        # ================================================================
        if torch.cuda.is_available():
            torch.cuda.synchronize()

            # Triple cleanup
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()

            allocated_final = torch.cuda.memory_allocated() / 1024**2
            reserved_final = torch.cuda.memory_reserved() / 1024**2

            logger.info(
                f"[Epoch {current_epoch} END] Final memory state:\n"
                f"  Allocated: {allocated_final:.2f} MB\n"
                f"  Reserved: {reserved_final:.2f} MB"
            )

        # ================================================================
        # ✅ NEW FIX 7: Clear DataLoader Worker Memory
        # ================================================================
        try:
            if hasattr(self.trainer, 'train_dataloader'):
                train_dl = self.trainer.train_dataloader
                if train_dl is not None and hasattr(train_dl, 'loaders'):
                    logger.info(f"[Epoch {current_epoch}] Clearing DataLoader worker caches...")
                    # Force DataLoader to restart workers
                    # This prevents worker memory accumulation
                    if hasattr(train_dl, '_iterator'):
                        train_dl._iterator = None

                    # Clear any cached batches
                    if hasattr(train_dl, '_cache'):
                        train_dl._cache = {}

                    logger.debug(f"[Epoch {current_epoch}] DataLoader worker memory cleared")

        except Exception as e:
            logger.warning(f"[Epoch {current_epoch}] Could not clear DataLoader memory: {e}")
        # ================================================================
        # ✅ NEW FIX 8: Clear Model-Specific Caches
        # ================================================================
        try:
            # Clear any internal caches in the model
            if hasattr(self.model, 'clear_cache'):
                logger.info(f"[Epoch {current_epoch}] Clearing model internal caches...")
                self.model.clear_cache()

            # Clear node-specific caches
            if hasattr(self.model, 'nodes'):
                for node_name, node in self.model.nodes.items():
                    if hasattr(node, 'clear_cache'):
                        node.clear_cache()

                    # Clear implicit network caches
                    if hasattr(node, 'implicit_network') and hasattr(node.implicit_network, 'clear_cache'):
                        node.implicit_network.clear_cache()

            logger.debug(f"[Epoch {current_epoch}] Model caches cleared")

        except Exception as e:
            logger.warning(f"[Epoch {current_epoch}] Could not clear model caches: {e}")
        # ================================================================
        # ✅ CRITICAL: Reset metrics to prevent accumulation
        # ================================================================
        if hasattr(self, 'metrics') and self.metrics is not None:
            try:
                self.metrics.reset()
                logger.debug(f"[Epoch {current_epoch}] Metrics state reset")
            except Exception as e:
                logger.warning(f"[Epoch {current_epoch}] Could not reset metrics: {e}")
        # ================================================================
        # ✅ DIAGNOSTIC: Find growing modules
        # ================================================================
        if current_epoch % 5 == 0:
            logger.info(f"\n{'='*70}")
            logger.info(f"[Epoch {current_epoch}] Module Memory Diagnostic")
            logger.info(f"{'='*70}")

            for name, module in self.model.named_modules():
                try:
                    # Count parameters
                    num_params = sum(p.numel() for p in module.parameters(recurse=False))

                    # Count buffers
                    num_buffers = len(list(module.buffers(recurse=False)))

                    # Check for growing attributes
                    for attr_name in dir(module):
                        if attr_name.startswith('_'):
                            continue
                        try:
                            attr = getattr(module, attr_name)
                            if isinstance(attr, list) and len(attr) > 100:
                                logger.warning(
                                    f"  {name}.{attr_name}: LIST with {len(attr)} items - POTENTIAL LEAK"
                                )
                            elif isinstance(attr, dict) and len(attr) > 100:
                                logger.warning(
                                    f"  {name}.{attr_name}: DICT with {len(attr)} items - POTENTIAL LEAK"
                                )
                        except:
                            pass
                except:
                    pass

            logger.info(f"{'='*70}\n")

        return super().training_epoch_end(outputs)

    def on_train_end(self):
        """
        Called when training ends.

        Performs final cleanup of temporal history and caches.
        """
        logger.info("[Training End] Final memory cleanup...")

        # ================================================================
        # Clear Phase 5 temporal history completely
        # ================================================================
        if self.phase5_enabled and hasattr(self, 'temporal_module') and self.temporal_module is not None:
            logger.info("[Training End] Clearing all temporal history...")
            self.temporal_module.reset_history(sequence_id=None)  # Clear all sequences

            # Report statistics
            logger.info(
                f"[Training End] Temporal module statistics:\n"
                f"  - Window size: {self.temporal_module.window_size}\n"
                f"  - Max sequences: {self.temporal_module.max_sequences}\n"
                f"  - Final sequences: {len(self.temporal_module.pose_history)}"
            )

        # ================================================================
        # Clear Phase 4 contact caches
        # ================================================================
        if self.phase4_enabled and hasattr(self, 'contact_refiner') and self.contact_refiner is not None:
            if hasattr(self.contact_refiner, 'clear_cache'):
                logger.info("[Training End] Clearing contact refiner cache...")
                self.contact_refiner.clear_cache()

        # ================================================================
        # Final CUDA memory report
        # ================================================================
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / 1024 ** 2
            mem_reserved = torch.cuda.memory_reserved() / 1024 ** 2

            logger.info(
                f"[Training End] Final GPU memory:\n"
                f"  - Allocated: {mem_allocated:.2f} MB\n"
                f"  - Reserved: {mem_reserved:.2f} MB"
            )

            # Clear cache one final time
            torch.cuda.empty_cache()

        logger.info("[Training End] Cleanup complete")

    def meshing_cano(self, current_step):
        """Extract canonical meshes for all nodes without gradient tracking.

        This is a visualization/evaluation operation that should not build
        computation graphs. All mesh extraction is wrapped in torch.no_grad()
        to prevent memory leaks from accumulated gradient graphs.

        Args:
            current_step (int): Current training step for logging

        Returns:
            dict: Mesh dictionary {node_id: trimesh object}
        """
        # ================================================================
        # FIX 3: Wrap entire meshing operation in torch.no_grad()
        # ================================================================
        # CRITICAL: Meshing is an evaluation operation that:
        # 1. Queries SDF network thousands of times
        # 2. Runs Marching Cubes algorithm
        # 3. Should NEVER build gradient graphs
        #
        # Without no_grad(), each mesh extraction accumulates ~10-50 MB of
        # gradient graphs that are never freed, leading to OOM.
        # ================================================================
        with torch.no_grad():
            # Set model to eval mode (important for BatchNorm, Dropout)
            self.model.eval()

            mesh_dict = {}

            for node in self.model.nodes.values():
                try:
                    # Call node's meshing method (also wrapped in no_grad internally)
                    mesh_c = node.meshing_cano()
                    # For object node, reject meshes with no faces so we hit the fallback
                    if node.node_id == "object":
                        if mesh_c is None or not hasattr(mesh_c, "faces") or mesh_c.faces is None or mesh_c.faces.shape[
                            0] == 0:
                            raise ValueError(f"[meshing_cano] {node.node_id} returned mesh with no faces")
                    # ================================================================
                    # FIX 3: Verify mesh has no gradient tracking
                    # ================================================================
                    # If mesh vertices are torch tensors, ensure they're detached
                    if hasattr(mesh_c, 'vertices') and isinstance(mesh_c.vertices, torch.Tensor):
                        if mesh_c.vertices.requires_grad:
                            logger.warning(
                                f"[meshing_cano] {node.node_id} mesh has requires_grad=True. "
                                f"Detaching to prevent memory leak."
                            )
                            # Note: trimesh vertices are usually numpy arrays, but be safe
                            mesh_c.vertices = mesh_c.vertices.detach().cpu().numpy()

                    # Export mesh
                    out_p = op.join(
                        self.args.log_dir,
                        "mesh_cano",
                        f"mesh_cano_{node.node_id}_step_{current_step}.obj",
                    )
                    os.makedirs(op.dirname(out_p), exist_ok=True)
                    mesh_c.export(out_p)
                    logger.info(f"Exported canonical to {out_p}")

                    mesh_dict[f"{node.node_id}_cano"] = mesh_c

                except Exception as e:
                    logger.error(f"Failed to mesh out {node.node_id}: {e}")

                    # Fallback: try vertex-based canonical mesh for object node
                    if node.node_id == "object":
                        try:
                            if hasattr(node, 'server') and hasattr(node.server, 'object_model'):
                                obj_model = node.server.object_model

                                import trimesh
                                verts = None
                                if hasattr(obj_model, 'v3d_cano'):
                                    verts = obj_model.v3d_cano.detach().cpu().numpy()

                                # 1) Prefer stored faces if they exist and are non-empty
                                if verts is not None and hasattr(obj_model, 'f3d_cano'):
                                    faces = obj_model.f3d_cano
                                    if isinstance(faces, torch.Tensor):
                                        faces = faces.detach().cpu().numpy()

                                    if faces is not None and faces.shape[0] > 0:
                                        mesh_c = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                                        mesh_dict[f"{node.node_id}_cano"] = mesh_c
                                        logger.warning(
                                            f"[meshing_cano] Using v3d_cano/f3d_cano fallback for {node.node_id} "
                                            f"({verts.shape[0]} verts, {faces.shape[0]} faces)"
                                        )
                                        return  # or `continue` outer loop

                                # 2) No usable faces: build a convex hull from verts (gives valid faces)
                                if verts is not None:
                                    hull = trimesh.Trimesh(vertices=verts, process=True)
                                    if hull.faces.shape[0] == 0:
                                        hull = hull.convex_hull  # ensure faces
                                    mesh_c = hull
                                    mesh_dict[f"{node.node_id}_cano"] = mesh_c
                                    logger.warning(
                                        f"[meshing_cano] Using convex-hull fallback for {node.node_id} "
                                        f"({verts.shape[0]} verts, {mesh_c.faces.shape[0]} faces)"
                                    )
                                else:
                                    # 3) Absolute last resort: minimal dummy triangle so evaluation doesn't crash
                                    dummy_verts = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]], dtype=np.float32)
                                    dummy_faces = np.array([[0, 1, 2]], dtype=np.int64)
                                    mesh_c = trimesh.Trimesh(vertices=dummy_verts, faces=dummy_faces, process=False)
                                    mesh_dict[f"{node.node_id}_cano"] = mesh_c
                                    logger.error(
                                        f"[meshing_cano] All fallbacks failed, using dummy mesh for {node.node_id}"
                                    )

                        except Exception as e2:
                            logger.error(f"[meshing_cano] Fallback for object failed: {e2}")

            return mesh_dict

    def inference_step(self, batch, *args, **kwargs):
        batch = xdict(batch).to("cuda")
        self.model.eval()
        batch = xdict(batch)
        batch["current_epoch"] = self.current_epoch
        batch["global_step"] = self.global_step

        # ================================================================
        # CRITICAL FIX: Get parameters from nodes and add to batch
        # ================================================================
        # Try to find nodes attribute in different locations
        nodes = None
        if hasattr(self, 'nodes'):
            nodes = self.nodes
        elif hasattr(self.model, 'nodes'):
            nodes = self.model.nodes
        elif hasattr(self, '_modules') and 'model' in self._modules:
            if hasattr(self._modules['model'], 'nodes'):
                nodes = self._modules['model'].nodes

        if nodes is not None:
            logger.info(f"[HOLD.inference_step] Found nodes at correct location, fetching checkpoint params")
            for node in nodes.values():
                params = node.params(batch["idx"])  # Returns dict with all parameters

                # params already contains full_pose, betas, etc.
                # Use dict methods to avoid xdict's duplicate key check
                for key, value in params.items():
                    if key in batch:
                        # Key already exists, use underlying dict to overwrite
                        dict.__setitem__(batch, key, value)
                    else:
                        # New key, can use normal assignment
                        batch[key] = value
            logger.info(f"[HOLD.inference_step] Successfully loaded checkpoint params into batch")
        else:
            logger.warning("[HOLD.inference_step] ⚠️ Could not find nodes attribute!")
            logger.warning("  Checking if checkpoint preservation is enabled...")
            logger.warning("  If not, will use batch GT instead of checkpoint params")
        # ================================================================

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
        logger.info(f"Exported normal to {out_p}")
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


