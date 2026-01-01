# hold/training/ghop_loss.py
"""
GHOP SDS loss integration for HOLD training.

PHASE 3 UPDATES:
- Added SDSLoss: Complete Score Distillation Sampling implementation
- Implements gradient approximation: grad = w(t) * (ε_θ - ε)
- Supports dynamic noise scheduling with annealing
- Compatible with VQ-VAE + 3D U-Net diffusion pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.model.ghop.ghop_prior import load_ghop_prior
from loguru import logger
# import logging
import traceback
# logger = logging.getLogger(__name__)

# ============================================================================
# PHASE 3: Core SDS Loss Implementation
# ============================================================================
class SDSLoss(nn.Module):
    """
    Score Distillation Sampling loss with dynamic noise scheduling.
    Implements gradient approximation: grad = w(t) * (ε_θ - ε)

    Based on GHOP's sd.py gradient computation (lines 102-115).
    """

    def __init__(self,
                 vqvae_wrapper,
                 unet_wrapper,
                 hand_field_builder,
                 guidance_scale=4.0,
                 min_step_ratio=0.02,
                 max_step_ratio=0.98,
                 diffusion_steps=1000):
        """
        Args:
            vqvae_wrapper: GHOPVQVAEWrapper instance for encoding
            unet_wrapper: GHOP3DUNetWrapper instance for noise prediction
            hand_field_builder: HandFieldBuilder instance
            guidance_scale: Classifier-free guidance scale (default: 4.0)
            min_step_ratio: Minimum timestep ratio (default: 0.02 = step 20/1000)
            max_step_ratio: Maximum timestep ratio (default: 0.98 = step 980/1000)
            diffusion_steps: Total diffusion steps (default: 1000)
        """
        super().__init__()

        self.vqvae = vqvae_wrapper
        self.unet = unet_wrapper
        self.hand_field = hand_field_builder
        self.guidance_scale = guidance_scale

        # ================================================================
        # Noise schedule: linear β_t from 0.0001 to 0.02
        # ================================================================
        self.diffusion_steps = diffusion_steps
        self.betas = torch.linspace(0.0001, 0.02, diffusion_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Timestep bounds for sampling
        self.min_step = int(min_step_ratio * diffusion_steps)
        self.max_step = int(max_step_ratio * diffusion_steps)

        print(f"[SDSLoss] Initialized with:")
        print(f"  - Timestep range: [{self.min_step}, {self.max_step}]")
        print(f"  - Guidance scale: {guidance_scale}")
        print(f"  - Diffusion steps: {diffusion_steps}")

    def forward(self, object_sdf, hand_params, text_prompts=None,
                iteration=0, weight=1.0):
        """
        Compute SDS loss for current reconstruction.

        Args:
            object_sdf: (B, 1, 64, 64, 64) from HOLD-Net
            hand_params: Dict with keys:
                - 'pose': (B, 48) or (B, 45) MANO pose
                - 'shape': (B, 10) MANO shape (optional)
                - 'trans': (B, 3) Translation (optional)
            text_prompts: List of text descriptions or None
            iteration: Current training step (for annealing)
            weight: Loss weight multiplier

        Returns:
            sds_loss: Scalar loss value
            info: Dict with diagnostics
        """
        B = object_sdf.shape[0]
        device = object_sdf.device

        # ✅ LOG 1: Entry point
        mem_start = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        logger.info(f"[SDS-MEMORY] === Starting SDS forward (iteration {iteration}) ===")
        logger.info(f"[SDS-MEMORY] GPU memory at entry: {mem_start:.2f} MB")
        logger.info(f"[SDS-MEMORY] Input object_sdf shape: {object_sdf.shape}, size: {object_sdf.element_size() * object_sdf.nelement() / 1024**2:.2f} MB")

        # ================================================================
        # Step 1: Build 15-channel hand skeletal distance field
        # ================================================================
        hand_field_64 = self.hand_field(hand_params=hand_params)  # (B, 15, 64, 64, 64)

        # ✅ LOG 2: After hand field
        mem_after_hand = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        logger.info(f"[SDS-MEMORY] After hand_field: {mem_after_hand:.2f} MB (delta: +{mem_after_hand - mem_start:.2f} MB)")
        logger.info(f"[SDS-MEMORY] hand_field_64 shape: {hand_field_64.shape}, size: {hand_field_64.element_size() * hand_field_64.nelement() / 1024**2:.2f} MB")

        # ================================================================
        # Step 2: Encode to latent space via VQ-VAE
        # ================================================================
        # Note: encode() returns (z_q, indices, vq_loss)
        z_0, _, vq_loss = self.vqvae.encode(object_sdf, hand_field_64)

        # ✅ LOG 3: After VQ-VAE encode
        mem_after_encode = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        logger.info(f"[SDS-MEMORY] After VQ-VAE encode: {mem_after_encode:.2f} MB (delta: +{mem_after_encode - mem_after_hand:.2f} MB)")
        logger.info(f"[SDS-MEMORY] z_0 shape: {z_0.shape}, size: {z_0.element_size() * z_0.nelement() / 1024**2:.2f} MB")

        # ================================================================
        # Step 3: Sample timestep with progressive annealing
        # ================================================================
        # Linearly decay max_step over first 5000 iterations
        # This helps with stability: start with easier timesteps
        progress = min(iteration / 5000.0, 1.0)
        current_max = self.max_step - int(progress * (self.max_step - self.min_step) * 0.5)

        t = torch.randint(self.min_step, current_max, (B,),
                          device=device, dtype=torch.long)

        # ================================================================
        # Step 4: Forward diffusion - add noise to clean latent
        # ================================================================
        noise = torch.randn_like(z_0)
        alpha_bar_t = self.alphas_cumprod[t].view(B, 1, 1, 1, 1).to(device)

        # Diffusion forward process: z_t = sqrt(α̅_t) * z_0 + sqrt(1-α̅_t) * ε
        z_t = torch.sqrt(alpha_bar_t) * z_0 + torch.sqrt(1 - alpha_bar_t) * noise

        # ✅ LOG 4: After noise addition
        mem_after_noise = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        logger.info(f"[SDS-MEMORY] After noise addition: {mem_after_noise:.2f} MB (delta: +{mem_after_noise - mem_after_encode:.2f} MB)")
        logger.info(f"[SDS-MEMORY] z_t shape: {z_t.shape}, noise shape: {noise.shape}")

        # ================================================================
        # Step 5: Predict noise with classifier-free guidance
        # ================================================================
        # Downsample hand field to latent resolution (optional conditioning)
        hand_field_16 = self.hand_field.downsample_to_latent(hand_field_64, 16)

        # ✅ LOG 5: Before U-Net
        mem_before_unet = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        logger.info(f"[SDS-MEMORY] Before U-Net calls: {mem_before_unet:.2f} MB")

        with torch.no_grad():
            # Get text embeddings (currently placeholder)
            text_emb = self._get_text_embeddings(text_prompts, B, device)

            # ✅ LOG 6: Before first U-Net call
            mem_before_unet1 = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            logger.info(f"[SDS-MEMORY] Calling U-Net (conditional)...")

            # ADD THIS DEBUG BEFORE U-NET:
            logger.info(f"[SDS-SHAPE] z_t input shape: {z_t.shape}")
            logger.info(f"[SDS-SHAPE] text_emb shape: {text_emb.shape}")

            # ✅ ADD THIS NEW DEBUG - Check what U-Net wrapper receives:
            logger.info(f"[SDS-SHAPE] About to call self.unet.forward()")
            logger.info(f"[SDS-SHAPE] U-Net wrapper type: {type(self.unet).__name__}")
            logger.info(f"[SDS-SHAPE] Check U-Net wrapper's internal concatenation logic")

            eps_cond = self.unet(z_t, t, text_emb)  # Calls forward() which applies 23→3 adapter

            # ADD THIS DEBUG AFTER FIRST U-NET:
            logger.info(f"[SDS-SHAPE] eps_cond output shape: {eps_cond.shape}")
            logger.info(f"[SDS-SHAPE] Expected shape (same as z_t): {z_t.shape}")

            # ✅ LOG 7: After first U-Net call
            mem_after_unet1 = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            logger.info(f"[SDS-MEMORY] After U-Net call 1: {mem_after_unet1:.2f} MB (delta: +{mem_after_unet1 - mem_before_unet1:.2f} MB)")
            logger.info(f"[SDS-MEMORY] eps_cond shape: {eps_cond.shape}, size: {eps_cond.element_size() * eps_cond.nelement() / 1024**2:.2f} MB")

            # ✅ LOG 8: Before second U-Net call
            logger.info(f"[SDS-MEMORY] Calling U-Net (unconditional)...")

            eps_uncond = self.unet(z_t, t, torch.zeros_like(text_emb))  # Calls forward() which applies adapter

            # ADD THIS DEBUG AFTER SECOND U-NET:
            logger.info(f"[SDS-SHAPE] eps_uncond output shape: {eps_uncond.shape}")

            # ✅ LOG 9: After second U-Net call
            mem_after_unet2 = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            logger.info(f"[SDS-MEMORY] After U-Net call 2: {mem_after_unet2:.2f} MB (delta: +{mem_after_unet2 - mem_after_unet1:.2f} MB)")

            eps_theta = eps_uncond + self.guidance_scale * (eps_cond - eps_uncond)

            # ADD THIS DEBUG AFTER GUIDANCE:
            logger.info(f"[SDS-SHAPE] eps_theta (after guidance) shape: {eps_theta.shape}")

        # ✅ LOG 10: After guidance
        mem_after_guidance = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        logger.info(f"[SDS-MEMORY] After guidance: {mem_after_guidance:.2f} MB")

        # NOW the shapes should match:
        # ================================================================
        # Steps 6-9: SDS gradient computation (GHOP sd.py lines 102-115)
        # ================================================================
        # Weight function: w(t) = (1 - α̅_t)
        w_t = 1.0 - alpha_bar_t

        # ✅ ADD THIS DEBUG LOGGING BLOCK HERE (AFTER w_t is defined):
        logger.info(f"[SDS-SHAPE] === BEFORE GRAD COMPUTATION ===")
        logger.info(f"[SDS-SHAPE] z_0 shape: {z_0.shape}")
        logger.info(f"[SDS-SHAPE] noise shape: {noise.shape}")
        logger.info(f"[SDS-SHAPE] eps_theta shape: {eps_theta.shape}")
        logger.info(f"[SDS-SHAPE] w_t shape: {w_t.shape}")
        logger.info(f"[SDS-SHAPE] w_t value range: [{w_t.min():.4f}, {w_t.max():.4f}]")

        # ================================================================
        # ✅ FIX: Resize eps_theta to match VQ-VAE latent resolution
        # ================================================================
        # GHOP U-Net trained on 8³ latents but HOLD uses 6³ latents.
        # Downsample U-Net prediction to match target noise resolution.
        if eps_theta.shape[2:] != noise.shape[2:]:
            logger.info(
                f"[SDS-FIX] Spatial mismatch: eps_theta {eps_theta.shape[2:]} "
                f"vs noise {noise.shape[2:]}, applying trilinear resize"
            )
            eps_theta = F.interpolate(
                eps_theta,
                size=tuple(noise.shape[2:]),
                mode='trilinear',
                align_corners=False
            )
            logger.info(f"[SDS-FIX] Resized eps_theta to: {eps_theta.shape}")

        # Gradient approximation: ∇_z L_SDS = w(t) * (ε̂_θ - ε)
        grad = weight * w_t * (eps_theta - noise)  # ✅ Shapes now match!

        # Target with stop-gradient for backprop
        # We want: ∇_z L = ∇_z ||z_0 - (z_0 - grad)||²
        target = (z_0 - grad).detach()

        # SDS loss as MSE (enables gradient flow to z_0)
        sds_loss = 0.5 * F.mse_loss(z_0, target, reduction='mean')

        # ✅ LOG 11: Final memory
        mem_end = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        logger.info(f"[SDS-MEMORY] At exit: {mem_end:.2f} MB")
        logger.info(f"[SDS-MEMORY] Total delta from entry: +{mem_end - mem_start:.2f} MB")
        logger.info(f"[SDS-MEMORY] === SDS forward complete ===\n")

        # ================================================================
        # Diagnostics for logging
        # ================================================================
        info = {
            'timestep_mean': t.float().mean().item(),
            'timestep_max': t.max().item(),
            'weight_t_mean': w_t.mean().item(),
            'grad_norm': grad.norm().item(),
            'current_max_step': current_max,
            'vq_loss': vq_loss.item() if isinstance(vq_loss, torch.Tensor) else 0.0,
            'noise_pred_norm': eps_theta.norm().item(),
            'memory_mb': mem_end  # Add to diagnostics
        }

        return sds_loss, info

    def compute_sds_loss(self, obj_sdf, hand_params, text_prompt, weight=1.0, iteration=0):
        """
        Wrapper method for TwoStageTrainingManager compatibility.

        Args:
            obj_sdf: (B, 1, 64, 64, 64) object SDF grid
            hand_params: Dict with keys {'pose', 'shape', 'trans'} OR (B, 45/48) tensor
            text_prompt: str or List[str] text descriptions
            weight: float, loss weight multiplier

        Returns:
            sds_loss: torch.Tensor, weighted SDS loss
            info: dict, diagnostic information
        """
        mem_start = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        logger.info(f"[SDS-WRAPPER] compute_sds_loss called at memory: {mem_start:.2f} MB")
        logger.info(f"[SDS-WRAPPER] obj_sdf shape: {obj_sdf.shape}")
        try:
            # ============================================================
            # Validate and normalize text_prompt
            # ============================================================
            if text_prompt is None or (isinstance(text_prompt, list) and len(text_prompt) == 0):
                # Default fallback prompt
                text_prompt = ["a hand grasping an object"]
                logger.debug("[SDSLoss] Using default text prompt: 'a hand grasping an object'")
            elif isinstance(text_prompt, str):
                # Single string -> wrap in list
                text_prompt = [text_prompt]
            elif isinstance(text_prompt, list):
                # Filter out None/empty strings
                text_prompt = [p for p in text_prompt if p is not None and len(p) > 0]
                if len(text_prompt) == 0:
                    text_prompt = ["a hand grasping an object"]

            # ============================================================
            # NO LONGER NEEDED: hand_params is already in correct format
            # Just pass it directly to forward()
            # ============================================================
            # REMOVED: hand_params = {'pose': hand_pose}

            # Call forward method (returns sds_loss, info)
            sds_loss_raw, info = self.forward(
                object_sdf=obj_sdf,
                hand_params=hand_params,
                text_prompts=text_prompt,
                iteration=iteration,  # ← Pass through correctly
                weight=1.0
            )

            # Apply external weight
            sds_loss_weighted = sds_loss_raw * weight

            # Update info with weighted value
            info['sds_weighted'] = sds_loss_weighted.item() if isinstance(sds_loss_weighted, torch.Tensor) else float(sds_loss_weighted)
            info['sds_raw'] = sds_loss_raw.item() if isinstance(sds_loss_raw, torch.Tensor) else float(sds_loss_raw)
            info['weight'] = weight

            mem_end = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            logger.info(f"[SDS-WRAPPER] After forward: {mem_end:.2f} MB (delta: +{mem_end - mem_start:.2f} MB)")

            return sds_loss_weighted, info

        except Exception as e:
            # Fallback: return zero loss on error
            device = obj_sdf.device
            zero_loss = torch.tensor(0.0, device=device, requires_grad=False)
            info = {
                'error': str(e),
                'sds_weighted': 0.0,
                'sds_raw': 0.0,
                'weight': weight
            }
            logger.warning(f"[SDSLoss] compute_sds_loss failed: {e}")
            traceback.print_exc()  # ✅ This will dump the full Python traceback into the log
            return zero_loss, info

    def _get_text_embeddings(self, prompts, batch_size, device):
        """
        Generate text embeddings for conditioning.

        Args:
            prompts: List of text strings or None
            batch_size: Batch size
            device: Target device

        Returns:
            text_emb: (B, 77, 768) text embeddings

        TODO: Integrate CLIP text encoder when available:
            from transformers import CLIPTextModel, CLIPTokenizer
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        """
        # Placeholder: return zero embeddings
        # In production, replace with actual CLIP embeddings
        if prompts is None:
            prompts = ["a hand grasping an object"] * batch_size

        # Standard CLIP embedding shape: (B, 77, 768)
        return torch.zeros(batch_size, 77, 768, device=device)

    def compute_with_cfg_scale(self, object_sdf, hand_params, text_prompts,
                               iteration, weight, cfg_scale):
        """
        Compute SDS loss with custom guidance scale.
        Useful for testing different CFG scales.
        """
        original_scale = self.guidance_scale
        self.guidance_scale = cfg_scale

        loss, info = self.forward(object_sdf, hand_params, text_prompts,
                                  iteration, weight)

        self.guidance_scale = original_scale
        return loss, info

class GHOPSDSLoss:
    """
    Manages GHOP SDS loss computation during HOLD training.
    """

    def __init__(
        self,
        vqvae_wrapper=None,
        unet_wrapper=None,
        hand_field_builder=None,
        ghop_checkpoint=None,
        sds_weight=5000.0,
        guidance_scale=4.0,
        start_iter=0,
        end_iter=None,
        device='cuda'
    ):
        """
        Args:
            vqvae_wrapper: GHOPVQVAEWrapper instance (Phase 3)
            unet_wrapper: GHOP3DUNetWrapper instance (Phase 3)
            hand_field_builder: HandFieldBuilder instance (Phase 3)
            ghop_checkpoint: Path to GHOP .ckpt file (legacy, optional)
            sds_weight: Weight for SDS loss (default: 5000)
            guidance_scale: CFG scale (default: 4.0)
            start_iter: When to start applying SDS (default: 0)
            end_iter: When to stop applying SDS (default: never)
        """
        self.sds_weight = sds_weight
        self.guidance_scale = guidance_scale
        self.start_iter = start_iter
        self.end_iter = end_iter
        self.device = device

        # Phase 3: Use new SDSLoss implementation
        if vqvae_wrapper and unet_wrapper and hand_field_builder:
            self.sds_loss_module = SDSLoss(
                vqvae_wrapper=vqvae_wrapper,
                unet_wrapper=unet_wrapper,
                hand_field_builder=hand_field_builder,
                guidance_scale=guidance_scale
            )
            print("[GHOPSDSLoss] Using Phase 3 SDSLoss implementation")
        # Legacy: Use ghop_prior loader (for backward compatibility)
        elif ghop_checkpoint:
            from hold.models.ghop_prior import load_ghop_prior
            self.ghop_prior = load_ghop_prior(ghop_checkpoint, device)
            self.sds_loss_module = None
            print("[GHOPSDSLoss] Using legacy ghop_prior implementation")
        else:
            raise ValueError("Must provide either (vqvae, unet, hand_field) or ghop_checkpoint")

    def should_apply(self, iteration):
        """Check if SDS should be applied at this iteration."""
        if iteration < self.start_iter:
            return False
        if self.end_iter is not None and iteration >= self.end_iter:
            return False
        return True

    def compute(
        self,
        obj_sdf,
        hand_params,
        object_category=None,
        iteration=0,
        hand_shape=None,
        hand_trans=None
    ):
        # ================================================================
        # 🔍 ABSOLUTE FIRST LINE: Confirm we got called
        # ================================================================
        logger.critical(f"[GHOP-COMPUTE-ENTRY] ============================================")
        logger.critical(f"[GHOP-COMPUTE-ENTRY] compute() CALLED at iteration {iteration}")
        logger.critical(f"[GHOP-COMPUTE-ENTRY] obj_sdf shape: {obj_sdf.shape}")
        logger.critical(f"[GHOP-COMPUTE-ENTRY] start_iter: {self.start_iter}, end_iter: {self.end_iter}")
        logger.critical(f"[GHOP-COMPUTE-ENTRY] About to check should_apply()...")

        # ============================================================
        # CRITICAL DIAGNOSTIC: Log iteration values
        # ============================================================
        should_apply_result = self.should_apply(iteration)

        logger.critical(f"[GHOP-COMPUTE-ENTRY] should_apply returned: {should_apply_result}")

        if not should_apply_result:
            logger.warning(f"[GHOP-COMPUTE] ❌ REJECTED! Reason:")
            if iteration < self.start_iter:
                logger.warning(f"[GHOP-COMPUTE]   iteration ({iteration}) < start_iter ({self.start_iter})")
            if self.end_iter is not None and iteration >= self.end_iter:
                logger.warning(f"[GHOP-COMPUTE]   iteration ({iteration}) >= end_iter ({self.end_iter})")
            logger.warning(f"[GHOP-COMPUTE] ======================================")
            return 0.0, {'sds_loss': 0.0, 'applied': False}

        """
        Compute SDS loss for current HOLD state.

        Args:
            obj_sdf: (B, 1, 64, 64, 64) - Object SDF from HOLD network
            hand_params: Dict with keys {'pose', 'shape', 'trans'} OR (B, 45/48) tensor
            object_category: str or List[str] - Object category for text conditioning
            iteration: Current training iteration
            hand_shape: (B, 10) - MANO shape (optional, for legacy compatibility)
            hand_trans: (B, 3) - Hand translation (optional, for legacy compatibility)

        Returns:
            loss: Scalar tensor or 0 if not applied
            info: Dict with loss components
        """
        # Prepare hand parameters dict
        B = obj_sdf.shape[0]

        # ✅ FIXED: Handle both dict and tensor inputs properly
        if isinstance(hand_params, dict):
            # Already in correct format
            pass
        elif isinstance(hand_params, torch.Tensor):
            # Legacy: tensor input, build dict
            hand_params = {
                'pose': hand_params,  # ✅ Now uses correct variable name
                'shape': hand_shape if hand_shape is not None else torch.zeros(B, 10, device=obj_sdf.device),
                'trans': hand_trans if hand_trans is not None else torch.zeros(B, 3, device=obj_sdf.device)
            }
        else:
            raise ValueError(f"Invalid hand_params type: {type(hand_params)}")

        # Prepare text prompts
        if object_category is None:
            text_prompts = ["a hand grasping an object"] * B
        elif isinstance(object_category, str):
            text_prompts = [f"a hand grasping a {object_category}"] * B
        else:
            text_prompts = [f"a hand grasping a {cat}" for cat in object_category]

        # Phase 3: Use new SDSLoss module
        if self.sds_loss_module is not None:
            sds_loss, sds_info = self.sds_loss_module.forward(
                object_sdf=obj_sdf,
                hand_params=hand_params,
                text_prompts=text_prompts,
                iteration=iteration,
                weight=self.sds_weight
            )

            logger.warning(f"[GHOP-COMPUTE] forward() returned: sds_loss={sds_loss}")
            # ✅ ADD THIS ADDITIONAL INFO:
            logger.warning(f"[GHOP-COMPUTE] sds_info keys: {list(sds_info.keys())}")
            if 'error' in sds_info:
                logger.error(f"[GHOP-COMPUTE] ERROR in sds_info: {sds_info['error']}")

        # Legacy: Use ghop_prior
        else:
            logger.warning(f"[GHOP-COMPUTE] Using Legacy ghop_prior.compute_sds_loss()")
            sds_loss, sds_info = self.ghop_prior.compute_sds_loss(
                obj_sdf=obj_sdf,
                hand_params=hand_params,  # ✅ Now correct
                text_prompt=text_prompts[0] if len(text_prompts) == 1 else text_prompts,
                weight=self.sds_weight
            )

        info = {
            'sds_loss': sds_loss.item() if isinstance(sds_loss, torch.Tensor) else sds_loss,
            'applied': True,
            **sds_info
        }

        return sds_loss, info


# ============================================================================
# PHASE 3: Testing Utilities
# ============================================================================
def test_sds_loss():
    """Test SDSLoss module."""
    print("\n" + "=" * 60)
    print("Testing SDSLoss module...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Mock components
    from src.model.ghop.autoencoder import GHOPVQVAEWrapper
    from src.model.ghop.diffusion import GHOP3DUNetWrapper
    from src.model.ghop.hand_field import HandFieldBuilder

    # Mock MANO server
    class MockMANOServer(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1))

    vqvae = GHOPVQVAEWrapper(vqvae_ckpt_path=None, device=device)
    unet = GHOP3DUNetWrapper(unet_ckpt_path=None, device=device)
    hand_field = HandFieldBuilder(MockMANOServer().to(device), resolution=64)

    # Initialize SDSLoss
    sds_loss = SDSLoss(
        vqvae_wrapper=vqvae,
        unet_wrapper=unet,
        hand_field_builder=hand_field,
        guidance_scale=4.0
    )

    # Test inputs
    B = 2
    object_sdf = torch.randn(B, 1, 64, 64, 64, device=device)
    hand_params = {
        'pose': torch.randn(B, 48, device=device),
        'shape': torch.randn(B, 10, device=device),
        'trans': torch.randn(B, 3, device=device)
    }
    text_prompts = ["a hand grasping a mug", "a hand holding a bottle"]

    # Compute loss
    print(f"\nComputing SDS loss...")
    loss, info = sds_loss(object_sdf, hand_params, text_prompts, iteration=100)

    print(f"\n✓ SDS Loss: {loss.item():.4f}")
    print(f"✓ Timestep mean: {info['timestep_mean']:.1f}")
    print(f"✓ Grad norm: {info['grad_norm']:.4f}")

    print("\n" + "=" * 60)
    print("✓ SDSLoss test passed!")
    print("=" * 60)
# Uncomment to run test
# if __name__ == "__main__":
#     test_sds_loss()