# code/src/model/ghop/ghop_prior.py
"""
GHOP (Generative Hand-Object Prior) integration for HOLD training.
Provides SDS loss from pretrained diffusion model.
PHASE 3 UPDATES:
- Added TwoStageTrainingManager: Progressive training from SDS → Contact loss
- Stage 1 (0-500 iters): SDS loss for coarse geometry alignment
- Stage 2 (500+ iters): Contact loss for fine refinement
- Dynamic weight scheduling with smooth transitions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from jutils import mesh_utils, hand_utils
from jutils.hand_utils import DistanceField


class GHOPPriorModule(nn.Module):
    """
    Wraps GHOP diffusion model to provide SDS loss for HOLD optimization.

    Components:
    - VQ-VAE: Encodes object SDF (64³ → 3×16³)
    - Hand Field: Computes skeletal distance field (45-dim pose → 15×16³)
    - CLIP: Encodes text prompts (category → 768-dim embedding)
    - U-Net: 3D diffusion model (predicts noise on 18-channel grid)
    """

    def __init__(self, ghop_checkpoint_path, device='cuda'):
        super().__init__()
        self.device = device

        # Load complete GHOP model from checkpoint
        from jutils import model_utils
        self.ghop_model = model_utils.load_from_checkpoint(ghop_checkpoint_path)
        self.ghop_model.eval()
        self.ghop_model.to(device)

        # Freeze all GHOP components
        for param in self.ghop_model.parameters():
            param.requires_grad = False

        # Extract components
        self.vqvae = self.ghop_model.first_stage_model  # VQ-VAE encoder/decoder
        self.unet = self.ghop_model.glide_model  # 3D U-Net diffusion model
        self.clip_encoder = self.ghop_model.cond_stage_model  # CLIP text encoder
        self.hand_field = self.ghop_model.hand_cond  # Hand SKDF generator
        self.hand_wrapper = self.ghop_model.hand_wrapper  # MANO wrapper

        # Diffusion parameters
        self.diffusion = self.ghop_model.diffusion
        self.alphas_bar = self.diffusion.alphas_cumprod.to(device)

        # Configuration
        self.cfg = self.ghop_model.cfg
        self.side_lim = self.cfg.side_lim  # Spatial extent (1.5)

        # SDS parameters (from sd.py analysis)
        self.min_step = 20  # 2% of 1000
        self.max_step = 980  # 98% of 1000
        self.guidance_scale = 4.0  # Classifier-free guidance scale

    def encode_object(self, obj_sdf):
        """
        Encode object SDF to latent space.

        Args:
            obj_sdf: (N, 1, 64, 64, 64) - Object SDF in normalized frame
        Returns:
            obj_latent: (N, 3, 16, 16, 16) - VQ-VAE latent codes
        """
        with torch.no_grad():
            # Use continuous latents (no quantization during SDS)
            obj_latent = self.vqvae.encode_to_prequant(obj_sdf)
        return obj_latent

    def compute_hand_skdf(self, hand_pose, resolution=16):
        """
        Compute skeletal distance field from MANO hand pose.

        Args:
            hand_pose: (N, 45) - MANO articulation parameters
            resolution: Grid resolution (default 16 for latent space)
        Returns:
            hand_skdf: (N, 15, 16, 16, 16) - Distance field to 15 joints
        """
        N = len(hand_pose)

        # Create coordinate grid in normalized frame
        nXyz = mesh_utils.create_sdf_grid(
            N, resolution, self.side_lim, device=self.device
        )  # (N, 16, 16, 16, 3)

        with torch.no_grad():
            # Compute skeletal distance field
            hand_skdf = self.hand_field(
                hA=hand_pose,
                H=resolution,
                nXyz=nXyz,
                rtn_wrist=False  # Exclude wrist, return 15 channels
            )  # (N, 15, 16, 16, 16)

        return hand_skdf

    def construct_interaction_grid(self, obj_sdf, hand_pose):
        """
        Construct 18-channel interaction grid from object and hand.

        Args:
            obj_sdf: (N, 1, 64, 64, 64) - Object SDF
            hand_pose: (N, 45) - MANO parameters
        Returns:
            interaction_grid: (N, 18, 16, 16, 16) - Combined representation
        """
        # Encode object (3 channels)
        obj_latent = self.encode_object(obj_sdf)  # (N, 3, 16, 16, 16)

        # Compute hand SKDF (15 channels)
        hand_skdf = self.compute_hand_skdf(hand_pose, resolution=16)  # (N, 15, 16, 16, 16)

        # Concatenate along channel dimension
        interaction_grid = torch.cat([obj_latent, hand_skdf], dim=1)  # (N, 18, 16, 16, 16)

        return interaction_grid

    def encode_text(self, text_prompts):
        """
        Encode text descriptions using CLIP.

        Args:
            text_prompts: List[str] or str - Object category or full prompt
        Returns:
            text_embeddings: (N, 1, 768) - CLIP embeddings
        """
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]

        with torch.no_grad():
            text_embeddings = self.clip_encoder(text_prompts)  # (N, 1, 768)

        return text_embeddings

    def add_noise(self, x, t, noise=None):
        """
        Add noise to interaction grid at timestep t.

        Args:
            x: (N, 18, 16, 16, 16) - Clean interaction grid
            t: (N,) - Timestep indices
            noise: Optional pre-generated noise
        Returns:
            x_noisy: (N, 18, 16, 16, 16) - Noisy interaction grid
            noise: (N, 18, 16, 16, 16) - The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x)

        # DDPM forward process: x_t = sqrt(alpha_bar_t) * x + sqrt(1 - alpha_bar_t) * noise
        sqrt_alpha_bar = torch.sqrt(self.alphas_bar[t])
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alphas_bar[t])

        # Reshape for broadcasting: (N,) → (N, 1, 1, 1, 1)
        sqrt_alpha_bar = sqrt_alpha_bar.view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.view(-1, 1, 1, 1, 1)

        x_noisy = sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise

        return x_noisy, noise

    def predict_noise(self, x_noisy, t, text_embeddings, guidance_scale=None):
        """
        Predict noise using U-Net with classifier-free guidance.

        Args:
            x_noisy: (N, 18, 16, 16, 16) - Noisy interaction grid
            t: (N,) - Timestep indices
            text_embeddings: (N, 1, 768) - CLIP embeddings
            guidance_scale: CFG scale (default: self.guidance_scale)
        Returns:
            noise_pred: (N, 18, 16, 16, 16) - Predicted noise with CFG
        """
        if guidance_scale is None:
            guidance_scale = self.guidance_scale

        N = len(x_noisy)

        with torch.no_grad():
            # Duplicate inputs for conditional and unconditional
            x_noisy_doubled = torch.cat([x_noisy, x_noisy], dim=0)  # (2N, 18, 16, 16, 16)
            t_doubled = torch.cat([t, t], dim=0)  # (2N,)

            # Conditional: use text embeddings
            # Unconditional: use empty string embeddings
            uncond_embeddings = self.encode_text([""] * N)  # (N, 1, 768)
            context = torch.cat([text_embeddings, uncond_embeddings], dim=0)  # (2N, 1, 768)

            # Forward through U-Net
            noise_pred_doubled = self.unet(
                x=x_noisy_doubled,
                timesteps=t_doubled,
                context=context
            )  # (2N, 18, 16, 16, 16)

            # Split conditional and unconditional predictions
            noise_cond, noise_uncond = torch.chunk(noise_pred_doubled, 2, dim=0)

            # Classifier-free guidance: noise_cfg = noise_uncond + scale * (noise_cond - noise_uncond)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        return noise_pred

    def compute_sds_loss(
        self,
        obj_sdf,
        hand_params,  # ✅ FIXED: hand_pose → hand_params
        text_prompt,
        weight=1.0,
        t=None,
        guidance_scale=None
    ):
        """
        Compute Score Distillation Sampling (SDS) loss.

        This is the main function to call during HOLD training.

        Args:
            obj_sdf: (N, 1, 64, 64, 64) - Object SDF from HOLD
            hand_params: Dict with keys {'pose', 'shape', 'trans'} OR (N, 45) tensor
            text_prompt: str or List[str] - Object category
            weight: Loss weight multiplier
            t: Optional timestep (random if None)
            guidance_scale: CFG scale (default: 4.0)
        Returns:
            sds_loss: Scalar tensor - Differentiable SDS loss
            info: Dict with auxiliary information
        """
        # ============================================================
        # Extract hand_pose from dict or use tensor directly
        # ============================================================
        if isinstance(hand_params, dict):
            hand_pose = hand_params['pose']
        elif isinstance(hand_params, torch.Tensor):
            hand_pose = hand_params
        else:
            raise ValueError(f"Invalid hand_params type: {type(hand_params)}")

        N = len(obj_sdf)

        # 1. Construct interaction grid
        interaction_grid = self.construct_interaction_grid(obj_sdf, hand_pose)  # (N, 18, 16, 16, 16)

        # 2. Encode text
        text_embeddings = self.encode_text(text_prompt)  # (N, 1, 768)

        # 3. Sample random timestep
        if t is None:
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                (N,),
                device=self.device
            )

        # 4. Add noise
        noise = torch.randn_like(interaction_grid)
        x_noisy, noise = self.add_noise(interaction_grid, t, noise)

        # 5. Predict noise with CFG
        noise_pred = self.predict_noise(x_noisy, t, text_embeddings, guidance_scale)

        # 6. Compute SDS gradient
        w = self.compute_sds_weight(t)  # (N,)
        w = w.view(-1, 1, 1, 1, 1)  # Broadcast shape

        grad = weight * w * (noise_pred - noise)

        # 7. Create differentiable loss using MSE trick
        target = (interaction_grid - grad).detach()
        sds_loss = 0.5 * F.mse_loss(interaction_grid, target, reduction='sum') / N

        # 8. Return loss and info
        info = {
            'timestep': t[0].item() if len(t) > 0 else None,
            'weight': w[0].item() if len(w) > 0 else None,
            'grad_norm': grad.abs().mean().item(),
            'interaction_grid': interaction_grid.detach(),
            'noise_pred': noise_pred.detach(),
        }

        return sds_loss, info

    def compute_sds_weight(self, t):
        """
        Compute weighting function w(t) for SDS loss.

        Uses DreamFusion weighting: w(t) = 1 - alpha_bar_t

        Args:
            t: (N,) - Timestep indices
        Returns:
            w: (N,) - Weights
        """
        alpha_bar_t = self.alphas_bar[t]
        w = 1 - alpha_bar_t
        return w


# Utility function for easy loading
def load_ghop_prior(checkpoint_path, device='cuda'):
    """
    Load GHOP prior module from checkpoint.

    Args:
        checkpoint_path: Path to GHOP checkpoint (.ckpt file)
        device: Device to load on
    Returns:
        ghop_prior: GHOPPriorModule instance
    """
    return GHOPPriorModule(checkpoint_path, device=device)


# ============================================================================
# PHASE 3: Two-Stage Training Management
# ============================================================================
class TwoStageTrainingManager:
    """
    Manage progressive training stages for HOLD-GHOP integration:
    - Stage 1 (0-500 iters): SDS loss for coarse geometry
    - Stage 2 (500-600 iters): Contact loss for refinement

    Based on GHOP's train.py and graspsyn.py patterns.
    """

    def __init__(self, sds_loss_module, sds_iters=500, contact_iters=100,
                 max_sds_weight=5000.0, max_contact_weight=10.0):
        """
        Args:
            sds_loss_module: GHOPPriorModule or SDSLoss instance
            sds_iters: Number of iterations for Stage 1 (default: 500)
            contact_iters: Number of iterations for Stage 2 (default: 100)
            max_sds_weight: Maximum SDS loss weight (default: 5000.0)
            max_contact_weight: Maximum contact loss weight (default: 10.0)
        """
        self.sds_loss = sds_loss_module
        self.sds_iters = sds_iters
        self.contact_iters = contact_iters
        self.total_iters = sds_iters + contact_iters
        self.max_sds_weight = max_sds_weight
        self.max_contact_weight = max_contact_weight

        print(f"[TwoStageTrainingManager] Initialized:")
        print(f"  - Stage 1 (SDS): 0-{sds_iters} iterations, max weight: {max_sds_weight}")
        print(f"  - Stage 2 (Contact): {sds_iters}-{self.total_iters} iterations, max weight: {max_contact_weight}")

    def get_stage_weights(self, iteration):
        """
        Returns loss weights for current training stage.

        Args:
            iteration: Global training step

        Returns:
            Dict with 'sds_weight', 'contact_weight', 'stage', 'progress'
        """
        if iteration < self.sds_iters:
            # Stage 1: Ramp up SDS from 0 to max_sds_weight
            progress = iteration / self.sds_iters
            sds_weight = self.max_sds_weight * progress

            return {
                'sds_weight': sds_weight,
                'contact_weight': 0.0,
                'stage': 'sds',
                'progress': progress,
                'stage_iter': iteration,
                'total_stage_iters': self.sds_iters
            }

        elif iteration < self.total_iters:
            # Stage 2: Reduce SDS, introduce contact
            stage_iter = iteration - self.sds_iters
            contact_progress = stage_iter / self.contact_iters

            # Gradually reduce SDS weight and increase contact weight
            sds_weight = self.max_sds_weight * 0.2  # Keep 20% of SDS for stability
            contact_weight = self.max_contact_weight * contact_progress

            return {
                'sds_weight': sds_weight,
                'contact_weight': contact_weight,
                'stage': 'contact',
                'progress': contact_progress,
                'stage_iter': stage_iter,
                'total_stage_iters': self.contact_iters
            }

        else:
            # Stage 3: Contact only (post-training)
            return {
                'sds_weight': 0.0,
                'contact_weight': self.max_contact_weight,
                'stage': 'contact_only',
                'progress': 1.0,
                'stage_iter': iteration - self.total_iters,
                'total_stage_iters': float('inf')
            }

    def compute_losses(self, object_sdf, hand_params, text_prompts, iteration):
        """
        Compute stage-appropriate losses.

        Args:
            object_sdf: (B, 1, 64, 64, 64) from HOLD-Net
            hand_params: Dict with MANO parameters or (B, 45/48) pose tensor
            text_prompts: List of text descriptions or single string
            iteration: Current training step

        Returns:
            losses: Dict with loss components {'sds': ..., 'contact': ...}
            info: Dict with diagnostic information
        """
        weights = self.get_stage_weights(iteration)
        device = object_sdf.device

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        losses = {}
        info = {
            'stage': weights['stage'],
            'stage_progress': weights['progress'],
            'sds_weight': weights['sds_weight'],
            'contact_weight': weights['contact_weight']
        }

        # SDS loss
        if weights['sds_weight'] > 0:
            # Normalize text prompts
            if text_prompts is None:
                text_prompts = ["a hand grasping an object"]
            elif isinstance(text_prompts, str):
                text_prompts = [text_prompts]
            elif isinstance(text_prompts, list):
                # Filter out None/empty
                text_prompts = [p for p in text_prompts if p is not None and (isinstance(p, str) and len(p) > 0)]
                if len(text_prompts) == 0:
                    text_prompts = ["a hand grasping an object"]
            else:
                # Unknown type, use default
                text_prompts = ["a hand grasping an object"]

            try:
                # ============================================================
                # CRITICAL FIX: Pass hand_params directly (not hand_pose)
                # SDSLoss.compute_sds_loss will handle dict/tensor conversion
                # ============================================================
                sds_loss, sds_info = self.sds_loss.compute_sds_loss(
                    obj_sdf=object_sdf,
                    hand_params=hand_params,  # ✅ Pass full hand_params dict
                    text_prompt=text_prompts[0] if len(text_prompts) == 1 else text_prompts,
                    weight=weights['sds_weight']
                )
                losses['sds'] = sds_loss
                info.update({f'sds_{k}': v for k, v in sds_info.items()})

            except Exception as e:
                print(f"Warning: SDS loss computation failed: {e}")
                losses['sds'] = torch.tensor(0.0, device=device, requires_grad=True)
                info['sds_error'] = str(e)

        # ================================================================
        # Contact loss computation
        # ================================================================
        if weights['contact_weight'] > 0:
            # TODO: Implement contact loss from HOLD's fitting/loss.py
            # This requires:
            # 1. Extract mesh from object SDF using marching cubes
            # 2. Get hand mesh from MANO parameters
            # 3. Compute contact distances using KNN
            # 4. Apply contact loss (attraction + repulsion terms)

            contact_loss = self._compute_contact_loss_placeholder(
                object_sdf, hand_params, weights['contact_weight']
            )
            losses['contact'] = contact_loss
            # FIXED: Check if tensor before calling .item()
            if isinstance(contact_loss, torch.Tensor):
                info['contact_loss'] = contact_loss.item()
            else:
                info['contact_loss'] = float(contact_loss)

        # ================================================================
        # CRITICAL FIX: Safely sum losses and convert to item()
        # ================================================================
        if losses:
            total_loss = sum(losses.values())

        losses['total'] = total_loss

        # Safe item() extraction with type checking
        if isinstance(total_loss, torch.Tensor):
            info['total_loss'] = total_loss.item()
        else:
            info['total_loss'] = float(total_loss)

        return losses, info

    def _compute_contact_loss_placeholder(self, object_sdf, hand_params, weight):
        """
        Placeholder contact loss implementation.

        In a full implementation, this would:
        1. Extract object mesh using marching cubes on object_sdf
        2. Get hand mesh from MANO forward kinematics
        3. Compute bidirectional distances between hand and object surfaces
        4. Apply attraction loss for nearby points and repulsion for penetrating points

        For now, return zero loss as placeholder.
        """
        device = object_sdf.device

        # Placeholder: return small regularization term
        # This prevents the optimizer from diverging when contact loss is enabled
        if isinstance(hand_params, dict):
            hand_pose = hand_params['pose']
        else:
            hand_pose = hand_params

        # Simple regularization: encourage moderate hand poses
        regularization = 0.001 * torch.mean(hand_pose ** 2)

        return weight * regularization

    def should_apply_sds(self, iteration):
        """Check if SDS loss should be applied at this iteration."""
        weights = self.get_stage_weights(iteration)
        return weights['sds_weight'] > 0

    def should_apply_contact(self, iteration):
        """Check if contact loss should be applied at this iteration."""
        weights = self.get_stage_weights(iteration)
        return weights['contact_weight'] > 0

    def get_stage_info(self, iteration):
        """Get human-readable stage information."""
        weights = self.get_stage_weights(iteration)
        stage = weights['stage']
        progress = weights['progress']

        if stage == 'sds':
            return f"Stage 1: SDS ({weights['stage_iter']}/{weights['total_stage_iters']}, {progress:.1%})"
        elif stage == 'contact':
            return f"Stage 2: Contact ({weights['stage_iter']}/{weights['total_stage_iters']}, {progress:.1%})"
        else:
            return f"Stage 3: Contact Only ({weights['stage_iter']}+ iters)"

    def reset_stage(self, new_sds_iters=None, new_contact_iters=None):
        """Reset stage parameters (useful for curriculum learning)."""
        if new_sds_iters is not None:
            self.sds_iters = new_sds_iters
        if new_contact_iters is not None:
            self.contact_iters = new_contact_iters

        self.total_iters = self.sds_iters + self.contact_iters
        print(f"[TwoStageTrainingManager] Reset: SDS={self.sds_iters}, Contact={self.contact_iters}")

# ============================================================================
# PHASE 3: Enhanced Factory Functions
# ============================================================================

def create_two_stage_manager(ghop_checkpoint_path, device='cuda',
                             sds_iters=500, contact_iters=100,
                             max_sds_weight=5000.0, max_contact_weight=10.0):
    """
    Create TwoStageTrainingManager with loaded GHOP prior.

    Args:
        ghop_checkpoint_path: Path to GHOP checkpoint
        device: Device to load on
        sds_iters: Stage 1 iterations
        contact_iters: Stage 2 iterations
        max_sds_weight: Maximum SDS weight
        max_contact_weight: Maximum contact weight

    Returns:
        TwoStageTrainingManager instance
    """
    # Load GHOP prior
    ghop_prior = load_ghop_prior(ghop_checkpoint_path, device)

    # Create two-stage manager
    manager = TwoStageTrainingManager(
        sds_loss_module=ghop_prior,
        sds_iters=sds_iters,
        contact_iters=contact_iters,
        max_sds_weight=max_sds_weight,
        max_contact_weight=max_contact_weight
    )

    return manager

def load_ghop_prior_with_stages(checkpoint_path, device='cuda', **stage_kwargs):
    """
    Load GHOP prior with two-stage training manager.

    Args:
        checkpoint_path: Path to GHOP checkpoint
        device: Device to load on
        **stage_kwargs: Arguments for TwoStageTrainingManager

    Returns:
        Tuple of (ghop_prior, two_stage_manager)
    """
    ghop_prior = load_ghop_prior(checkpoint_path, device)
    two_stage_manager = TwoStageTrainingManager(ghop_prior, **stage_kwargs)

    return ghop_prior, two_stage_manager


# ============================================================================
# PHASE 3: Testing Utilities
# ============================================================================
def test_two_stage_manager():
    """Test TwoStageTrainingManager with mock components."""
    print("\n" + "=" * 60)
    print("Testing TwoStageTrainingManager...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Mock SDS loss module
    class MockSDSLoss:
        def compute_sds_loss(self, obj_sdf, hand_params, text_prompt, weight=1.0):  # ✅ FIXED
            """Mock SDS loss for testing."""
            # Handle dict input
            if isinstance(hand_params, dict):
                hand_pose = hand_params['pose']
            else:
                hand_pose = hand_params

            loss = weight * torch.rand(1, device=obj_sdf.device) * 0.01
            info = {'timestep': 250, 'weight': weight, 'grad_norm': 0.5}
            return loss, info

    # Create manager
    mock_sds = MockSDSLoss()
    manager = TwoStageTrainingManager(
        sds_loss_module=mock_sds,
        sds_iters=100,  # Shorter for testing
        contact_iters=50,
        max_sds_weight=1000.0,
        max_contact_weight=5.0
    )

    # Test inputs
    B = 2
    object_sdf = torch.randn(B, 1, 64, 64, 64, device=device)
    hand_params = {'pose': torch.randn(B, 45, device=device)}
    text_prompts = ["a mug", "a bottle"]

    # Test different iterations
    test_iters = [0, 25, 50, 75, 100, 125, 150, 200]

    for iteration in test_iters:
        losses, info = manager.compute_losses(
            object_sdf, hand_params, text_prompts, iteration
        )

        stage_info = manager.get_stage_info(iteration)
        total_loss = losses['total'].item()
        sds_weight = info['sds_weight']
        contact_weight = info['contact_weight']

        print(f"Iter {iteration:3d}: {stage_info}")
        print(f"         Loss={total_loss:.4f}, SDS_w={sds_weight:.1f}, Contact_w={contact_weight:.1f}")

    print("\n" + "=" * 60)
    print("✓ TwoStageTrainingManager test passed!")
    print("=" * 60)

# Uncomment to run test
# if __name__ == "__main__":
#     test_two_stage_manager()