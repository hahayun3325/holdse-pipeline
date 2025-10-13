"""
Test script for Phase 2 GHOP integration.
Run before training to verify all components work.
"""
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf
from src.model.ghop.autoencoder import load_vqvae
from src.model.ghop.hand_field import HandSkeletalField
from src.model.ghop.interaction_grid import InteractionGridBuilder
from src.model.ghop.ghop_prior import GHOPPrior


def test_vqvae_loading():
    """Test VQ-VAE loading."""
    print("\n[1/4] Testing VQ-VAE loading...")
    try:
        config = OmegaConf.load("confs/general.yaml")
        vqvae_path = config.phase2.ghop.vqvae_checkpoint

        if not os.path.exists(vqvae_path):
            print(f"❌ VQ-VAE checkpoint not found: {vqvae_path}")
            return False

        vqvae = load_vqvae(vqvae_path, device='cuda')
        print(f"✓ VQ-VAE loaded successfully")
        print(f"  - Parameters: {sum(p.numel() for p in vqvae.parameters()) / 1e6:.1f}M")
        return True
    except Exception as e:
        print(f"❌ VQ-VAE loading failed: {e}")
        return False


def test_hand_field():
    """Test hand skeletal distance field."""
    print("\n[2/4] Testing hand skeletal distance field...")
    try:
        config = OmegaConf.load("confs/general.yaml")
        hand_field = HandSkeletalField(
            mano_dir=config.get('mano_dir', 'assets/mano'),
            device='cuda'
        )

        # Test forward pass
        B = 2
        hand_pose = torch.randn(B, 45, device='cuda')
        skdf = hand_field(hand_pose, resolution=16, spatial_lim=1.5)

        print(f"✓ Hand field computed successfully")
        print(f"  - Input shape: {hand_pose.shape}")
        print(f"  - Output shape: {skdf.shape}")
        assert skdf.shape == (B, 15, 16, 16, 16), f"Wrong shape: {skdf.shape}"
        return True
    except Exception as e:
        print(f"❌ Hand field test failed: {e}")
        return False


def test_interaction_grid():
    """Test interaction grid construction."""
    print("\n[3/4] Testing interaction grid construction...")
    try:
        config = OmegaConf.load("confs/general.yaml")

        # Load VQ-VAE
        vqvae = load_vqvae(config.phase2.ghop.vqvae_checkpoint, device='cuda')

        # Initialize builder
        builder = InteractionGridBuilder(vqvae, config.phase2)

        # Test inputs
        B = 2
        obj_sdf = torch.randn(B, 1, 64, 64, 64, device='cuda')
        hand_pose = torch.randn(B, 45, device='cuda')

        # Build grid
        interaction_grid, components = builder(obj_sdf, hand_pose)

        print(f"✓ Interaction grid built successfully")
        print(f"  - Object SDF: {obj_sdf.shape}")
        print(f"  - Hand pose: {hand_pose.shape}")
        print(f"  - Interaction grid: {interaction_grid.shape}")
        print(f"  - Object latent: {components['obj_latent'].shape}")
        print(f"  - Hand SKDF: {components['hand_skdf'].shape}")
        assert interaction_grid.shape == (B, 18, 16, 16, 16)
        return True
    except Exception as e:
        print(f"❌ Interaction grid test failed: {e}")
        return False


def test_ghop_prior():
    """Test complete GHOP prior."""
    print("\n[4/4] Testing GHOP prior...")
    try:
        config = OmegaConf.load("confs/general.yaml")
        ghop_prior = GHOPPrior(config.phase2)

        # Test SDS loss computation
        B = 2
        interaction_grid = torch.randn(B, 18, 16, 16, 16, device='cuda', requires_grad=True)
        categories = ['bottle', 'mug']

        loss_sds, sds_info = ghop_prior.compute_sds_loss(
            interaction_grid,
            categories,
            weight=5000.0
        )

        print(f"✓ GHOP prior forward pass successful")
        print(f"  - SDS loss: {loss_sds.item():.4f}")
        print(f"  - Timestep: {sds_info['timestep']}")
        print(f"  - Weight: {sds_info['weight']:.4f}")

        # Test backpropagation
        loss_sds.backward()
        print(f"  - Gradients: {interaction_grid.grad is not None}")
        return True
    except Exception as e:
        print(f"❌ GHOP prior test failed: {e}")
        return False


def main():
    print("=" * 60)
    print("Phase 2: GHOP Integration Tests")
    print("=" * 60)

    results = []
    results.append(("VQ-VAE Loading", test_vqvae_loading()))
    results.append(("Hand Field", test_hand_field()))
    results.append(("Interaction Grid", test_interaction_grid()))
    results.append(("GHOP Prior", test_ghop_prior()))

    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"{name:30s} {status}")

    all_passed = all(r[1] for r in results)
    print("=" * 60)
    if all_passed:
        print("All tests passed! ✓")
        print("You can now enable Phase 2 and start training.")
    else:
        print("Some tests failed. Please fix issues before training.")
    print("=" * 60)


if __name__ == "__main__":
    main()