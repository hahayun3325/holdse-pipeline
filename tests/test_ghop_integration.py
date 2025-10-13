"""
Unit tests for GHOP integration components.
"""

import torch
import pytest


def test_vqvae_encoding():
    """Test VQ-VAE encoding 64³ → 16³"""
    from src.model.ghop.autoencoder import GHOPVQVAEWrapper

    vqvae = GHOPVQVAEWrapper(vqvae_ckpt_path=None, device='cpu')

    # Test input
    object_sdf = torch.randn(2, 1, 64, 64, 64)
    hand_field = torch.randn(2, 15, 64, 64, 64)

    # Encode
    z_q, indices = vqvae.encode(object_sdf, hand_field)

    assert z_q.shape == (2, 3, 16, 16, 16), f"Wrong latent shape: {z_q.shape}"
    assert indices.shape == (2, 16, 16, 16), f"Wrong indices shape: {indices.shape}"
    print("✓ VQ-VAE encoding test passed")


def test_hand_field_generation():
    """Test 15-channel hand field generation"""
    from src.model.ghop.hand_field import HandFieldBuilder

    # Mock MANO server
    class MockMANO:
        def forward(self, pose, shape, trans):
            B = pose.shape[0]
            joints = torch.randn(B, 21, 3)
            return None, joints

    builder = HandFieldBuilder(MockMANO(), resolution=64)

    hand_params = {
        'pose': torch.randn(2, 48),
        'shape': torch.randn(2, 10),
        'trans': torch.randn(2, 3)
    }

    hand_field = builder(hand_params)

    assert hand_field.shape == (2, 15, 64, 64, 64), f"Wrong field shape: {hand_field.shape}"
    print("✓ Hand field generation test passed")


def test_sds_loss_computation():
    """Test SDS loss forward pass"""
    from src.model.ghop.ghop_loss import SDSLoss
    from src.model.ghop.autoencoder import GHOPVQVAEWrapper
    from src.model.ghop.diffusion import GHOP3DUNetWrapper
    from src.model.ghop.hand_field import HandFieldBuilder

    # Mock components
    vqvae = GHOPVQVAEWrapper(None, device='cpu')
    unet = GHOP3DUNetWrapper(None, device='cpu')

    class MockMANO:
        def forward(self, pose, shape, trans):
            return None, torch.randn(pose.shape[0], 21, 3)

    hand_builder = HandFieldBuilder(MockMANO())

    sds = SDSLoss(vqvae, unet, hand_builder)

    # Test inputs
    object_sdf = torch.randn(2, 1, 64, 64, 64)
    hand_params = {
        'pose': torch.randn(2, 48),
        'shape': torch.randn(2, 10),
        'trans': torch.randn(2, 3)
    }

    loss, info = sds(object_sdf, hand_params, ["test"], iteration=0)

    assert loss.ndim == 0, "Loss should be scalar"
    assert 'timestep_mean' in info, "Missing timestep info"
    print("✓ SDS loss computation test passed")


if __name__ == '__main__':
    test_vqvae_encoding()
    test_hand_field_generation()
    test_sds_loss_computation()
    print("\n✓ All tests passed!")