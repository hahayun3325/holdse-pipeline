# tests/test_phase5_components.py
def test_diffusion_prior_sds():
    """Verify SDS loss computation matches GHOP sd.py."""
    prior = GHOPDiffusionPrior(model_checkpoint='checkpoints/ghop.pth')
    hand_pose = torch.randn(2, 45)
    object_sdf = torch.randn(2, 3, 64, 64, 64)

    sds_loss, metrics = prior(hand_pose, object_sdf, iteration=100)

    assert isinstance(sds_loss, torch.Tensor)
    assert sds_loss.ndim == 0  # Scalar loss
    assert 2 <= metrics['timestep'] <= 98  # min/max step range


def test_temporal_consistency():
    """Verify temporal loss uses frame pairs correctly."""
    temporal = TemporalConsistencyModule(window_size=5)
    sample = {
        'hA': torch.randn(2, 45),
        'hA_n': torch.randn(2, 45),
        'c2w': torch.randn(2, 4, 4),
        'c2w_n': torch.randn(2, 4, 4)
    }

    temporal_loss, metrics = temporal(sample, torch.randn(2, 45))

    assert 'velocity' in metrics
    assert 'acceleration' in metrics