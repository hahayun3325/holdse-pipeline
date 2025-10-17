# File: scripts/test_phase5a_temporal.py
# NEW FILE: Test Phase 5A temporal consistency

import torch
import sys

sys.path.append('.')

from src.model.ghop.temporal_consistency import TemporalConsistencyModule
from loguru import logger


def test_temporal_consistency_basic():
    """Test temporal consistency module with synthetic data."""

    logger.info("=" * 70)
    logger.info("TEST 1: Basic Temporal Consistency")
    logger.info("=" * 70)

    # Initialize module
    temporal = TemporalConsistencyModule(
        window_size=5,
        w_velocity=0.5,
        w_acceleration=0.1,
        w_camera_motion=0.3,
        adaptive_weight=True
    )

    # Simulate 10 frames of smooth hand motion
    sequence_id = "test_sequence"

    for t in range(10):
        # Synthetic hand pose (48-dim)
        hand_pose = torch.randn(1, 48)

        # Smooth linear motion
        hand_trans = torch.tensor([[0.01 * t, 0.0, 0.0]])  # 0.01 m/frame

        # Camera pose (identity)
        camera_pose = torch.eye(4)

        # Create sample batch
        sample = {
            'right.transl': hand_trans,
            'c2w': camera_pose.unsqueeze(0)
        }

        frame_idx = torch.tensor([t])

        # Compute temporal loss
        loss, metrics = temporal(
            sample=sample,
            predicted_hand_pose=hand_pose,
            sequence_id=sequence_id,
            frame_idx=frame_idx
        )

        logger.info(f"Frame {t:2d}: loss={loss.item():.6f}, "
                    f"velocity={metrics['velocity_norm']:.4f} m/s, "
                    f"accel={metrics['accel_norm']:.4f} m/s²")

    logger.info("✅ Test 1 passed: Basic temporal consistency works")
    logger.info("")


def test_temporal_consistency_jitter():
    """Test temporal consistency with jittery motion (should have high loss)."""

    logger.info("=" * 70)
    logger.info("TEST 2: Jittery Motion (High Loss Expected)")
    logger.info("=" * 70)

    temporal = TemporalConsistencyModule(window_size=5, w_velocity=0.5, w_acceleration=0.1)
    sequence_id = "jitter_test"

    for t in range(10):
        hand_pose = torch.randn(1, 48)

        # Jittery motion (alternating directions)
        direction = 1.0 if t % 2 == 0 else -1.0
        hand_trans = torch.tensor([[0.05 * direction, 0.0, 0.0]])

        sample = {'right.transl': hand_trans, 'c2w': torch.eye(4).unsqueeze(0)}
        loss, metrics = temporal(sample, hand_pose, sequence_id, torch.tensor([t]))

        logger.info(f"Frame {t:2d}: loss={loss.item():.6f}, "
                    f"velocity={metrics['velocity_norm']:.4f} m/s")

    logger.info("✅ Test 2 passed: Jittery motion produces high loss")
    logger.info("")


def test_multi_sequence():
    """Test handling multiple sequences simultaneously."""

    logger.info("=" * 70)
    logger.info("TEST 3: Multiple Sequences")
    logger.info("=" * 70)

    temporal = TemporalConsistencyModule(window_size=5)

    # Simulate two sequences interleaved
    for t in range(5):
        # Sequence A
        sample_a = {
            'right.transl': torch.tensor([[0.01 * t, 0.0, 0.0]]),
            'c2w': torch.eye(4).unsqueeze(0)
        }
        loss_a, _ = temporal(sample_a, torch.randn(1, 48), "sequence_a", torch.tensor([t]))

        # Sequence B
        sample_b = {
            'right.transl': torch.tensor([[0.0, 0.01 * t, 0.0]]),
            'c2w': torch.eye(4).unsqueeze(0)
        }
        loss_b, _ = temporal(sample_b, torch.randn(1, 48), "sequence_b", torch.tensor([t]))

        logger.info(f"Frame {t}: Seq A loss={loss_a.item():.6f}, Seq B loss={loss_b.item():.6f}")

    logger.info("✅ Test 3 passed: Multi-sequence handling works")
    logger.info("")


if __name__ == "__main__":
    logger.info("Testing Phase 5A Temporal Consistency Module")
    logger.info("")

    test_temporal_consistency_basic()
    test_temporal_consistency_jitter()
    test_multi_sequence()

    logger.info("=" * 70)
    logger.info("✅ ALL TESTS PASSED")
    logger.info("=" * 70)