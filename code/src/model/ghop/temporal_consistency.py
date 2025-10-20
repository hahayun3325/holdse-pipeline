"""
Temporal Consistency Module for Video Sequences

Implements temporal smoothness using consecutive frame pairs from hoi.py
SceneDataset. Enforces first and second-order temporal derivatives for
physically plausible hand motion trajectories.

Architecture verified from:
- hoi.py (Lines 170-214): Frame pairing with (t, t+1) consecutive frames
- hoi.py (Lines 197-206): Next frame fields with '_n' suffix
- hoi.py (Line 172): Dataset length = N-1 for frame pairs
- mixdata.py: Multi-dataset temporal handling

Key Implementation Details from GHOP Source:
1. Frame pairs: idx and idx+1 (Lines 197, 204-205)
2. Camera poses: c2w (current), c2w_n (next)
3. Hand poses: hA (current), hA_n (next)
4. Transforms: hTo, wTh, onTo with '_n' variants

Author: HOLD Team
Date: October 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from collections import deque
import logging
import numpy as np

logger = logging.getLogger(__name__)


class TemporalConsistencyModule(nn.Module):
    """
    Temporal consistency enforcement using GHOP's frame pairing strategy.

    Uses consecutive frames (t, t+1) from hoi.py SceneDataset to enforce:
    1. Velocity consistency: First-order temporal derivative
    2. Acceleration regularization: Second-order smoothness
    3. Camera motion compensation: Camera-relative pose consistency

    Args:
        window_size: History buffer size for temporal context (default: 5)
        w_velocity: Hand pose velocity loss weight (default: 0.5)
        w_acceleration: Second-order smoothness weight (default: 0.1)
        w_camera_motion: Camera motion consistency weight (default: 0.3)
        adaptive_weight: Enable dynamic weighting based on motion magnitude
    """

    def __init__(
            self,
            window_size: int = 5,
            w_velocity: float = 0.5,
            w_acceleration: float = 0.1,
            w_camera_motion: float = 0.3,
            adaptive_weight: bool = True
    ):
        super().__init__()

        self.window_size = window_size
        self.w_velocity = w_velocity
        self.w_acceleration = w_acceleration
        self.w_camera_motion = w_camera_motion
        self.adaptive_weight = adaptive_weight

        # History buffers (per-sequence tracking)
        # Dict[seq_id, deque of [45]] for hand poses
        self.pose_history: Dict[str, deque] = {}
        # Dict[seq_id, deque of [4, 4]] for camera poses
        self.camera_history: Dict[str, deque] = {}
        # Dict[seq_id, deque of floats] for velocity magnitudes
        self.velocity_history: Dict[str, deque] = {}

        logger.info(
            f"[Phase 5] Temporal Consistency initialized:\n"
            f"  - Window size: {window_size}\n"
            f"  - Velocity weight: {w_velocity}\n"
            f"  - Acceleration weight: {w_acceleration}\n"
            f"  - Camera motion weight: {w_camera_motion}\n"
            f"  - Adaptive weighting: {adaptive_weight}"
        )

    def forward(
            self,
            sample: Dict,
            predicted_hand_pose: torch.Tensor,
            sequence_id: Optional[str] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute temporal consistency loss using frame pairs from hoi.py.

        Verified from hoi.py Lines 170-214:
        - sample['hA']: [B, 45] current frame hand pose (Line 204)
        - sample['hA_n']: [B, 45] next frame hand pose (Line 205)
        - sample['c2w']: [B, 4, 4] current camera pose (Line 195)
        - sample['c2w_n']: [B, 4, 4] next camera pose (Line 197)

        Args:
            sample: Batch dict from SceneDataset (hoi.py)
            predicted_hand_pose: [B, 45] model's predicted pose
            sequence_id: Video sequence identifier for history tracking

        Returns:
            temporal_loss: Scalar temporal consistency loss
            metrics: Dictionary with loss components
        """
        batch_size = predicted_hand_pose.shape[0]
        device = predicted_hand_pose.device

        if sequence_id is None:
            sequence_id = "default"

        # Initialize history if needed
        if sequence_id not in self.pose_history:
            self.pose_history[sequence_id] = deque(maxlen=self.window_size)
            self.camera_history[sequence_id] = deque(maxlen=self.window_size)
            self.velocity_history[sequence_id] = deque(maxlen=self.window_size)

        total_loss = torch.tensor(0.0, device=device)
        metrics = {
            'velocity': 0.0,
            'acceleration': 0.0,
            'camera_motion': 0.0,
            'adaptive_weight': 1.0
        }

        # Extract ground truth next frame poses from hoi.py Lines 204-206
        hA_next_gt = sample.get('hA_n', None)  # [B, 45]
        c2w_current = sample.get('c2w', None)  # [B, 4, 4]
        c2w_next = sample.get('c2w_n', None)  # [B, 4, 4]

        # Compute losses for each batch element
        for b in range(batch_size):
            hA_pred = predicted_hand_pose[b]  # [45]

            # =================================================================
            # 1. VELOCITY CONSISTENCY (First-order temporal derivative)
            # =================================================================
            if len(self.pose_history[sequence_id]) > 0:
                hA_prev = self.pose_history[sequence_id][-1]  # [45]

                # Current velocity: v_t = pose_t - pose_{t-1}
                velocity_current = hA_pred - hA_prev

                # Expected velocity from ground truth next frame
                if hA_next_gt is not None:
                    velocity_expected = hA_next_gt[b] - hA_pred

                    # Velocity smoothness loss
                    velocity_loss = F.mse_loss(velocity_current, velocity_expected)

                    # Adaptive weighting based on motion magnitude
                    if self.adaptive_weight:
                        motion_magnitude = torch.norm(velocity_expected)
                        # Reduce weight for small motion, increase for large motion
                        adaptive_factor = torch.sigmoid(motion_magnitude - 0.1)
                    else:
                        adaptive_factor = 1.0

                    total_loss = total_loss + self.w_velocity * adaptive_factor * velocity_loss
                    metrics['velocity'] += velocity_loss.item()
                    metrics['adaptive_weight'] += adaptive_factor.item()

            # =================================================================
            # 2. ACCELERATION REGULARIZATION (Second-order temporal derivative)
            # =================================================================
            if len(self.pose_history[sequence_id]) >= 2:
                hA_prev = self.pose_history[sequence_id][-1]  # pose_{t-1}
                hA_prev_prev = self.pose_history[sequence_id][-2]  # pose_{t-2}

                # Compute acceleration (second derivative)
                # a_t = pose_t - 2*pose_{t-1} + pose_{t-2}
                acceleration = hA_pred - 2 * hA_prev + hA_prev_prev

                # L2 regularization on acceleration
                accel_loss = torch.norm(acceleration, p=2)

                total_loss = total_loss + self.w_acceleration * accel_loss
                metrics['acceleration'] += accel_loss.item()

            # =================================================================
            # 3. CAMERA MOTION CONSISTENCY
            # =================================================================
            if (c2w_current is not None and
                    c2w_next is not None and
                    len(self.camera_history[sequence_id]) > 0):
                c2w_prev = self.camera_history[sequence_id][-1]  # [4, 4]

                # Compute relative camera motion between consecutive frames
                # Relative motion from t-1 to t
                relative_motion_current = self._compute_relative_transform(
                    c2w_prev, c2w_current[b]
                )  # [6] (translation + axis-angle)

                # Relative motion from t to t+1
                relative_motion_next = self._compute_relative_transform(
                    c2w_current[b], c2w_next[b]
                )  # [6]

                # Enforce smooth camera motion (consecutive motions should be similar)
                camera_loss = F.mse_loss(relative_motion_current, relative_motion_next)

                total_loss = total_loss + self.w_camera_motion * camera_loss
                metrics['camera_motion'] += camera_loss.item()

            # Update history for this batch element
            self.pose_history[sequence_id].append(hA_pred.detach().clone())
            if c2w_current is not None:
                self.camera_history[sequence_id].append(c2w_current[b].detach().clone())

        # Average over batch
        if batch_size > 0:
            total_loss = total_loss / batch_size
            for key in ['velocity', 'acceleration', 'camera_motion', 'adaptive_weight']:
                metrics[key] /= batch_size

        return total_loss, metrics

    def _compute_relative_transform(
            self,
            T1: torch.Tensor,
            T2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute relative transformation between two poses.

        Args:
            T1, T2: [4, 4] transformation matrices (c2w poses)

        Returns:
            relative: [6] translation (3) + rotation axis-angle (3)
        """
        # Relative transform: T1^{-1} @ T2
        try:
            # Try GPU inverse
            T1_inv = torch.inverse(T1)
        except RuntimeError as e:
            # Fallback to CPU if GPU fails
            logger.debug(f"[Temporal] GPU inverse failed: {e}, using CPU")
            T1_inv = torch.inverse(T1.cpu()).to(T1.device)
        T_rel = T1_inv @ T2

        # Extract translation
        translation = T_rel[:3, 3]  # [3]

        # Extract rotation and convert to axis-angle
        rotation_matrix = T_rel[:3, :3]  # [3, 3]
        rotation_aa = self._matrix_to_axis_angle(rotation_matrix)  # [3]

        # Concatenate
        relative = torch.cat([translation, rotation_aa])  # [6]

        return relative

    def _matrix_to_axis_angle(self, R: torch.Tensor) -> torch.Tensor:
        """
        Convert 3x3 rotation matrix to axis-angle representation.

        Args:
            R: [3, 3] rotation matrix

        Returns:
            axis_angle: [3] axis-angle vector
        """
        # Compute rotation angle
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        angle = torch.acos(((trace - 1) / 2).clamp(-1.0, 1.0))

        # Handle near-zero rotation
        if torch.abs(angle) < 1e-6:
            return torch.zeros(3, device=R.device)

        # Compute rotation axis
        axis = torch.stack([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ]) / (2 * torch.sin(angle))

        # axis-angle representation
        axis_angle = axis * angle

        return axis_angle

    def reset_history(self, sequence_id: Optional[str] = None):
        """
        Reset temporal history for sequence transitions.

        Args:
            sequence_id: Specific sequence to reset, or None for all
        """
        if sequence_id is None:
            self.pose_history.clear()
            self.camera_history.clear()
            self.velocity_history.clear()
            logger.info("[Phase 5] Temporal: All history cleared")
        else:
            if sequence_id in self.pose_history:
                del self.pose_history[sequence_id]
                del self.camera_history[sequence_id]
                del self.velocity_history[sequence_id]
                logger.info(f"[Phase 5] Temporal: History cleared for sequence '{sequence_id}'")

    def get_smoothed_prediction(
            self,
            predicted_pose: torch.Tensor,
            sequence_id: str = "default",
            alpha: float = 0.7
    ) -> torch.Tensor:
        """
        Apply exponential moving average smoothing for inference.

        Useful for real-time applications where smooth trajectories are desired.

        Args:
            predicted_pose: [B, 45] current prediction
            sequence_id: Sequence identifier
            alpha: Smoothing factor (0=full history, 1=no smoothing)

        Returns:
            smoothed_pose: [B, 45] temporally smoothed prediction
        """
        if sequence_id not in self.pose_history or len(self.pose_history[sequence_id]) == 0:
            return predicted_pose

        prev_pose = self.pose_history[sequence_id][-1]

        # Exponential moving average
        smoothed = alpha * predicted_pose + (1 - alpha) * prev_pose.unsqueeze(0)

        return smoothed

    def get_velocity_statistics(self, sequence_id: str = "default") -> Dict[str, float]:
        """
        Get velocity statistics for a sequence.

        Returns:
            stats: Dictionary with mean, std, max velocity magnitudes
        """
        if sequence_id not in self.velocity_history or len(self.velocity_history[sequence_id]) == 0:
            return {'mean': 0.0, 'std': 0.0, 'max': 0.0}

        velocities = list(self.velocity_history[sequence_id])

        stats = {
            'mean': float(np.mean(velocities)),
            'std': float(np.std(velocities)),
            'max': float(np.max(velocities))
        }

        return stats