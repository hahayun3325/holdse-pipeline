"""
SDS Weight Scheduler for Progressive Loss Balancing
Addresses the 10× weight imbalance issue that caused object reconstruction failure.
"""

import torch
import logging
from typing import Dict, Union

logger = logging.getLogger(__name__)


class SDSWeightScheduler:
    """
    Dynamically adjusts SDS loss weight during training to balance RGB and SDS gradients.

    Problem: Fixed w_sds=10.0 caused 99.5% loss contribution from SDS, suppressing RGB.
    Solution: Schedule SDS weight from 0.5 → 0.05 → 0.0 to prioritize geometric accuracy.
    """

    def __init__(
        self,
        schedule: Dict[int, float],
        interpolation: str = "linear",
        enabled: bool = True
    ):
        """
        Args:
            schedule: Dict mapping step -> weight. Example:
                {0: 0.5, 1000: 0.2, 1500: 0.0}
            interpolation: "linear" or "step" (no interpolation)
            enabled: If False, returns first weight value (fallback to fixed weight)
        """
        self.enabled = enabled
        self.interpolation = interpolation

        # Sort schedule by step
        self.schedule = sorted(schedule.items())  # [(step, weight), ...]

        if not self.schedule:
            raise ValueError("Schedule cannot be empty")

        # Store for diagnostics
        self.last_weight = self.schedule[0][1]
        self.last_step = -1

        logger.info("=" * 70)
        logger.info("SDS Weight Scheduler Initialized")
        logger.info("=" * 70)
        logger.info(f"  Enabled: {self.enabled}")
        logger.info(f"  Interpolation: {self.interpolation}")
        logger.info(f"  Schedule keyframes:")
        for step, weight in self.schedule:
            logger.info(f"    Step {step:6d}: w_sds = {weight:.3f}")
        logger.info("=" * 70)

    def get_weight(self, current_step: int) -> float:
        """
        Get SDS weight for current training step.

        Args:
            current_step: Current global training step

        Returns:
            weight: SDS loss weight (float)
        """
        if not self.enabled:
            return self.schedule[0][1]  # Return first weight as fixed value

        # Before first keyframe
        if current_step <= self.schedule[0][0]:
            weight = self.schedule[0][1]

        # After last keyframe
        elif current_step >= self.schedule[-1][0]:
            weight = self.schedule[-1][1]

        # Between keyframes
        else:
            weight = self._interpolate(current_step)

        # Update diagnostics
        self.last_weight = weight
        self.last_step = current_step

        return weight

    def _interpolate(self, current_step: int) -> float:
        """Interpolate weight between surrounding keyframes."""

        # Find surrounding keyframes
        for i in range(len(self.schedule) - 1):
            step_start, weight_start = self.schedule[i]
            step_end, weight_end = self.schedule[i + 1]

            if step_start <= current_step < step_end:
                if self.interpolation == "linear":
                    # Linear interpolation
                    progress = (current_step - step_start) / (step_end - step_start)
                    weight = weight_start + progress * (weight_end - weight_start)
                    return weight

                elif self.interpolation == "step":
                    # Step function (no interpolation)
                    return weight_start

                else:
                    raise ValueError(f"Unknown interpolation: {self.interpolation}")

        # Shouldn't reach here, but return last weight as fallback
        return self.schedule[-1][1]

    def get_diagnostics(self) -> Dict[str, Union[float, int]]:
        """Get diagnostic information for logging."""
        return {
            "current_weight": self.last_weight,
            "current_step": self.last_step,
            "enabled": self.enabled
        }

    def get_schedule_info(self) -> str:
        """Get human-readable schedule summary."""
        lines = ["SDS Weight Schedule:"]
        for step, weight in self.schedule:
            lines.append(f"  Step {step:6d}: {weight:.3f}")
        return "\n".join(lines)


def create_sds_scheduler_from_config(phase3_config) -> SDSWeightScheduler:
    """
    Factory function to create scheduler from config.

    Args:
        phase3_config: Phase 3 configuration dict with w_sds_schedule

    Returns:
        SDSWeightScheduler instance
    """
    # Check if scheduling is enabled
    if 'w_sds_schedule' not in phase3_config:
        # Fallback to fixed weight
        fixed_weight = phase3_config.get('w_sds', 10.0)
        logger.warning(
            f"No w_sds_schedule found in config. "
            f"Using fixed weight: {fixed_weight}"
        )
        return SDSWeightScheduler(
            schedule={0: fixed_weight},
            enabled=False
        )

    sds_schedule_cfg = phase3_config['w_sds_schedule']

    # Extract schedule dict (all keys except 'enabled' and 'interpolation')
    schedule = {}
    for key, value in sds_schedule_cfg.items():
        if key.startswith('step_'):
            step = int(key.replace('step_', ''))
            schedule[step] = float(value)

    if not schedule:
        raise ValueError(
            "w_sds_schedule must contain at least one 'step_X' entry. "
            f"Found keys: {list(sds_schedule_cfg.keys())}"
        )

    enabled = sds_schedule_cfg.get('enabled', True)
    interpolation = sds_schedule_cfg.get('interpolation', 'linear')

    return SDSWeightScheduler(
        schedule=schedule,
        interpolation=interpolation,
        enabled=enabled
    )
