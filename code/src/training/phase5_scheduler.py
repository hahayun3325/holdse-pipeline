"""
Phase 5 Training Scheduler

Implements multi-stage training with automatic phase transitions based on
GHOP's two-stage optimization strategy.

Architecture verified from:
- train.py (Lines 162-171): Warm-up and base reconstruction
- train.py (Line 270): SDS activation condition
- graspsyn.py (Lines 88-270): Two-stage SDS→Contact optimization
- graspsyn.py (Lines 166-167): Loss weight configuration

Training Stages:
1. Warm-up (0-100): Base HOLD reconstruction losses only
2. Phase 3 (100-500): Add SDS guidance from diffusion prior
3. Phase 4 (500-600): Add contact refinement losses
4. Phase 5 (600-800): Full integration with temporal consistency
5. Fine-tuning (800-1000): Reduce guidance, increase data fidelity

Author: HOLD Team
Date: October 2025
"""

import torch
from typing import Dict, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


class Phase5TrainingScheduler:
    """
    Multi-stage training scheduler for GHOP-HOLD integration.

    Implements progressive loss weighting and learning rate scheduling
    verified from GHOP's training scripts:
    - graspsyn.py Lines 132, 220: Learning rate configuration
    - graspsyn.py Lines 166-167: Loss weight w_sds=10
    - graspsyn.py Lines 238-243: Contact weights (w_pen=1, w_miss=1, w_damp=0.1)

    Args:
        total_iterations: Total training iterations (default: 1000)
        warmup_iters: Warm-up duration before GHOP activation
        phase3_start: Iteration to activate SDS guidance
        phase4_start: Iteration to activate contact refinement
        phase5_start: Iteration to activate full integration
        finetune_start: Iteration to begin reducing guidance
    """

    def __init__(
            self,
            total_iterations: int = 1000,
            warmup_iters: int = 100,
            phase3_start: int = 100,
            phase4_start: int = 500,
            phase5_start: int = 600,
            finetune_start: int = 800
    ):
        self.total_iterations = total_iterations
        self.warmup_iters = warmup_iters
        self.phase3_start = phase3_start
        self.phase4_start = phase4_start
        self.phase5_start = phase5_start
        self.finetune_start = finetune_start

        # Verify logical ordering
        assert warmup_iters <= phase3_start <= phase4_start <= phase5_start <= finetune_start <= total_iterations, \
            "Phase transitions must be in ascending order"

        # Store update frequency for adaptive contacts
        self.contact_update_frequency = 10  # From adaptive_contact_zones.py

        logger.info(
            f"[Phase 5] Scheduler: Training schedule initialized:\n"
            f"  - Warm-up:       [0, {warmup_iters})\n"
            f"  - Phase 3 (SDS): [{phase3_start}, {phase4_start})\n"
            f"  - Phase 4 (Contact): [{phase4_start}, {phase5_start})\n"
            f"  - Phase 5 (Full): [{phase5_start}, {finetune_start})\n"
            f"  - Fine-tuning:   [{finetune_start}, {total_iterations}]"
        )

    def get_loss_weights(self, iteration: int) -> Dict[str, float]:
        """
        Get dynamic loss weights for current iteration.

        Weight schedule verified from graspsyn.py:
        - Lines 166-167: w_sds = 10
        - Lines 238-243: w_pen=1, w_miss=1, w_damp=0.1

        Args:
            iteration: Current training iteration

        Returns:
            weights: Dictionary of loss component weights
        """
        weights = {
            'rgb': 1.0,  # Always active (base reconstruction)
            'semantic': 1.0,  # Always active
            'eikonal': 0.1,  # SDF regularization
            'sds': 0.0,  # Phase 3+
            'contact': 0.0,  # Phase 4+
            'diffusion': 0.0,  # Phase 5+ (enhanced SDS)
            'temporal': 0.0  # Phase 5+
        }

        # ====================================================================
        # Phase 3: SDS Guidance with Ramp-up
        # Verified from graspsyn.py Lines 166-167: w_sds = 10
        # ====================================================================
        if iteration >= self.phase3_start:
            # Linear ramp-up over 100 iterations for stable optimization
            rampup_progress = min((iteration - self.phase3_start) / 100, 1.0)
            weights['sds'] = 10.0 * rampup_progress

        # ====================================================================
        # Phase 4: Contact Refinement
        # Verified from graspsyn.py Lines 238-243: w_pen=1, w_miss=1
        # ====================================================================
        if iteration >= self.phase4_start:
            # Linear ramp-up over 100 iterations
            rampup_progress = min((iteration - self.phase4_start) / 100, 1.0)
            # Combined penetration + missed contact = 1.0
            weights['contact'] = 1.0 * rampup_progress

        # ====================================================================
        # Phase 5: Diffusion Prior + Temporal Consistency
        # Enhanced SDS via diffusion_prior (distinct from Phase 3)
        # ====================================================================
        if iteration >= self.phase5_start:
            # Linear ramp-up over 100 iterations
            rampup_progress = min((iteration - self.phase5_start) / 100, 1.0)

            # Enhanced diffusion prior (geometry refinement)
            weights['diffusion'] = 5.0 * rampup_progress

            # Temporal consistency (video sequences)
            weights['temporal'] = 0.5 * rampup_progress

        # ====================================================================
        # Fine-tuning: Reduce Guidance, Increase Data Fidelity
        # ====================================================================
        if iteration >= self.finetune_start:
            decay_progress = (iteration - self.finetune_start) / (self.total_iterations - self.finetune_start)

            # Reduce SDS guidance (50% reduction)
            weights['sds'] *= (1.0 - 0.5 * decay_progress)

            # Reduce diffusion prior (70% reduction)
            weights['diffusion'] *= (1.0 - 0.7 * decay_progress)

            # Increase RGB fidelity (50% increase)
            weights['rgb'] *= (1.0 + 0.5 * decay_progress)

        return weights

    def get_learning_rate_multiplier(self, iteration: int, base_lr: float = 1e-4) -> float:
        """
        Get learning rate multiplier for current iteration.

        Schedule verified from graspsyn.py:
        - Line 132: lr = 1e-2 / bs for SDS stage
        - Line 220: lr = 1e-3 for contact refinement
        - Lines 141-143: Cosine annealing scheduler

        Args:
            iteration: Current iteration
            base_lr: Base learning rate (not used, kept for API consistency)

        Returns:
            lr_multiplier: Multiplier for base learning rate
        """
        if iteration < self.warmup_iters:
            # Linear warm-up
            return iteration / self.warmup_iters

        elif iteration >= self.finetune_start:
            # Cosine annealing during fine-tuning
            # Verified from graspsyn.py Lines 141-143
            decay_iters = self.total_iterations - self.finetune_start
            progress = (iteration - self.finetune_start) / decay_iters

            # Cosine decay: 0.5 * (1 + cos(π * progress))
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        else:
            # Constant learning rate during main training
            return 1.0

    def should_update_contact_zones(self, current_iter: int) -> bool:
        """
        Determine if contact zones should be updated at current iteration.

        Contact zones are updated:
        1. Every `update_frequency` iterations (from config, typically 10)
        2. At phase transition boundaries (Phase 4→5, start of fine-tuning)
        3. If never initialized (first call)

        Args:
            current_iter: Current training iteration

        Returns:
            bool: True if contact zones should be recomputed
        """
        # Not active before Phase 4 starts (contact zones not used yet)
        if current_iter < self.phase4_start:
            return False

        # Get update frequency from adaptive contacts module
        # Default to every 10 iterations (from adaptive_contact_zones.py)
        update_freq = getattr(self, 'contact_update_frequency', 10)

        # Check if at update boundary
        iterations_since_start = current_iter - self.phase4_start
        is_update_step = (iterations_since_start % update_freq) == 0

        # Force update at phase transitions
        is_phase_transition = (
                current_iter == self.phase4_start or  # Phase 4 start (first contact)
                current_iter == self.phase5_start or  # Phase 5 start (adaptive→enhanced)
                current_iter == self.finetune_start  # Fine-tuning start
        )

        should_update = is_update_step or is_phase_transition

        if should_update:
            logger.debug(
                f"[Phase 5] Scheduler: Updating contact zones at iter {current_iter} "
                f"(update_freq={update_freq}, transition={is_phase_transition})"
            )

        return should_update

    def should_apply_temporal_consistency(self, iteration: int) -> bool:
        """
        Determine if temporal consistency should be applied.

        Temporal consistency is only active during Phase 5 and later.

        Args:
            iteration: Current training iteration

        Returns:
            bool: True if temporal consistency should be computed
        """
        return iteration >= self.phase5_start

    def get_phase_name(self, iteration: int) -> str:
        """
        Get human-readable phase name for logging.

        Args:
            iteration: Current training iteration

        Returns:
            phase_name: String identifier for current phase
        """
        if iteration < self.warmup_iters:
            return "Warm-up"
        elif iteration < self.phase3_start:
            return "Base HOLD"
        elif iteration < self.phase4_start:
            return "Phase 3 (SDS Guidance)"
        elif iteration < self.phase5_start:
            return "Phase 4 (Contact Refinement)"
        elif iteration < self.finetune_start:
            return "Phase 5 (Full Integration)"
        else:
            return "Fine-tuning"

    def is_phase_active(self, phase_name: str, iteration: int) -> bool:
        """
        Check if a specific phase is currently active.

        Args:
            phase_name: Phase identifier ('phase3', 'phase4', 'phase5')
            iteration: Current training iteration

        Returns:
            bool: True if phase is active
        """
        phase_map = {
            'phase3': (self.phase3_start, self.total_iterations),
            'phase4': (self.phase4_start, self.total_iterations),
            'phase5': (self.phase5_start, self.total_iterations),
            'finetune': (self.finetune_start, self.total_iterations)
        }

        if phase_name not in phase_map:
            raise ValueError(f"Unknown phase: {phase_name}")

        start, end = phase_map[phase_name]
        return start <= iteration < end

    def get_scheduler_state(self, iteration: int) -> Dict:
        """
        Get comprehensive scheduler state for monitoring.

        Args:
            iteration: Current training iteration

        Returns:
            state: Dictionary with all scheduler information
        """
        return {
            'iteration': iteration,
            'phase': self.get_phase_name(iteration),
            'loss_weights': self.get_loss_weights(iteration),
            'lr_multiplier': self.get_learning_rate_multiplier(iteration),
            'update_contacts': self.should_update_contact_zones(iteration),
            'apply_temporal': self.should_apply_temporal_consistency(iteration),
            'progress': iteration / self.total_iterations
        }