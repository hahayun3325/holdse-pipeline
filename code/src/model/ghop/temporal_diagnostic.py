"""
Temporal Memory Diagnostic Tool for Phase 5

Instruments temporal_consistency.py to track memory usage and detect leaks.
"""

import torch
import gc
from typing import Dict, Any
from loguru import logger


class TemporalMemoryDiagnostic:
    """Diagnostic tool for temporal consistency memory tracking"""

    def __init__(self):
        self.snapshots = []
        self.baseline_memory = 0

    def get_temporal_module_memory(self, temporal_module) -> Dict[str, Any]:
        """
        Analyze memory usage of temporal consistency module.

        Returns:
            Dictionary with detailed memory breakdown
        """
        memory_info = {
            'total_mb': 0,
            'sequence_count': 0,
            'frame_count': 0,
            'avg_frames_per_sequence': 0,
            'leak_detected': False,
            'leak_reasons': []
        }

        if temporal_module is None:
            return memory_info

        # Analyze pose_history
        if hasattr(temporal_module, 'pose_history'):
            pose_hist = temporal_module.pose_history
            memory_info['sequence_count'] = len(pose_hist)

            total_frames = 0
            total_size_bytes = 0

            for seq_id, seq_deque in pose_hist.items():
                frame_count = len(seq_deque)
                total_frames += frame_count

                # Calculate memory for each frame
                for frame in seq_deque:
                    if isinstance(frame, torch.Tensor):
                        total_size_bytes += frame.element_size() * frame.nelement()

            memory_info['frame_count'] = total_frames
            memory_info['total_mb'] = total_size_bytes / (1024 ** 2)

            if memory_info['sequence_count'] > 0:
                memory_info['avg_frames_per_sequence'] = (
                        total_frames / memory_info['sequence_count']
                )

        # Analyze camera_history
        if hasattr(temporal_module, 'camera_history'):
            camera_hist = temporal_module.camera_history

            for seq_id, seq_deque in camera_hist.items():
                for frame in seq_deque:
                    if isinstance(frame, torch.Tensor):
                        total_size_bytes += frame.element_size() * frame.nelement()

            memory_info['total_mb'] = total_size_bytes / (1024 ** 2)

        # LEAK DETECTION LOGIC
        window_size = getattr(temporal_module, 'window_size', 5)
        max_sequences = getattr(temporal_module, 'max_sequences', 100)

        # Check 1: Too many sequences
        if memory_info['sequence_count'] > max_sequences * 1.5:
            memory_info['leak_detected'] = True
            memory_info['leak_reasons'].append(
                f"Sequence count ({memory_info['sequence_count']}) "
                f"exceeds limit ({max_sequences})"
            )

        # Check 2: Frames per sequence exceeds window
        if memory_info['avg_frames_per_sequence'] > window_size * 2:
            memory_info['leak_detected'] = True
            memory_info['leak_reasons'].append(
                f"Avg frames/sequence ({memory_info['avg_frames_per_sequence']:.1f}) "
                f"exceeds window_size ({window_size})"
            )

        # Check 3: Total memory excessive
        expected_memory_mb = memory_info['sequence_count'] * window_size * 0.5
        if memory_info['total_mb'] > expected_memory_mb * 3:
            memory_info['leak_detected'] = True
            memory_info['leak_reasons'].append(
                f"Total memory ({memory_info['total_mb']:.1f} MB) "
                f"exceeds expected ({expected_memory_mb:.1f} MB)"
            )

        return memory_info

    def log_temporal_state(
            self,
            temporal_module,
            epoch: int,
            step: int = 0,
            tag: str = ""
    ) -> Dict[str, Any]:
        """Log detailed temporal state with leak detection"""
        mem_info = self.get_temporal_module_memory(temporal_module)

        logger.info(f"[TemporalDiagnostic] Epoch {epoch} Step {step} {tag}")
        logger.info(f"  Sequences: {mem_info['sequence_count']}")
        logger.info(f"  Total frames: {mem_info['frame_count']}")
        logger.info(f"  Avg frames/seq: {mem_info['avg_frames_per_sequence']:.1f}")
        logger.info(f"  Memory: {mem_info['total_mb']:.2f} MB")

        if mem_info['leak_detected']:
            logger.error("  ❌ MEMORY LEAK DETECTED!")
            for reason in mem_info['leak_reasons']:
                logger.error(f"     - {reason}")
        else:
            logger.info("  ✅ Memory bounded correctly")

        self.snapshots.append({
            'epoch': epoch,
            'step': step,
            'tag': tag,
            'memory_info': mem_info
        })

        return mem_info

    def generate_leak_report(self) -> str:
        """
        Generate comprehensive leak analysis report.

        Analyzes memory growth across snapshots and projects OOM timing.
        """
        if len(self.snapshots) < 2:
            return "Insufficient data for leak analysis (need at least 2 snapshots)"

        report = []
        report.append("=" * 70)
        report.append("TEMPORAL MEMORY LEAK ANALYSIS REPORT")
        report.append("=" * 70)
        report.append("")

        # Access full snapshot objects
        first_snapshot = self.snapshots[0]
        last_snapshot = self.snapshots[-1]

        # Extract memory info from snapshots
        first_mem = first_snapshot['memory_info']
        last_mem = last_snapshot['memory_info']

        # Calculate growth
        memory_growth = last_mem['total_mb'] - first_mem['total_mb']
        frame_growth = last_mem['frame_count'] - first_mem['frame_count']
        seq_growth = last_mem['sequence_count'] - first_mem['sequence_count']

        # Report initial and final state
        report.append(f"Memory Growth Analysis:")
        report.append(f"  Initial State:")
        report.append(f"    Memory:    {first_mem['total_mb']:.2f} MB")
        report.append(f"    Frames:    {first_mem['frame_count']}")
        report.append(f"    Sequences: {first_mem['sequence_count']}")
        report.append(f"")
        report.append(f"  Final State:")
        report.append(f"    Memory:    {last_mem['total_mb']:.2f} MB")
        report.append(f"    Frames:    {last_mem['frame_count']}")
        report.append(f"    Sequences: {last_mem['sequence_count']}")
        report.append(f"")
        report.append(f"  Growth:")
        report.append(f"    Memory:    +{memory_growth:.2f} MB")
        report.append(f"    Frames:    +{frame_growth}")
        report.append(f"    Sequences: +{seq_growth}")
        report.append("")

        # Calculate growth rate
        first_epoch = first_snapshot.get('epoch', 0)
        last_epoch = last_snapshot.get('epoch', 0)
        epochs_elapsed = last_epoch - first_epoch

        if epochs_elapsed > 0:
            memory_per_epoch = memory_growth / epochs_elapsed
            frames_per_epoch = frame_growth / epochs_elapsed

            report.append(f"Growth Rate:")
            report.append(f"  Epochs analyzed: {first_epoch} → {last_epoch} ({epochs_elapsed} epochs)")
            report.append(f"  Memory per epoch: +{memory_per_epoch:.2f} MB/epoch")
            report.append(f"  Frames per epoch: +{frames_per_epoch:.1f} frames/epoch")
            report.append("")

            # Memory leak detection
            if memory_per_epoch > 1.0:
                # Significant growth detected
                epochs_to_oom = (20000 - last_mem['total_mb']) / memory_per_epoch
                report.append(f"⚠️  LINEAR MEMORY GROWTH DETECTED!")
                report.append(f"  Current trajectory will cause OOM at epoch: {last_epoch + int(epochs_to_oom)}")
                report.append(f"  Action required: Investigate memory leak")
                report.append("")
            elif memory_per_epoch > 0.1:
                report.append(f"⚠️  Slow memory growth detected ({memory_per_epoch:.2f} MB/epoch)")
                report.append(f"  Monitor closely - may indicate slow leak")
                report.append("")
            else:
                report.append(f"✅ Memory is bounded (growth: {memory_per_epoch:.2f} MB/epoch)")
                report.append(f"  No memory leak detected")
                report.append("")

            # Frame accumulation check
            avg_frames_first = first_mem.get('avg_frames_per_sequence', 0)
            avg_frames_last = last_mem.get('avg_frames_per_sequence', 0)

            if avg_frames_last > avg_frames_first * 2 and avg_frames_last > 5:
                report.append(f"⚠️  Frame accumulation detected!")
                report.append(f"  Avg frames/sequence: {avg_frames_first:.1f} → {avg_frames_last:.1f}")
                report.append(f"  Expected: ≤ window_size (typically 3-5)")
                report.append("")
        else:
            report.append(f"Note: All snapshots from same epoch - insufficient data for rate analysis")
            report.append("")

        # Leak status summary
        leak_detected = last_mem.get('leak_detected', False)
        if leak_detected:
            report.append(f"❌ MEMORY LEAK STATUS: DETECTED")
            leak_reasons = last_mem.get('leak_reasons', [])
            if leak_reasons:
                report.append(f"  Reasons:")
                for reason in leak_reasons:
                    report.append(f"    - {reason}")
        else:
            report.append(f"✅ MEMORY LEAK STATUS: NONE DETECTED")

        report.append("")
        report.append("=" * 70)
        return "\n".join(report)