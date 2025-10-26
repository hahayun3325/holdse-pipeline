# FILE: src/utils/memory_profiler.py (NEW FILE)
# ================================================================
import torch
import gc
from loguru import logger
from collections import defaultdict
import time


class MemoryProfiler:
    """
    Track memory allocations at each stage of training_step.

    Usage:
        profiler = MemoryProfiler()
        profiler.checkpoint("start")
        # ... some code ...
        profiler.checkpoint("after_forward")
        profiler.report()
    """

    def __init__(self):
        self.checkpoints = []
        self.tensor_counts = []

    def checkpoint(self, name):
        """Record memory state at this point."""
        if not torch.cuda.is_available():
            return

        # Force synchronization
        torch.cuda.synchronize()

        # Get memory stats
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2  # ✅ ADD THIS
        cached = reserved - allocated  # ✅ ADD THIS

        # Count live tensors
        tensor_count = sum(1 for obj in gc.get_objects()
                          if torch.is_tensor(obj) and obj.is_cuda)

        self.checkpoints.append({
            'name': name,
            'allocated': allocated,
            'reserved': reserved,  # ✅ ADD THIS
            'cached': cached,  # ✅ ADD THIS
            'tensor_count': tensor_count,
            'timestamp': time.time()
        })

    def report(self):
        """Generate and log memory usage report."""
        if len(self.checkpoints) < 2:
            return

        logger.info("=" * 70)
        logger.info("MEMORY PROFILER REPORT")
        logger.info("=" * 70)

        prev = None
        for cp in self.checkpoints:
            if prev is None:
                logger.info(
                    f"[{cp['name']:20s}] "
                    f"Alloc: {cp['allocated']:7.1f} MB | "
                    f"Tensors: {cp['tensor_count']:5d}"
                )
            else:
                delta_alloc = cp['allocated'] - prev['allocated']
                delta_tensors = cp['tensor_count'] - prev['tensor_count']
                delta_time = (cp['timestamp'] - prev['timestamp']) * 1000

                logger.info(
                    f"[{cp['name']:20s}] "
                    f"Alloc: {cp['allocated']:7.1f} MB ({delta_alloc:+7.1f}) | "
                    f"Tensors: {cp['tensor_count']:5d} ({delta_tensors:+5d}) | "
                    f"Time: {delta_time:6.1f}ms"
                )
            prev = cp

        # Show biggest growth
        max_growth = max((self.checkpoints[i + 1]['allocated'] - self.checkpoints[i]['allocated'],
                          self.checkpoints[i]['name'],
                          self.checkpoints[i + 1]['name'])
                         for i in range(len(self.checkpoints) - 1))

        logger.info("=" * 70)
        logger.info(f"BIGGEST MEMORY GROWTH: {max_growth[0]:.1f} MB")
        logger.info(f"  From: {max_growth[1]}")
        logger.info(f"  To:   {max_growth[2]}")
        logger.info("=" * 70)

    def clear(self):
        """Clear checkpoint history."""
        self.checkpoints.clear()