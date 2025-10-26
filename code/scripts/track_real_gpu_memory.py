#!/usr/bin/env python3
"""
Track REAL GPU memory usage using nvidia-smi during training.
This bypasses PyTorch's tracking to see actual GPU VRAM usage.

# Monitor GPU memory
python scripts/track_real_gpu_memory.py 5 > gpu_monitor_phase5_disabled.log 2>&1 &
GPU_PID=$!

# After test completes
kill $GPU_PID

# Check results
grep "Epoch.*100%" test_defrag_fix_*.log
tail gpu_memory_log.txt
"""

import subprocess
import time
import os
import sys

def get_gpu_memory_detailed():
    """Get both nvidia-smi and PyTorch memory"""

    # nvidia-smi memory
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.free,memory.total',
             '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )
        used, free, total = map(int, result.strip().split(','))
    except Exception as e:
        return {'error': str(e)}

    # PyTorch memory (if available)
    pytorch_mem = {}
    try:
        if torch.cuda.is_available():
            pytorch_mem['allocated'] = torch.cuda.memory_allocated() / 1024**2  # MB
            pytorch_mem['reserved'] = torch.cuda.memory_reserved() / 1024**2    # MB
            pytorch_mem['cache'] = pytorch_mem['reserved'] - pytorch_mem['allocated']
    except:
        pytorch_mem = {'allocated': -1, 'reserved': -1, 'cache': -1}

    return {
        'nvidia_used': used,
        'nvidia_free': free,
        'nvidia_total': total,
        'pytorch_allocated': pytorch_mem.get('allocated', -1),
        'pytorch_reserved': pytorch_mem.get('reserved', -1),
        'pytorch_cache': pytorch_mem.get('cache', -1),
        'discrepancy': used - pytorch_mem.get('reserved', 0)
    }


def monitor_training_detailed(log_file='gpu_memory_detailed_log.txt', interval=5):
    """Monitor GPU memory with PyTorch comparison"""
    print(f"Monitoring GPU memory every {interval} seconds...")
    print(f"Logging to: {log_file}")

    with open(log_file, 'w') as f:
        # Enhanced CSV header
        f.write("timestamp,nvidia_used_mb,nvidia_free_mb,nvidia_total_mb,"
                "pytorch_allocated_mb,pytorch_reserved_mb,pytorch_cache_mb,"
                "discrepancy_mb\n")

        while True:
            mem = get_gpu_memory_detailed()
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

            if 'error' in mem:
                print(f"[{timestamp}] ERROR: {mem['error']}")
            else:
                line = (f"{timestamp},{mem['nvidia_used']},{mem['nvidia_free']},"
                        f"{mem['nvidia_total']},{mem['pytorch_allocated']:.1f},"
                        f"{mem['pytorch_reserved']:.1f},{mem['pytorch_cache']:.1f},"
                        f"{mem['discrepancy']:.1f}\n")
                f.write(line)
                f.flush()

                # Enhanced output
                print(f"[{timestamp}] "
                      f"nvidia-smi: {mem['nvidia_used']:>6} MB | "
                      f"PyTorch reserved: {mem['pytorch_reserved']:>6.1f} MB | "
                      f"Discrepancy: {mem['discrepancy']:>6.1f} MB")

                # Warning if discrepancy is large
                if mem['discrepancy'] > 1000:  # > 1 GB
                    print(f"              ⚠️  WARNING: {mem['discrepancy']:.1f} MB "
                          f"allocated outside PyTorch!")

            time.sleep(interval)


if __name__ == '__main__':
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    monitor_training_detailed(interval=interval)