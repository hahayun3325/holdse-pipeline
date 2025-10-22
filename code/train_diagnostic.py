#!/usr/bin/env python3
import torch
import subprocess
import time
import sys


# Monitor memory usage
def monitor_memory():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    return int(result.stdout.strip())


# Log memory to file
log_file = open('memory_profile.csv', 'w')
log_file.write('timestamp,step,memory_mb\n')

# Start training in background
import subprocess

training_process = subprocess.Popen([
    'python', 'train.py',
    '--config', 'confs/ghop_production_32dim_20251021_064416.yaml',
    '--case', 'ghop_bottle_1',
    '--use_ghop',
    '--gpu_id', '0',
    '--num_epoch', '100'
], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

step = 0
start_time = time.time()

print("Monitoring memory usage...")

while training_process.poll() is None:
    try:
        # Read output
        line = training_process.stdout.readline()
        if line:
            print(line.strip())

            # Parse step number if present
            if 'Epoch' in line and '/' in line:
                step += 1

        # Log memory every 5 seconds
        if time.time() - start_time > 5:
            memory = monitor_memory()
            timestamp = time.time()
            log_file.write(f'{timestamp},{step},{memory}\n')
            log_file.flush()

            print(f"[Monitor] Step ~{step}, Memory: {memory}MB")

            start_time = time.time()

    except KeyboardInterrupt:
        training_process.terminate()
        break

log_file.close()
print(f"Memory profile saved to memory_profile.csv")