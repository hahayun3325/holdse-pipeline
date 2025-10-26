#!/bin/bash
# verify_latest_run.sh

cd ~/Projects/holdse/code

echo "========================================="
echo "VERIFYING LATEST TRAINING RUN"
echo "========================================="
echo ""

# Find ALL log files modified today
echo "1. Log files from today:"
find . -maxdepth 1 -name "*.log*" -mtime 0 -printf "%T@ %Tc %p\n" | sort -rn | head -10

echo ""
echo "2. GPU memory logs from today:"
find . -name "gpu_memory_log.txt" -mtime 0 -ls

echo ""
echo "3. Most recent training log:"
LATEST=$(ls -t test_*.log* training_*.log* 2>/dev/null | head -1)
if [ -f "$LATEST" ]; then
    echo "File: $LATEST"
    echo "Size: $(wc -l "$LATEST") lines"
    echo "Modified: $(stat -c %y "$LATEST")"
    echo ""
    echo "First 10 lines:"
    head -10 "$LATEST"
    echo ""
    echo "Last 10 lines:"
    tail -10 "$LATEST"
else
    echo "NO LOG FILES FOUND!"
fi

echo ""
echo "4. Checking if training is running:"
ps aux | grep "train.py" | grep -v grep

echo ""
echo "5. Current GPU status:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
