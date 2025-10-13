#!/bin/bash
# Monitor sanity training progress

echo "==================================================================="
echo "HOISE Sanity Training Monitor"
echo "Environment: ghop_hold_integrated"
echo "==================================================================="

# Check if training is running
if pgrep -f "sanity_train.py" > /dev/null; then
    echo "✓ Sanity training is running"

    # Show GPU usage
    echo ""
    echo "--- GPU Usage ---"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
        awk -F',' '{printf "GPU %s (%s): %s%% GPU, %s/%s MB\n", $1, $2, $3, $4, $5}'

    # Show recent log output
    if [ -f "sanity_train_output.txt" ]; then
        echo ""
        echo "--- Last 10 Log Lines ---"
        tail -n 10 sanity_train_output.txt
    fi

    # Show checkpoint status
    if [ -d "logs" ]; then
        echo ""
        echo "--- Checkpoints ---"
        find logs -name "*.ckpt" -type f -printf "%T@ %p\n" | sort -n | tail -3 | \
            awk '{print "  " $2}'
    fi
else
    echo "✗ No sanity training process found"
fi

echo "==================================================================="