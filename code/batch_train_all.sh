#!/bin/bash
# Batch training script for all HOLD sequences

cd ~/Projects/holdse/code

SEQUENCES=($(ls -1 ./data/ | grep "^hold_"))

for SEQ in "${SEQUENCES[@]}"; do
    echo "======================================================================"
    echo "Training on: $SEQ"
    echo "======================================================================"

    python sanity_train.py \
        --case "$SEQ" \
        --shape_init 75268d864 \
        --gpu_id 0 \
        2>&1 | tee logs/train_${SEQ}.log

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Training completed: $SEQ"
    else
        echo "❌ Training failed: $SEQ (exit code: $EXIT_CODE)"
    fi

    echo ""
done

echo "======================================================================"
echo "Batch training complete!"
echo "======================================================================"
