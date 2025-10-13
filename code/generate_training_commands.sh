#!/bin/bash
# File: ~/Projects/holdse/code/generate_training_commands.sh

echo "======================================================================"
echo "HOLDSE Training Command Generator"
echo "======================================================================"

cd ~/Projects/holdse/code

# List available sequences
echo ""
echo "Available sequences:"
echo ""

SEQUENCES=($(ls -1 ./data/ | grep "^hold_"))

for i in "${!SEQUENCES[@]}"; do
    SEQ="${SEQUENCES[$i]}"
    DATA_FILE="./data/$SEQ/build/data.npy"

    if [ -f "$DATA_FILE" ]; then
        # Get frame count
        FRAME_INFO=$(python -c "
import numpy as np
data = np.load('$DATA_FILE', allow_pickle=True).item()
cameras = data.get('cameras', {})
frame_count = len([k for k in cameras.keys() if k.startswith('world_mat_')])
seq_name = data.get('seq_name', 'N/A')
print(f'{frame_count} frames - {seq_name}')
" 2>/dev/null)

        printf "[%2d] %-25s %s\n" "$((i+1))" "$SEQ" "$FRAME_INFO"
    else
        echo "[$((i+1))] $SEQ - ⚠️  data.npy not found"
    fi
done

echo ""
echo "======================================================================"
echo "TRAINING COMMANDS"
echo "======================================================================"
echo ""

# Generate commands for common sequences
COMMON_SEQS=("hold_mug1_itw" "hold_bottle1_itw" "hold_MC1_ho3d" "hold_toycar1_itw")

for SEQ in "${COMMON_SEQS[@]}"; do
    if [ -d "./data/$SEQ" ]; then
        echo "# Train on $SEQ:"
        echo "python sanity_train.py --case $SEQ --shape_init 75268d864 --gpu_id 0"
        echo ""
    fi
done

echo "======================================================================"
echo "BATCH TRAINING (all sequences)"
echo "======================================================================"
echo ""

cat > batch_train_all.sh <<'EOF'
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
EOF

chmod +x batch_train_all.sh
echo "Generated: batch_train_all.sh"
echo "Run with: bash batch_train_all.sh"