#!/bin/bash

# Stage 3: Full Pipeline (Boundary-Based Mode)
# No progressive scheduler - uses fixed phase boundaries

set -e

STAGE2_CKPT="logs/ac71c88b7/checkpoints/last.ckpt" # Stage 3 Checkpoint 40-epoch(full SDS) on Official Checkpoint
SEQ_NAME="hold_MC1_ho3d"
STAGE3_CONFIG="confs/stage3_hold_MC1_ho3d_sds_from_official.yaml"

echo "=================================================="
echo "STAGE 3: Boundary-Based Training (No Scheduler)"
echo "=================================================="
echo "Stage 2 checkpoint: $STAGE2_CKPT"
echo "Config: $STAGE3_CONFIG"
echo

# Validation (same as above)
if [ ! -f "$STAGE2_CKPT" ]; then
    echo "❌ ERROR: Stage 2 checkpoint not found!"
    exit 1
fi
echo "✓ Stage 2 checkpoint found"

# Run training with fixed boundaries
python train.py \
    --config $STAGE3_CONFIG \
    --infer_ckpt $STAGE2_CKPT \
    --case $SEQ_NAME \
    --no-comet \
    --gpu_id 0 \
    --num_epoch 50 \
    --no-pin-memory

echo "✅ STAGE 3 COMPLETE (boundary mode)"

# chmod +x scripts/train_MC1_stage3_continue.sh
# ./scripts/train_MC1_stage3_continue.sh 2>&1 | tee logs/stage3_40to50_hold_MC1_ho3d_from_official_$(date +%Y%m%d_%H%M%S).log