#!/bin/bash

# Stage 2: SDS-guided optimization starting from official Stage 1 checkpoint
# Uses official HOLD checkpoint for MC1 sequence instead of our Stage 1

set -e

#OFFICIAL_CKPT="/home/fredcui/Projects/hold/code/logs/cb20a1702/checkpoints/last.ckpt"
#SEQ_NAME="hold_MC1_ho3d"
OFFICIAL_CKPT="/home/fredcui/Projects/hold/code/logs/00bc6dc5e/checkpoints/last.ckpt"
SEQ_NAME="hold_GPMF12_ho3d"
STAGE2_CONFIG="confs/stage2_hold_MC1_ho3d_sds_from_official.yaml"

echo "=================================================="
echo "STAGE 2: SDS Training from Official Checkpoint"
echo "=================================================="
echo "Official checkpoint: $OFFICIAL_CKPT"
echo "Sequence: $SEQ_NAME"
echo "Config: $STAGE2_CONFIG"
echo

# Verify official checkpoint exists
if [ ! -f "$OFFICIAL_CKPT" ]; then
    echo "❌ ERROR: Official checkpoint not found!"
    echo "   Expected: $OFFICIAL_CKPT"
    exit 1
fi

echo "✓ Official checkpoint found"
echo

# Run Stage 2 training
python train.py \
    --config $STAGE2_CONFIG \
    --load_ckpt $OFFICIAL_CKPT \
    --case hold_GPMF12_ho3d \
    --no-comet \
    --gpu_id 0 \
    --num_epoch 10 \
    --no-pin-memory

echo
echo "✅ STAGE 2 COMPLETE (from official checkpoint)"

# ./scripts/train_MC1_stage2_from_official.sh 2>&1 | tee logs/stage2_1to1_hold_MC1_ho3d_from_official_$(date +%Y%m%d_%H%M%S).log
