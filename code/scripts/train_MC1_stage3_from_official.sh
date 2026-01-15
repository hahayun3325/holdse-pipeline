#!/bin/bash

# Stage 3: Full Pipeline (Boundary-Based Mode)
# No progressive scheduler - uses fixed phase boundaries

set -e

#STAGE2_CKPT="/home/fredcui/Projects/hold/code/logs/cb20a1702/checkpoints/last.ckpt" # Official Checkpoint
#STAGE2_CKPT="logs/afb17c622/checkpoints/last.ckpt" # Stage 2 Checkpoint 70-epoch(full SDS) on Official Checkpoint
#STAGE2_CKPT="logs/40de820f3/checkpoints/last.ckpt" # Train from beginning 30-epoch ckpt
#STAGE2_CKPT="logs/2c3a14d25/checkpoints/last.ckpt" # Train from beginning 60-epoch ckpt
#STAGE2_CKPT="logs/176872f9f/checkpoints/last.ckpt" # Train from beginning 90-epoch ckpt
STAGE2_CKPT="logs/e97e1df6d/checkpoints/last.ckpt" # hold_SM4_ho3d Train from beginning 30-epoch ckpt

#SEQ_NAME="hold_MC1_ho3d"
#SEQ_NAME="hold_SM4_ho3d"
#SEQ_NAME="hold_GPMF14_ho3d"
#SEQ_NAME="hold_SM2_ho3d"
#SEQ_NAME="hold_SMu40_ho3d"
#SEQ_NAME="hold_BB13_ho3d"
SEQ_NAME="hold_ShSu12_ho3d"
STAGE3_CONFIG="confs/stage3_hold_MC1_ho3d_sds_from_official.yaml"
#STAGE3_CONFIG="confs/stage3_hold_MC1_ho3d_sds_test_1epoch.yaml"
#STAGE3_CONFIG="confs/stage3_hold_MC1_ho3d_sds_test_10epoch.yaml"
#STAGE3_CONFIG="confs/stage3_hold_MC1_ho3d_sds_test_2epoch_verify.yaml"

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
    --case $SEQ_NAME \
    --no-comet \
    --gpu_id 0 \
    --num_epoch 30 \
    --no-pin-memory

## Run training with fixed boundaries
#python train.py \
#    --config $STAGE3_CONFIG \
#    --load_ckpt $STAGE2_CKPT \
#    --case $SEQ_NAME \
#    --no-comet \
#    --gpu_id 0 \
#    --num_epoch 30 \
#    --no-pin-memory

echo "✅ STAGE 3 COMPLETE (boundary mode)"

# chmod +x scripts/train_MC1_stage3_from_official.sh
# ./scripts/train_MC1_stage3_from_official.sh 2>&1 | tee logs/stage3_1to30_hold_ShSu12_ho3d_refinedPhases_2r1_beginning_$(date +%Y%m%d_%H%M%S).log