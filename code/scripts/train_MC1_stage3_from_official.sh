#!/bin/bash

# Stage 3: Full Pipeline (Boundary-Based Mode)
# No progressive scheduler - uses fixed phase boundaries

set -e

#STAGE2_CKPT="/home/fredcui/Projects/hold/code/logs/cb20a1702/checkpoints/last.ckpt" # Official hold_MC1_ho3d Checkpoint
#STAGE2_CKPT="/home/fredcui/Projects/hold/code/logs/b7c26b798/checkpoints/last.ckpt" # Official hold_SM4_ho3d Checkpoint
#STAGE2_CKPT="/home/fredcui/Projects/hold/code/logs/81a2bea9a/checkpoints/last.ckpt" # Official hold_ABF12_ho3d Checkpoint
#STAGE2_CKPT="/home/fredcui/Projects/hold/code/logs/db6508d7f/checkpoints/last.ckpt" # Official hold_GSF12_ho3d Checkpoint
#STAGE2_CKPT="/home/fredcui/Projects/hold/code/logs/76fbd4d33/checkpoints/last.ckpt" # Official hold_GSF13_ho3d Checkpoint
#STAGE2_CKPT="/home/fredcui/Projects/hold/code/logs/20b7fc070/checkpoints/last.ckpt" # Official hold_ABF14_ho3d Checkpoint
STAGE2_CKPT="/home/fredcui/Projects/hold/code/logs/00bc6dc5e/checkpoints/last.ckpt" # Official hold_GPMF12_ho3d Checkpoint
#STAGE2_CKPT="/home/fredcui/Projects/hold/code/logs/64834e9bb/checkpoints/last.ckpt" # Official hold_GPMF14_ho3d Checkpoint todo existing ckpt trained from beginning
#STAGE2_CKPT="/home/fredcui/Projects/hold/code/logs/c8d39e1aa/checkpoints/last.ckpt" # Official hold_MC4_ho3d Checkpoint
#STAGE2_CKPT="/home/fredcui/Projects/hold/code/logs/fd873a597/checkpoints/last.ckpt" # Official hold_MDF12_ho3d Checkpoint

#STAGE2_CKPT="logs/afb17c622/checkpoints/last.ckpt" # Stage 2 Checkpoint 70-epoch(full SDS) on Official Checkpoint
#STAGE2_CKPT="logs/40de820f3/checkpoints/last.ckpt" # Train from beginning 30-epoch ckpt
#STAGE2_CKPT="logs/2c3a14d25/checkpoints/last.ckpt" # Train from beginning 60-epoch ckpt
#STAGE2_CKPT="logs/176872f9f/checkpoints/last.ckpt" # Train from beginning 90-epoch ckpt
#STAGE2_CKPT="logs/e97e1df6d/checkpoints/last.ckpt" # hold_SM4_ho3d Train from beginning 30-epoch ckpt

#SEQ_NAME="hold_MC1_ho3d" # Trained ckpt: 694416964
#SEQ_NAME="hold_SM4_ho3d" # Trained ckpt: 71f17bb52
#SEQ_NAME="hold_GPMF14_ho3d" # Trained ckpt: 45b9c9ed4(from beginning)
#SEQ_NAME="hold_SM2_ho3d" # Trained ckpt:
#SEQ_NAME="hold_SMu40_ho3d" # Trained ckpt:
#SEQ_NAME="hold_BB13_ho3d" # Trained ckpt:
#SEQ_NAME="hold_ShSu12_ho3d" # Trained ckpt:
#SEQ_NAME="hold_BB12_ho3d" # Trained ckpt:
#SEQ_NAME="hold_ABF12_ho3d" # Trained ckpt: 9c60aa891
#SEQ_NAME="hold_GSF12_ho3d" # Trained ckpt: 44521f421
#SEQ_NAME="hold_GSF13_ho3d" # Trained ckpt: a09881c64 stage3_1to40_hold_GSF13_ho3d_object_phase3_official_20260125_103919.log
#SEQ_NAME="hold_ABF14_ho3d" # Trained ckpt: aa43c543e
SEQ_NAME="hold_GPMF12_ho3d" # Trained ckpt: 2ceb7bc79
#SEQ_NAME="hold_MC4_ho3d" # Trained ckpt: todo
#SEQ_NAME="hold_MDF12_ho3d" # Trained ckpt: todo
#STAGE3_CONFIG="confs/stage3_hold_MC1_ho3d_sds_from_official.yaml"
STAGE3_CONFIG="confs/3phases_hold_MC1_ho3d_phase3.yaml"
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

## Run training with fixed boundaries
#python train.py \
#    --config $STAGE3_CONFIG \
#    --case $SEQ_NAME \
#    --no-comet \
#    --gpu_id 0 \
#    --num_epoch 30 \
#    --no-pin-memory

# Run training with fixed boundaries
python train.py \
    --config $STAGE3_CONFIG \
    --load_ckpt $STAGE2_CKPT \
    --case $SEQ_NAME \
    --no-comet \
    --gpu_id 0 \
    --num_epoch 40 \
    --no-pin-memory

echo "✅ STAGE 3 COMPLETE (boundary mode)"

# chmod +x scripts/train_MC1_stage3_from_official.sh
# ./scripts/train_MC1_stage3_from_official.sh 2>&1 | tee logs/stage3_1to40_hold_GPMF12_ho3d_object_phase3_official_$(date +%Y%m%d_%H%M%S).log