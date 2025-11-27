#!/bin/bash
# train_two_stage_v2_stage2_continue.sh

cd ~/Projects/holdse/code

echo "========================================================================"
echo "STAGE 2 CONTINUATION: Epochs 21-30"
echo "========================================================================"
echo "  Resuming from: logs/20d1ec1e8/checkpoints/last.ckpt"
echo "  Target epochs: 30 (adding 10 more)"
echo "========================================================================"
echo ""

# ================================================================
# Verification: Stage 2 Checkpoint (20 epochs)
# ================================================================
STAGE2_CKPT="logs/4848a499d/checkpoints/last.ckpt" # Run A checkpoint trained with 16 epochs from finetuning 1 process

if [ ! -f "$STAGE2_CKPT" ]; then
    echo "❌ ERROR: Stage 2 checkpoint not found at $STAGE2_CKPT"
    exit 1
fi

echo "✅ Stage 2 checkpoint verified: $STAGE2_CKPT"

# Check checkpoint epoch
python3 << EOF
import torch
ckpt = torch.load('$STAGE2_CKPT', map_location='cpu')
epoch = ckpt.get('epoch', 'unknown')
global_step = ckpt.get('global_step', 'unknown')
print(f"   Current epoch: {epoch}")
print(f"   Global step: {global_step}")
print(f"   Will continue to epoch: 30")
EOF

echo ""

# ================================================================
# Run Continuation Training
# ================================================================
echo "Continuing Stage 2 training for 10 more epochs..."
echo ""

python train.py \
    --config confs/stage2_tuned_runA.yaml \
    --case hold_bottle1_itw \
    --num_epoch 25 \
    --infer_ckpt "$STAGE2_CKPT" \
    --no-comet \
    --gpu_id 0 \
    --no-pin-memory

# ================================================================
# Post-Training
# ================================================================
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✅ STAGE 2 CONTINUATION COMPLETE!"
    echo "========================================================================"
    echo "  Total epochs: 20"
    echo "  New checkpoint location: Check latest in logs/"
    echo "========================================================================"
else
    echo ""
    echo "❌ Continuation training failed"
    exit 1
fi


#chmod +x scripts/train_two_stage_v2_stage2_continue.sh
#./scripts/train_two_stage_v2_stage2_continue.sh 2>&1 | tee logs/stage2_tuned_runA_continuation_20to25_$(date +%Y%m%d_%H%M%S).log
# tail -f logs/stage2_tuned_runA_continuation_20to25_*.log | grep --line-buffered "Avg loss"
#tail -f logs/stage2_tuned_runA_continuation_20to25_*.log | grep -E "Stage|Checkpoint|✅|❌|ERROR"
# watch -n 5 nvidia-smi

## Use HOLD dataset (hold_bottle1_itw/build/)
## Enable GHOP SDS loss via config
#python train.py \
#    --case hold_bottle1_itw \
#    --config confs/ghop_stage2_hold_MC1_ho3d.yaml \
#    --load_ckpt logs/stage1.ckpt \
#    --num_epoch 30
#    # ❌ NO --use_ghop flag
#
## Result:
## - Dataset: data/hold_bottle1_itw/build/ (295 frames, HOLD ImageDataset)
## - GHOP SDS: Enabled (phase3.enabled: true in config)
## - Training: Stage 2 refinement with SDS guidance

## Use GHOP dataset (ghop_bottle_1/ghop_data → HOI4D)
## Enable GHOP SDS loss via config
#python train.py \
#    --case ghop_bottle_1 \
#    --config confs/ghop_stage2_hold_MC1_ho3d.yaml \
#    --use_ghop \  # ← Selects GHOP dataset
#    --num_epoch 30
#
## Result:
## - Dataset: data/ghop_bottle_1/ghop_data/ (71 frames, HOI4D)
## - GHOP SDS: Enabled (phase3.enabled: true)
## - Training: Video sequence with temporal consistency