#!/bin/bash
# ================================================================
# HOLDSE PRODUCTION BASELINE - Fresh Start
# Full network architecture, 50 epochs, HOLD-only (no Phase 3-5)
# ================================================================

set -e

cd ~/Projects/holdse/code

echo "========================================================================"
echo "PRODUCTION BASELINE TRAINING - Fresh Start"
echo "========================================================================"
echo "Architecture: Full HOLD (dims=[256,256,256,256])"
echo "Duration: 50 epochs"
echo "Dataset: TempoDataset (RGB only)"
echo "========================================================================"

CONFIG="./confs/ghop_stage1_rgb_only.yaml"

# Verify config exists
if [ ! -f "$CONFIG" ]; then
    echo "❌ Config not found: $CONFIG"
    exit 1
fi

# ✅ CRITICAL: Do NOT try to resume from old architecture
echo ""
echo "Starting FRESH (ignoring old small-network checkpoints)"
echo ""

# Train for 200 epochs
python train.py \
    --config "$CONFIG" \
    --case "hold_bottle2_itw" \
    --num_epoch 100 \
    --no-comet \
    --gpu_id 0 \
    --no-pin-memory 2>&1 | tee logs/production_baseline_full_200epochs.log

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Production baseline training complete!"

    # Find final checkpoint
    FINAL_CKPT=$(find logs -name "last.ckpt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    echo "Final checkpoint: $FINAL_CKPT"

#    # Render validation
#    echo ""
#    echo "Rendering validation samples..."
#    python scripts/render_validation_with_arg.py \
#        --checkpoint "$FINAL_CKPT" \
#        --config "$CONFIG" \
#        --output "logs/production_renders/" \
#        --num_frames 5
else
    echo "❌ Training failed!"
    exit 1
fi

echo "✅ STAGE 1 VERIFICATION COMPLETE"
echo ""

## Make script executable
#chmod +x scripts/train_two_stage_v2_stage1.sh
#
## Run full two-stage training
# ./scripts/train_two_stage_v2_stage1.sh 2>&1 | tee logs/full_training_hold_bottle1_itw_stage1_1to100_$(date +%Y%m%d_%H%M%S).log
# tail -f logs/full_training_hold_bottle1_itw_stage1_1to100_*.log | grep --line-buffered "Avg loss"
## Monitor in another terminal
#tail -f production_training.log | grep -E "Stage|Checkpoint|✅|❌|ERROR"