#!/bin/bash
# ================================================================
# HOLDSE Stage 1 Training - MC1 Sequence (HO3D)
# Full network architecture, 200 epochs, RGB-only baseline
# ================================================================

set -e

cd ~/Projects/holdse/code

echo "========================================================================"
echo "HOLDSE Stage 1 Training - MC1 (HO3D Dataset)"
echo "========================================================================"
echo "Architecture: Full HOLD (dims=[256,256,256,256,256,256,256,256])"
echo "Duration: 200 epochs (~18-24 hours)"
echo "Dataset: hold_MC1_ho3d (144 frames)"
echo "Phases: 1-2 only (RGB + Eikonal)"
echo "========================================================================"

CONFIG="./confs/stage1_hold_MC1_ho3d_8layer_implicit_official_match_fixed.yaml"

# Verify config exists
if [ ! -f "$CONFIG" ]; then
    echo "❌ Config not found: $CONFIG"
    echo "Creating from ghop_stage1_rgb_only.yaml..."
    cp ./confs/ghop_stage1_rgb_only.yaml "$CONFIG"
fi

echo ""
echo "Starting Stage 1 training for MC1..."
echo ""

# Train for 200 epochs
python train.py \
    --config "$CONFIG" \
    --case "hold_MC1_ho3d" \
    --num_epoch 20 \
    --no-comet \
    --gpu_id 0 \
    --no-pin-memory

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Stage 1 training complete!"

    # Find final checkpoint
    FINAL_CKPT=$(find logs -name "last.ckpt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    echo "Final checkpoint: $FINAL_CKPT"
    
    echo ""
    echo "Next step: Train Stage 2 with GHOP SDS"
else
    echo "❌ Training failed!"
    exit 1
fi

echo "✅ STAGE 1 COMPLETE"
EOF

## Make script executable
#chmod +x scripts/train_MC1_stage1.sh
#
## Run full two-stage training
# ./scripts/train_MC1_stage1.sh 2>&1 | tee logs/stage1_1to20_hold_MC1_ho3d_official_match_fixed_mano_$(date +%Y%m%d_%H%M%S).log
# tail -f logs/stage1_1to20_hold_MC1_ho3d_official_match_fixed_mano_*.lo | grep --line-buffered "Avg loss"
## Monitor in another terminal
#tail -f production_training.log | grep -E "Stage|Checkpoint|✅|❌|ERROR"