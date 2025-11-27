#!/bin/bash
# train_two_stage_v2_stage2_with_unified_ghop.sh

cd ~/Projects/holdse/code

echo "========================================================================"
echo "STAGE 2: GHOP SDS Temporal Refinement (FIXED)"
echo "========================================================================"
echo "  Stage 1 Checkpoint: logs/140dc5c18/checkpoints/last.ckpt"
echo "  GHOP Checkpoint: checkpoints/ghop/last.ckpt"
echo "  Config: confs/ghop_stage2_hold_MC1_ho3d.yaml"
echo "  Phases: RGB + Phase 3 (GHOP SDS)"
echo "========================================================================"
echo ""

# ================================================================
# USE OFFICIAL CHECKPOINT
# ================================================================
STAGE1_CKPT="/home/fredcui/Projects/hold/code/logs/cb20a1702/checkpoints/last.ckpt"

if [ ! -f "$STAGE1_CKPT" ]; then
    echo "❌ ERROR: Official checkpoint not found at $STAGE1_CKPT"
    exit 1
fi

echo "✅ Using official HOLD checkpoint: $STAGE1_CKPT"
echo "   Epoch: 200"
echo "   Global step: 80,000"
echo ""

# ================================================================
# USE MATCHING CONFIGURATION
# ================================================================
CONFIG="confs/ghop_stage2_hold_MC1_ho3d_cb20a1702.yaml"

if [ ! -f "$CONFIG" ]; then
    echo "❌ ERROR: Official-compatible config not found at $CONFIG"
    echo "   Please create config that matches official architecture"
    exit 1
fi

echo "✅ Using official-compatible config: $CONFIG"
echo ""

# ================================================================
# Run Stage 2 Training
# ================================================================
python train.py \
    --config "$CONFIG" \
    --case hold_MC1_ho3d \
    --num_epoch 1 \
    --load_ckpt "$STAGE1_CKPT" \
    --no-comet \
    --gpu_id 0 \
    --no-pin-memory

# ================================================================
# Post-Training: Save Final Checkpoint
# ================================================================
if [ $? -eq 0 ]; then
    # Find most recent checkpoint
    STAGE2_CKPT=$(find logs -name "last.ckpt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

    # Copy to named location
    cp "$STAGE2_CKPT" logs/stage2_final.ckpt

    echo ""
    echo "========================================================================"
    echo "✅ STAGE 2 COMPLETE!"
    echo "========================================================================"
    echo "  Final checkpoint: logs/stage2_final.ckpt"
    echo "  Original location: $STAGE2_CKPT"
    echo ""
    echo "Next steps:"
    echo "  1. Render validation: python scripts/render_validation_with_arg.py --checkpoint logs/stage2_final.ckpt"
    echo "  2. Compare with Stage 1 renders"
    echo "  3. Proceed to Stage 3 (if needed)"
    echo "========================================================================"
else
    echo ""
    echo "❌ Stage 2 training failed"
    echo "   Check latest log: logs/stage2_ghop_*.log"
    echo ""
    echo "Common issues:"
    echo "  1. Checkpoint dimension mismatch (check config)"
    echo "  2. GHOP model loading errors"
    echo "  3. Out of memory (reduce batch size)"
    exit 1
fi


#chmod +x scripts/train_two_stage_v2_stage2_with_ghop.sh
#./scripts/train_two_stage_v2_stage2_with_ghop.sh 2>&1 | tee logs/stage2_1to1_hold_MC1_ho3d_official_checkpoint_$(date +%Y%m%d_%H%M%S).log
# tail -f logs/stage2_1to1_hold_MC1_ho3d_official_checkpoint_*.log | grep --line-buffered "Avg loss"
# tail -f logs/stage2_1to1_hold_MC1_ho3d_official_checkpoint_*.log | grep -E "Stage|Checkpoint|✅|❌|ERROR"
# watch -n 5 nvidia-smi

## Use HOLD dataset (hold_bottle1_itw/build/)
## Enable GHOP SDS loss via config
#python train.py \
#    --case hold_bottle1_itw \
#    --config confs/ghop_stage2_hold_MC1_ho3d_cb20a1702.yaml \
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
#    --config confs/ghop_stage2_hold_MC1_ho3d_cb20a1702.yaml \
#    --use_ghop \  # ← Selects GHOP dataset
#    --num_epoch 30
#
## Result:
## - Dataset: data/ghop_bottle_1/ghop_data/ (71 frames, HOI4D)
## - GHOP SDS: Enabled (phase3.enabled: true)
## - Training: Video sequence with temporal consistency