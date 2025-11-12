#!/bin/bash
# train_two_stage_v2_stage2_with_unified_ghop.sh

cd ~/Projects/holdse/code

echo "========================================================================"
echo "STAGE 2: GHOP SDS Temporal Refinement (FIXED)"
echo "========================================================================"
echo "  Stage 1 Checkpoint: logs/e0ad72ec8/checkpoints/last.ckpt"
echo "  GHOP Checkpoint: checkpoints/ghop/last.ckpt"
echo "  Config: confs/ghop_stage2_temporal_only.yaml"
echo "  Phases: RGB + Phase 3 (GHOP SDS)"
echo "========================================================================"
echo ""

# ================================================================
# Verification: Stage 1 Checkpoint
# ================================================================
STAGE1_CKPT="logs/ab5edc20f/checkpoints/last.ckpt"

if [ ! -f "$STAGE1_CKPT" ]; then
    echo "❌ ERROR: Stage 1 checkpoint not found at $STAGE1_CKPT"
    echo "   Available checkpoints:"
    find logs -name "last.ckpt" -type f -printf '   %p\n'
    exit 1
fi

echo "✅ Stage 1 checkpoint verified: $STAGE1_CKPT"

# Check checkpoint epoch
python3 << EOF
import torch
ckpt = torch.load('$STAGE1_CKPT', map_location='cpu')
epoch = ckpt.get('epoch', 'unknown')
loss = ckpt['state_dict']['loss'] if 'loss' in ckpt['state_dict'] else 'N/A'
print(f"   Epoch: {epoch}")
if 'global_step' in ckpt:
    print(f"   Global step: {ckpt['global_step']}")
EOF

echo ""

# ================================================================
# Verification: GHOP Checkpoint
# ================================================================
if [ ! -f "checkpoints/ghop/last.ckpt" ]; then
    echo "❌ ERROR: GHOP checkpoint not found at checkpoints/ghop/last.ckpt"
    echo "   Please ensure GHOP model is downloaded/trained"
    exit 1
fi

if [ ! -f "checkpoints/ghop/config.yaml" ]; then
    echo "❌ ERROR: GHOP config not found at checkpoints/ghop/config.yaml"
    exit 1
fi

echo "✅ GHOP checkpoints verified"
echo ""

# ================================================================
# Verification: Configuration File
# ================================================================
if [ ! -f "confs/ghop_stage2_temporal_only.yaml" ]; then
    echo "❌ ERROR: Fixed Stage 2 config not found"
    echo "   Please create confs/ghop_stage2_temporal_only.yaml"
    exit 1
fi

echo "✅ Configuration file verified"
echo ""

# ================================================================
# Run Stage 2 Training
# ================================================================
echo "Starting Stage 2 training..."
echo ""

python train.py \
    --config confs/ghop_stage2_temporal_only.yaml \
    --case hold_bottle1_itw \
    --use_ghop \
    --num_epoch 30 \
    --load_ckpt "$STAGE1_CKPT" \
    --no-comet \
    --gpu_id 0 \
    --no-pin-memory \
    2>&1 | tee logs/stage2_ghop_$(date +%Y%m%d_%H%M%S).log

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
#./scripts/train_two_stage_v2_stage2_with_ghop.sh 2>&1 | tee logs/full_training_stage2_1to30_$(date +%Y%m%d_%H%M%S).log
# tail -f logs/full_training_stage2_1to30_*.log | grep --line-buffered "Avg loss"
#tail -f logs/full_training_stage2_1to30_*.log | grep -E "Stage|Checkpoint|✅|❌|ERROR"
# watch -n 5 nvidia-smi