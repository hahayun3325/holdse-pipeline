#!/bin/bash
# train_stage3_FIXED.sh
# Stage 3: Full Pipeline Training with proper phase scheduling

cd ~/Projects/holdse/code

echo ""
echo "========================================================================"
echo "STAGE 3: FULL PIPELINE TRAINING (FIXED)"
echo "========================================================================"
echo ""
echo "Configuration: confs/ghop_stage3_full_pipeline.yaml"
echo "Phases Enabled:"
echo "  ✅ Phase 3: GHOP SDS (Object Shape, Iter 0+)"
echo "  ✅ Phase 4: Contact Refinement (Iter 500+)"
echo "  ✅ Phase 5: Temporal Consistency (Iter 1000+)"
echo ""
echo "Input Checkpoint: logs/stage2_final.ckpt"
echo "Output Checkpoint: logs/stage3_final.ckpt"
echo ""
echo "Epochs: 20 (increased from 10 for phase convergence)"
echo "Expected Duration: ~1.5-2 hours"
echo ""
echo "Phase Schedule:"
echo "  Epochs 1-2:  Phase 3 only (GHOP SDS warmup)"
echo "  Epochs 2-4:  Phase 3+4 (Add contact refinement)"
echo "  Epochs 4-20: Phase 3+4+5 (Full pipeline)"
echo ""
echo "========================================================================"
echo ""

# ================================================================
# Verification: Stage 2 checkpoint
# ================================================================
if [ ! -f "logs/stage2_final.ckpt" ]; then
    echo "❌ ERROR: Stage 2 checkpoint not found: logs/stage2_final.ckpt"
    echo ""
    echo "Available checkpoints:"
    find logs -name "last.ckpt" -type f -printf '  %p\n' | tail -5
    exit 1
fi

echo "✅ Stage 2 checkpoint verified: logs/stage2_final.ckpt"

# Check checkpoint info
python3 << 'EOF'
import torch
try:
    ckpt = torch.load('logs/stage2_final.ckpt', map_location='cpu')
    epoch = ckpt.get('epoch', 'unknown')
    print(f"   Epoch: {epoch}")
    if 'global_step' in ckpt:
        print(f"   Global step: {ckpt['global_step']}")
except Exception as e:
    print(f"   Warning: Could not load checkpoint info: {e}")
EOF

echo ""

# ================================================================
# Verification: Configuration file
# ================================================================
if [ ! -f "confs/ghop_stage3_full_pipeline.yaml" ]; then
    echo "❌ ERROR: Fixed Stage 3 config not found"
    echo "   Please create confs/ghop_stage3_full_pipeline.yaml with correct dimensions"
    exit 1
fi

echo "✅ Configuration file verified"
echo ""

# ================================================================
# Verification: GHOP checkpoint
# ================================================================
if [ ! -f "checkpoints/ghop/last.ckpt" ]; then
    echo "❌ ERROR: GHOP checkpoint not found"
    exit 1
fi

echo "✅ GHOP checkpoint verified"
echo ""

# ================================================================
# Run Stage 3 Training
# ================================================================
echo "Starting Stage 3 full pipeline training..."
echo "  Press Ctrl+C to stop"
echo ""

# Run Stage 3 training
python train.py \
    --config confs/ghop_stage3_full_pipeline.yaml \
    --case hold_MC1_ho3d \
    --num_epoch 30 \
    --load_ckpt logs/140dc5c18/checkpoints/last.ckpt \
    --no-comet \
    --gpu_id 0 \
    --no-pin-memory
# logs/140dc5c18/checkpoints/last.ckpt stage 1 checkpoint
# ================================================================
# Post-Training: Save and Report
# ================================================================
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✅ STAGE 3 TRAINING COMPLETE!"
    echo "========================================================================"

    # Find and save final checkpoint
    LATEST_CKPT=$(find logs -name "last.ckpt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    cp "$LATEST_CKPT" logs/stage3_final.ckpt

    echo ""
    echo "Final Checkpoint: logs/stage3_final.ckpt"
    echo "Original location: $LATEST_CKPT"
    echo ""
    echo "Complete Training Pipeline:"
    echo "  ✅ Stage 1: RGB Learning (100 epochs)"
    echo "  ✅ Stage 2: RGB + GHOP SDS (30 epochs)"
    echo "  ✅ Stage 3: Full Pipeline (20 epochs) - FINAL"
    echo ""
    echo "Next Steps:"
    echo "  1. Render validation:"
    echo "     python scripts/render_validation_with_arg.py --checkpoint logs/stage3_final.ckpt"
    echo ""
    echo "  2. Compare all stages:"
    echo "     - Stage 1: Basic RGB"
    echo "     - Stage 2: + GHOP SDS"
    echo "     - Stage 3: + Contact + Temporal"
    echo ""
    echo "  3. Extract final meshes:"
    echo "     python scripts/extract_meshes.py --checkpoint logs/stage3_final.ckpt"
    echo ""
    echo "========================================================================"
else
    echo ""
    echo "========================================================================"
    echo "❌ STAGE 3 TRAINING FAILED"
    echo "========================================================================"
    echo "Check logs/stage3_full_pipeline_*.log for details"
    echo ""
    echo "Common issues:"
    echo "  1. Dimension mismatch (verify config matches Stage 2)"
    echo "  2. Phase 4 instability (reduce w_penetration)"
    echo "  3. Out of memory (reduce batch size or N_samples)"
    echo "  4. Contact mesh extraction errors"
    echo ""
    exit 1
fi

#chmod +x scripts/train_two_stage_v2_stage3.sh
# ./scripts/train_two_stage_v2_stage3.sh 2>&1 | tee logs/stage3_1to30_s1ckpt_$(date +%Y%m%d_%H%M%S).log
# tail -f logs/stage3_1to30_s1ckpt*.log | grep --line-buffered "Avg loss"
# tail -f logs/stage3_1to30_s1ckpt*.log | grep -E "Stage|Checkpoint|✅|❌|ERROR"
# watch -n 5 nvidia-smi