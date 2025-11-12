#!/bin/bash
# Two-stage training with EXPLICIT dataset control

set -e
cd "$(dirname "$0")/.."

CONFIG="confs/ghop_production_chunked_20251031_140851.yaml"
CONFIG_STAGE1="confs/ghop_stage1_tempodataset.yaml"
CONFIG_STAGE2="confs/ghop_stage2_ghopdataset.yaml"
CASE="ghop_bottle_1"
STAGE1_EPOCHS=20
STAGE2_EPOCHS=30
CHUNK_SIZE=5

# Clean previous logs
echo "Cleaning previous training logs..."
rm -rf logs/*
echo ""

# ================================================================
# STAGE 1: RGB Learning (ImageDataset - NO --use_ghop flag)
# ================================================================
echo "========================================================================"
echo "STAGE 1: RGB LEARNING (ImageDataset)"
echo "========================================================================"
echo "  Epochs: 0 → $STAGE1_EPOCHS"
echo "  Dataset: ImageDataset (RGB supervision)"
echo "  Flag: NO --use_ghop (forces ImageDataset)"
echo "========================================================================"
echo ""

CURRENT_EPOCH=0

while [ $CURRENT_EPOCH -lt $STAGE1_EPOCHS ]; do
    NEXT_EPOCH=$((CURRENT_EPOCH + CHUNK_SIZE))
    if [ $NEXT_EPOCH -gt $STAGE1_EPOCHS ]; then
        NEXT_EPOCH=$STAGE1_EPOCHS
    fi

    echo "[Stage 1] Chunk: Epochs $CURRENT_EPOCH → $NEXT_EPOCH"

    # Find latest checkpoint
    if [ $CURRENT_EPOCH -eq 0 ]; then
        RESUME_ARG=""
        echo "[Stage 1] Starting fresh (no checkpoint)"
    else
        LAST_CKPT=$(find logs -name "last.ckpt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
        if [ -z "$LAST_CKPT" ]; then
            echo "❌ ERROR: Could not find checkpoint to resume from"
            exit 1
        fi
        RESUME_ARG="--load_ckpt $LAST_CKPT"
        echo "[Stage 1] Resuming from: $LAST_CKPT"
    fi

    # ⭐ KEY: NO --use_ghop flag → Forces ImageDataset
    python train.py \
        --config "$CONFIG_STAGE1" \
        --case "$CASE" \
        --num_epoch $NEXT_EPOCH \
        $RESUME_ARG \
        --no-comet \
        --gpu_id 0 \
        --no-pin-memory
        # ← NO --use_ghop flag!

    if [ $? -ne 0 ]; then
        echo "❌ Stage 1 failed at epoch $CURRENT_EPOCH → $NEXT_EPOCH"
        exit 1
    fi

    CURRENT_EPOCH=$NEXT_EPOCH
    sleep 3
done

echo "✅ STAGE 1 COMPLETE (Epochs 0-$STAGE1_EPOCHS)"
echo ""

# ================================================================
# ✅ FIX #3: Save explicit Stage 1 checkpoint
# ================================================================
echo "========================================================================"
echo "SAVING STAGE 1 CHECKPOINT"
echo "========================================================================"

STAGE1_CKPT=$(find logs -name "last.ckpt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

if [ -z "$STAGE1_CKPT" ]; then
    echo "❌ ERROR: No Stage 1 checkpoint found!"
    exit 1
fi

# Copy to Stage 1 specific name
STAGE1_CKPT_COPY="logs/stage1_final.ckpt"
cp "$STAGE1_CKPT" "$STAGE1_CKPT_COPY"
echo "✅ Stage 1 checkpoint saved: $STAGE1_CKPT_COPY"

# Backup the full logs directory
STAGE1_LOGS_BACKUP="logs/stage1_logs_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$STAGE1_LOGS_BACKUP"
find logs -name "*.log" -type f -exec cp {} "$STAGE1_LOGS_BACKUP"/ \;
echo "✅ Stage 1 logs backed up to: $STAGE1_LOGS_BACKUP"

echo ""

# Verify RGB was learned
echo "========================================================================"
echo "VERIFYING STAGE 1 RGB LEARNING"
echo "========================================================================"

STAGE1_LOG=$(find logs -name "train.log" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

if grep -q "loss/rgb" "$STAGE1_LOG"; then
    echo "✅ RGB loss was computed"
    RGB_LOSS_COUNT=$(grep "loss/rgb" "$STAGE1_LOG" | wc -l)
    echo "   Total RGB loss entries: $RGB_LOSS_COUNT"
else
    echo "❌ WARNING: No RGB loss found in logs!"
    echo "   ImageDataset may not have been used"
fi

if grep -q "HAS COLOR VARIATION\|COLOR LEARNED" "$STAGE1_LOG"; then
    echo "✅ Color variation detected in renders"
else
    echo "⚠️  No color variation log found (may not have rendered)"
fi

echo "✅ STAGE 1 VERIFICATION COMPLETE"
echo ""

# ================================================================
# STAGE 2: Temporal Refinement (GHOPHOIDataset - WITH --use_ghop)
# ================================================================
echo "========================================================================"
echo "STAGE 2: TEMPORAL REFINEMENT (GHOPHOIDataset)"
echo "========================================================================"
echo "  Epochs: $STAGE1_EPOCHS → $STAGE2_EPOCHS"
echo "  Dataset: GHOPHOIDataset (temporal pairs)"
echo "  Flag: --use_ghop (forces GHOPHOIDataset)"
echo "========================================================================"
echo ""

CURRENT_EPOCH=$STAGE1_EPOCHS

while [ $CURRENT_EPOCH -lt $STAGE2_EPOCHS ]; do
    NEXT_EPOCH=$((CURRENT_EPOCH + CHUNK_SIZE))
    if [ $NEXT_EPOCH -gt $STAGE2_EPOCHS ]; then
        NEXT_EPOCH=$STAGE2_EPOCHS
    fi

    echo "[Stage 2] Chunk: Epochs $CURRENT_EPOCH → $NEXT_EPOCH"

    LAST_CKPT=$(find logs -name "last.ckpt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

    if [ -z "$LAST_CKPT" ]; then
        echo "❌ ERROR: Could not find checkpoint to resume Stage 2"
        exit 1
    fi

    echo "[Stage 2] Loading checkpoint: $LAST_CKPT"

    # ✅ FIX #2: Use $NEXT_EPOCH instead of hardcoded 30
    python train.py \
        --config "$CONFIG_STAGE2" \
        --case "$CASE" \
        --use_ghop \
        --num_epoch $NEXT_EPOCH \
        --load_ckpt "$LAST_CKPT" \
        --no-comet \
        --gpu_id 0 \
        --no-pin-memory

    if [ $? -ne 0 ]; then
        echo "❌ Stage 2 failed at epoch $CURRENT_EPOCH → $NEXT_EPOCH"

        # ✅ FIX #5: Save recovery checkpoint on Stage 2 failure
        RECOVERY_CKPT=$(find logs -name "last.ckpt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
        if [ -n "$RECOVERY_CKPT" ]; then
            RECOVERY_CKPT_COPY="logs/stage2_recovery_epoch_${CURRENT_EPOCH}_$(date +%Y%m%d_%H%M%S).ckpt"
            cp "$RECOVERY_CKPT" "$RECOVERY_CKPT_COPY"
            echo "💾 Recovery checkpoint saved: $RECOVERY_CKPT_COPY"
            echo "   To resume Stage 2 from this point, use:"
            echo "   python train.py --load_ckpt $RECOVERY_CKPT_COPY --num_epoch $NEXT_EPOCH ..."
        fi

        exit 1
    fi

    CURRENT_EPOCH=$NEXT_EPOCH
    sleep 3
done

echo "✅ STAGE 2 COMPLETE (Epochs $STAGE1_EPOCHS-$STAGE2_EPOCHS)"
echo ""

# ================================================================
# ✅ FIX #3: Save final checkpoint
# ================================================================
echo "========================================================================"
echo "SAVING FINAL CHECKPOINT"
echo "========================================================================"

FINAL_CKPT=$(find logs -name "last.ckpt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

if [ -z "$FINAL_CKPT" ]; then
    echo "❌ ERROR: No final checkpoint found!"
    exit 1
fi

FINAL_CKPT_COPY="logs/final_30epoch_$(date +%Y%m%d_%H%M%S).ckpt"
cp "$FINAL_CKPT" "$FINAL_CKPT_COPY"
echo "✅ Final 30-epoch checkpoint saved: $FINAL_CKPT_COPY"

echo ""
echo "========================================================================"
echo "✅✅✅ TWO-STAGE TRAINING COMPLETE ✅✅✅"
echo "========================================================================"
echo ""
echo "Final checkpoint: $FINAL_CKPT_COPY"
echo ""
echo "Next steps:"
echo "  1. Verify training quality: tensorboard --logdir logs/"
echo "  2. Render final output: python render.py --checkpoint $FINAL_CKPT_COPY"
echo "  3. Archive results: tar -czf ghop_training_$(date +%Y%m%d_%H%M%S).tar.gz logs/"

echo "✅ TWO-STAGE TRAINING COMPLETE"

#While training runs, monitor in separate terminal windows:
#
## Terminal 1: Watch training log in real-time
#tail -f logs/*/train.log | grep -E "Epoch|Phase|loss"
#
## Terminal 2: Monitor GPU memory
#nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1
#
## Terminal 3: Check for errors
#tail -f logs/*/train.log | grep -i "error\|warning\|exception"
