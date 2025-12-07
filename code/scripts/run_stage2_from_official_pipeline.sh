#!/bin/bash

set -e  # Exit on any error

echo "========================================="
echo "STAGE 2 COMPLETE PIPELINE AUTOMATION"
echo "========================================="

# Step 1: Run Stage 2 Training
echo ""
echo "=== Step 1/3: Starting Stage 2 Training ==="
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
STAGE2_LOG="logs/stage2_1to30_hold_MC1_ho3d_official_match_fixed_mano_${TIMESTAMP}.log"

./scripts/train_MC1_stage2_from_official.sh 2>&1 | tee "${STAGE2_LOG}"

# Check if training succeeded
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Stage 2 training failed!"
    exit 1
fi

echo "✅ Stage 2 training completed"

# Step 2: Find the most recently created checkpoint directory
echo ""
echo "=== Step 2/3: Locating Latest Checkpoint ==="

# Wait a moment for filesystem to sync
sleep 2

# Find the newest 9-character hex directory in logs/
LATEST_CHECKPOINT_DIR=$(find logs/ -maxdepth 1 -type d -name '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)

if [ -z "$LATEST_CHECKPOINT_DIR" ]; then
    echo "❌ ERROR: No checkpoint directory found!"
    exit 1
fi

CHECKPOINT_PATH="${LATEST_CHECKPOINT_DIR}/checkpoints/last.ckpt"

echo "Latest checkpoint directory: ${LATEST_CHECKPOINT_DIR}"
echo "Checkpoint path: ${CHECKPOINT_PATH}"

# Verify checkpoint exists
if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "❌ ERROR: Checkpoint file not found at ${CHECKPOINT_PATH}"
    exit 1
fi

echo "✅ Checkpoint verified"

# Step 3: Run Extraction Process
echo ""
echo "=== Step 3/3: Running Extraction ==="

EXTRACTION_OUTPUT="logs/evaluation_results/MC1_stage2_e1_official_match_fixed_mano_predictions_${TIMESTAMP}.pkl"
EXTRACTION_LOG="logs/evaluation_results/MC1_stage2_e1_official_match_fixed_mano_extraction_${TIMESTAMP}.log"

# Create evaluation_results directory if it doesn't exist
mkdir -p logs/evaluation_results

python scripts/extract_predictions.py \
    --checkpoint "${CHECKPOINT_PATH}" \
    --seq_name hold_MC1_ho3d \
    --config confs/stage2_hold_MC1_ho3d_sds_from_official.yaml \
    --output "${EXTRACTION_OUTPUT}" \
    2>&1 | tee "${EXTRACTION_LOG}"

if [ $? -ne 0 ]; then
    echo "❌ ERROR: Extraction failed!"
    exit 1
fi

echo "✅ Extraction completed"

# Step 4: Run Evaluation Process
echo ""
echo "=== Step 4/3: Running Evaluation ==="

python scripts/evaluate_predictions.py \
    --predictions "${EXTRACTION_OUTPUT}" \
    --compare ~/Projects/hold/code/logs/cb20a1702/checkpoints/last.ckpt.metric.json

if [ $? -ne 0 ]; then
    echo "❌ ERROR: Evaluation failed!"
    exit 1
fi

echo "✅ Evaluation completed"

# Summary
echo ""
echo "========================================="
echo "PIPELINE COMPLETED SUCCESSFULLY"
echo "========================================="
echo "Training log:        ${STAGE2_LOG}"
echo "Checkpoint used:     ${CHECKPOINT_PATH}"
echo "Predictions saved:   ${EXTRACTION_OUTPUT}"
echo "Extraction log:      ${EXTRACTION_LOG}"
echo "========================================="