#!/bin/bash

set -e  # Exit on any error

echo "========================================="
echo "Evaluation Process AUTOMATION"
echo "========================================="

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CHECKPOINT_PATH="/home/fredcui/Projects/hold/code/logs/cb20a1702/checkpoints/last.ckpt"

# Step 1: Run Extraction Process
echo ""
echo "=== Step 1/2: Running Extraction ==="

EXTRACTION_OUTPUT="logs/evaluation_results/MC1_official_predictions_${TIMESTAMP}.pkl"
EXTRACTION_LOG="logs/evaluation_results/MC1_official_extraction_${TIMESTAMP}.log"

# Create evaluation_results directory if it doesn't exist
mkdir -p logs/evaluation_results

python scripts/extract_predictions.py \
    --checkpoint "${CHECKPOINT_PATH}" \
    --seq_name hold_MC1_ho3d \
    --config confs/stage1_hold_MC1_ho3d_sds_from_official.yaml \
    --output "${EXTRACTION_OUTPUT}" \
    2>&1 | tee "${EXTRACTION_LOG}"

if [ $? -ne 0 ]; then
    echo "❌ ERROR: Extraction failed!"
    exit 1
fi

echo "✅ Extraction completed"

# Step 2: Run Evaluation Process
echo ""
echo "=== Step 2/2: Running Evaluation ==="

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
echo "Checkpoint used:     ${CHECKPOINT_PATH}"
echo "Predictions saved:   ${EXTRACTION_OUTPUT}"
echo "Extraction log:      ${EXTRACTION_LOG}"
echo "========================================="