#!/bin/bash
# GHOP Quick Test Script (2 minutes total) - FIXED VERSION
# Purpose: Fast validation to identify pipeline issues
# Generates proper config with model section

set -e

echo "========================================================================"
echo "GHOP QUICK TEST (2-minute validation mode)"
echo "========================================================================"
echo "Start: $(date)"
echo ""

cd ~/Projects/holdse/code

# ========================================================================
# PRE-CHECK: Ensure mock data exists
# ========================================================================
echo "Pre-check: Verifying mock HOLD data structure..."
echo "------------------------------------------------------------------------"

GHOP_ROOT=~/Projects/ghop/data/HOI4D_clip
TEST_OBJECTS=("Bottle_1")

for obj in "${TEST_OBJECTS[@]}"; do
    obj_lower=$(echo "$obj" | tr '[:upper:]' '[:lower:]')
    case_name="ghop_${obj_lower}"

    # Check if mock data exists
    if [ ! -f "./data/$case_name/build/data.npy" ]; then
        echo "❌ Mock data missing for $case_name"
        exit 1
    else
        echo "✓ Mock data exists for $case_name"
    fi
done

echo ""

# ========================================================================
# Test configuration
# ========================================================================
RESULTS_DIR="../ghop_quick_test_results"
mkdir -p "$RESULTS_DIR"

MAX_ITERATIONS=20
NUM_EPOCHS=1
GPU_ID=0
BATCH_SIZE=2
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Quick test configuration:"
echo "  Test object: ${TEST_OBJECTS[@]}"
echo "  Max iterations: $MAX_ITERATIONS"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Expected duration: ~2 minutes"
echo ""

# Summary file
SUMMARY_FILE="$RESULTS_DIR/quick_summary_${TIMESTAMP}.txt"
echo "GHOP Quick Test Summary" > "$SUMMARY_FILE"
echo "Start: $(date)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# ========================================================================
# Run test
# ========================================================================
for obj in "${TEST_OBJECTS[@]}"; do
    obj_lower=$(echo "$obj" | tr '[:upper:]' '[:lower:]')
    CASE_NAME="ghop_${obj_lower}"

    echo "" | tee -a "$SUMMARY_FILE"
    echo "========================================================================" | tee -a "$SUMMARY_FILE"
    echo "Quick Testing: $obj (case: $CASE_NAME)" | tee -a "$SUMMARY_FILE"
    echo "========================================================================" | tee -a "$SUMMARY_FILE"
    echo "" | tee -a "$SUMMARY_FILE"

    data_dir="$GHOP_ROOT/$obj"
    echo "✓ Dataset: $data_dir" | tee -a "$SUMMARY_FILE"
    echo "✓ Case name: $CASE_NAME" | tee -a "$SUMMARY_FILE"
    echo "" | tee -a "$SUMMARY_FILE"
    
    # ========================================================================
    # Create MINIMAL config (inherits from test_checkpoint_loading)
    # ========================================================================
    QUICK_CONFIG="confs/ghop_quick_${obj_lower}.yaml"

    echo "Creating config by copying test_checkpoint_loading.yaml..." | tee -a "$SUMMARY_FILE"

    # Copy and modify
    cp confs/test_checkpoint_loading.yaml "$QUICK_CONFIG"

    # Modify key parameters using sed
    sed -i 's/^  max_steps:.*/  max_steps: '$MAX_ITERATIONS'/' "$QUICK_CONFIG"
    sed -i 's/^  num_epochs:.*/  num_epochs: '$NUM_EPOCHS'/' "$QUICK_CONFIG"
    sed -i 's/^    batch_size: 2/    batch_size: '$BATCH_SIZE'/' "$QUICK_CONFIG"

    # Update experiment name
    cat >> "$QUICK_CONFIG" << EOF

# Quick test overrides
experiment:
  name: 'ghop_quick_${obj_lower}'
  description: 'Quick test for $obj'
EOF
    
    echo "✓ Config: $QUICK_CONFIG" | tee -a "$SUMMARY_FILE"
    echo "" | tee -a "$SUMMARY_FILE"
    
    # Launch test
    log_file="$RESULTS_DIR/quick_${obj_lower}_${TIMESTAMP}.log"
    
    echo "Starting quick test..." | tee -a "$SUMMARY_FILE"
    echo "Command: python train.py --config $QUICK_CONFIG --case $CASE_NAME --gpu_id $GPU_ID --num_epoch $NUM_EPOCHS" | tee -a "$SUMMARY_FILE"
    echo "Log: $log_file" | tee -a "$SUMMARY_FILE"
    echo "" | tee -a "$SUMMARY_FILE"
    
    START_TIME=$(date +%s)
    
    # Run test
    if python train.py \
        --config "$QUICK_CONFIG" \
        --case "$CASE_NAME" \
        --gpu_id $GPU_ID \
        --num_epoch $NUM_EPOCHS \
        2>&1 | tee "$log_file"; then
        
        TEST_STATUS="✅ COMPLETED"
    else
        TEST_STATUS="❌ FAILED"
    fi
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo "" | tee -a "$SUMMARY_FILE"
    echo "Test status: $TEST_STATUS" | tee -a "$SUMMARY_FILE"
    echo "Duration: ${DURATION}s" | tee -a "$SUMMARY_FILE"
    echo "" | tee -a "$SUMMARY_FILE"

    # Analysis
    echo "========================================================================" | tee -a "$SUMMARY_FILE"
    echo "ANALYSIS" | tee -a "$SUMMARY_FILE"
    echo "========================================================================" | tee -a "$SUMMARY_FILE"
    echo "" | tee -a "$SUMMARY_FILE"
    
    if grep -q "ImageDataset\|Text prompt" "$log_file"; then
        echo "✅ Dataset: Loaded" | tee -a "$SUMMARY_FILE"
        grep "Category\|Text prompt" "$log_file" | head -3 | tee -a "$SUMMARY_FILE"
    else
        echo "❌ Dataset: NOT loaded" | tee -a "$SUMMARY_FILE"
    fi
    echo "" | tee -a "$SUMMARY_FILE"
    
    if grep -q "phase5\|Phase 5" "$log_file"; then
        echo "✅ Phase 5: ACTIVATED" | tee -a "$SUMMARY_FILE"
        grep "phase5\|Phase 5" "$log_file" | head -5 | tee -a "$SUMMARY_FILE"
    else
        echo "❌ Phase 5: NOT activated" | tee -a "$SUMMARY_FILE"
    fi
    echo "" | tee -a "$SUMMARY_FILE"

    if grep -q "train/loss\|Epoch\|Step" "$log_file"; then
        echo "✅ Training: Progress" | tee -a "$SUMMARY_FILE"
        grep "Epoch\|train/loss" "$log_file" | tail -10 | tee -a "$SUMMARY_FILE"
    else
        echo "❌ Training: No progress" | tee -a "$SUMMARY_FILE"
    fi
    echo "" | tee -a "$SUMMARY_FILE"
    
    if grep -qi "error\|exception" "$log_file" | head -1; then
        echo "⚠️ Errors found" | tee -a "$SUMMARY_FILE"
        grep -i "error\|exception" "$log_file" | head -5 | tee -a "$SUMMARY_FILE"
    else
        echo "✅ No errors" | tee -a "$SUMMARY_FILE"
    fi
done

echo "" | tee -a "$SUMMARY_FILE"
echo "========================================================================" | tee -a "$SUMMARY_FILE"
echo "✅ QUICK TEST COMPLETE" | tee -a "$SUMMARY_FILE"
echo "========================================================================" | tee -a "$SUMMARY_FILE"
echo "End: $(date)" | tee -a "$SUMMARY_FILE"

echo ""
echo "Summary: $SUMMARY_FILE"
