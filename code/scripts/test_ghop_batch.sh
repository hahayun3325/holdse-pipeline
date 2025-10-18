#!/bin/bash
# GHOP Full Batch Testing Script (6-8 hours)
# Use AFTER quick test passes

set -e

echo "========================================================================"
echo "GHOP FULL BATCH TESTING"
echo "========================================================================"
echo "Start: $(date)"
echo ""

cd ~/Projects/holdse/code

# Full test configuration
GHOP_ROOT=~/Projects/ghop/data/HOI4D_clip
OBJECTS=("Bottle_1" "Bowl_1" "Kettle_1" "Knife_1" "Mug_1")

# Results directory
RESULTS_DIR="../ghop_test_results"
mkdir -p "$RESULTS_DIR"

# Full test parameters
NUM_EPOCHS=3
GPU_ID=0

# Timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Full test configuration:"
echo "  Objects: ${OBJECTS[@]}"
echo "  Epochs: $NUM_EPOCHS"
echo "  GPU: $GPU_ID"
echo "  Expected duration: 6-8 hours"
echo ""

# Summary file
SUMMARY_FILE="$RESULTS_DIR/full_summary_${TIMESTAMP}.txt"
echo "GHOP Full Batch Testing Summary" > "$SUMMARY_FILE"
echo "Start: $(date)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Test each object
for obj in "${OBJECTS[@]}"; do
    obj_lower=$(echo "$obj" | tr '[:upper:]' '[:lower:]')
    config_file="confs/ghop_${obj_lower}.yaml"
    
    echo "" | tee -a "$SUMMARY_FILE"
    echo "========================================================================" | tee -a "$SUMMARY_FILE"
    echo "Testing: $obj" | tee -a "$SUMMARY_FILE"
    echo "========================================================================" | tee -a "$SUMMARY_FILE"
    echo "" | tee -a "$SUMMARY_FILE"
    
    # Check prerequisites
    if [ ! -f "$config_file" ]; then
        echo "❌ Config not found: $config_file" | tee -a "$SUMMARY_FILE"
        continue
    fi
    
    data_dir="$GHOP_ROOT/$obj"
    if [ ! -d "$data_dir" ]; then
        echo "❌ Dataset not found: $data_dir" | tee -a "$SUMMARY_FILE"
        continue
    fi
    
    echo "✓ Config: $config_file" | tee -a "$SUMMARY_FILE"
    echo "✓ Dataset: $data_dir" | tee -a "$SUMMARY_FILE"
    echo "" | tee -a "$SUMMARY_FILE"
    
    # Launch training
    log_file="$RESULTS_DIR/${obj_lower}_${TIMESTAMP}.log"
    
    echo "Starting training..." | tee -a "$SUMMARY_FILE"
    echo "Log: $log_file" | tee -a "$SUMMARY_FILE"
    echo "" | tee -a "$SUMMARY_FILE"
    
    START_TIME=$(date +%s)
    
    python train.py \
        --config "$config_file" \
        --num_epoch $NUM_EPOCHS \
        --gpu_id $GPU_ID \
        2>&1 | tee "$log_file"
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo "" | tee -a "$SUMMARY_FILE"
    echo "Duration: ${DURATION}s ($((DURATION/60))min)" | tee -a "$SUMMARY_FILE"
    
    # Check Phase 5
    if grep -q "phase5/temporal_loss" "$log_file"; then
        echo "✅ Phase 5 ACTIVATED" | tee -a "$SUMMARY_FILE"
        grep "phase5/temporal_loss" "$log_file" | tail -5 | tee -a "$SUMMARY_FILE"
    else
        echo "❌ Phase 5 NOT ACTIVATED" | tee -a "$SUMMARY_FILE"
    fi
    
    echo "" | tee -a "$SUMMARY_FILE"
    echo "✓ $obj complete" | tee -a "$SUMMARY_FILE"
done

echo "" | tee -a "$SUMMARY_FILE"
echo "========================================================================" | tee -a "$SUMMARY_FILE"
echo "✅ ALL TESTING COMPLETE" | tee -a "$SUMMARY_FILE"
echo "========================================================================" | tee -a "$SUMMARY_FILE"
echo "End: $(date)" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

echo "Summary: $SUMMARY_FILE"
