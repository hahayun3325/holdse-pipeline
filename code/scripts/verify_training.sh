#!/bin/bash
LOG_FILE=$1

if [ -z "$LOG_FILE" ]; then
    echo "Usage: ./verify_training.sh <log_file_path>"
    exit 1
fi

echo "========================================="
echo "TRAINING PIPELINE VERIFICATION"
echo "========================================="

# 1. Phase Activation
echo -e "\n[1/10] Phase Activation Check..."
PHASE_COUNT=$(grep "ACTIVATED at step" $LOG_FILE | wc -l)
if [ $PHASE_COUNT -eq 3 ]; then
    echo "✅ PASS: All 3 phases activated"
    grep "ACTIVATED at step" $LOG_FILE
else
    echo "❌ FAIL: Expected 3 phases, found $PHASE_COUNT"
fi

# 2. IndexError Check
echo -e "\n[2/10] IndexError Check..."
INDEX_ERRORS=$(grep -i "index.*out of bounds" $LOG_FILE | wc -l)
if [ $INDEX_ERRORS -eq 0 ]; then
    echo "✅ PASS: No IndexErrors found"
else
    echo "❌ FAIL: Found $INDEX_ERRORS IndexErrors"
fi

# 3. NaN/Inf Check
echo -e "\n[3/10] Numerical Stability Check..."
NAN_COUNT=$(grep -iE "nan|inf" $LOG_FILE | grep -i loss | wc -l)
if [ $NAN_COUNT -eq 0 ]; then
    echo "✅ PASS: No NaN/Inf losses"
else
    echo "⚠️  WARNING: Found $NAN_COUNT NaN/Inf occurrences"
fi

# 4. Phase 5 Temporal Loss
echo -e "\n[4/10] Temporal Loss Check..."
TEMPORAL_NONZERO=$(grep "Temporal loss:" $LOG_FILE | grep -v "0.000000" | wc -l)
if [ $TEMPORAL_NONZERO -gt 10 ]; then
    echo "✅ PASS: Temporal loss active ($TEMPORAL_NONZERO non-zero values)"
else
    echo "❌ FAIL: Temporal loss mostly zero ($TEMPORAL_NONZERO non-zero values)"
fi

# 5. History Growth
echo -e "\n[5/10] History Management Check..."
MAX_HISTORY=$(grep "history_len=" $LOG_FILE | grep -o "history_len=[0-9]*" | sed 's/history_len=//' | sort -n | tail -1)
if [ "$MAX_HISTORY" -ge 5 ]; then
    echo "✅ PASS: History grows to $MAX_HISTORY (expected 5)"
else
    echo "❌ FAIL: History only reaches $MAX_HISTORY (expected 5)"
fi

# 6. Phase 5 Failures
echo -e "\n[6/10] Phase 5 Failure Check..."
TEMPORAL_FAILS=$(grep "Temporal consistency computation failed" $LOG_FILE | wc -l)
if [ $TEMPORAL_FAILS -eq 0 ]; then
    echo "✅ PASS: No Phase 5 failures"
else
    echo "❌ FAIL: Found $TEMPORAL_FAILS Phase 5 failures"
fi

# 7. Shape Squeeze Operations
echo -e "\n[7/10] Shape Normalization Check..."
SQUEEZE_COUNT=$(grep "Squeezed" $LOG_FILE | wc -l)
if [ $SQUEEZE_COUNT -gt 0 ]; then
    echo "✅ PASS: Shape squeezing active ($SQUEEZE_COUNT operations)"
else
    echo "⚠️  WARNING: No squeeze operations found"
fi

# 8. Gradient Flow
echo -e "\n[8/10] Gradient Flow Check..."
GRAD_COUNT=$(grep "grad_norm" $LOG_FILE | wc -l)
if [ $GRAD_COUNT -gt 0 ]; then
    echo "✅ PASS: Gradients flowing ($GRAD_COUNT gradient logs)"
else
    echo "⚠️  WARNING: No gradient logs found"
fi

# 9. Training Progression
echo -e "\n[9/10] Training Progress Check..."
LAST_STEP=$(grep "\[TRAIN STEP\] step=" $LOG_FILE | tail -1 | grep -o "step=[0-9]*" | sed 's/step=//')
if [ ! -z "$LAST_STEP" ] && [ $LAST_STEP -gt 1000 ]; then
    echo "✅ PASS: Training at step $LAST_STEP"
else
    echo "⚠️  INFO: Training at step $LAST_STEP"
fi

# 10. Loss Trend
echo -e "\n[10/10] Loss Convergence Check..."
echo "Recent total losses:"
grep "Final total:" $LOG_FILE | tail -10

echo -e "\n========================================="
echo "VERIFICATION COMPLETE"
echo "========================================="