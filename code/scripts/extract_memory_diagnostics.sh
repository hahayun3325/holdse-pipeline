#!/bin/bash
# extract_memory_diagnostics.sh
# ================================================================
# UPDATED: Match new training_step format with 10-step frequency
# and 500 MB threshold for memory summaries
# ================================================================

cd ~/Projects/holdse/code

# Find most recent log
LOG=$(ls -t ../ghop_production_chunked_results/diagnostic_chunk4_*.log 2>/dev/null | head -1)

if [ -z "$LOG" ]; then
    echo "❌ No diagnostic log found"
    exit 1
fi

echo "Analyzing: $LOG"
echo ""

# ================================================================
# 1. CRITICAL: Memory Gap Analysis (New Format)
# ================================================================
echo "========================================================================"
echo "1. MEMORY GAP ANALYSIS (Allocated vs Reserved)"
echo "========================================================================"
echo ""

echo "Memory measurements (every 10 steps):"
echo "Format: Step | Allocated | Reserved | Gap | Gap%"
echo ""

grep -A 3 "\[Memory Summary Step" "$LOG" | \
    grep -E "(Memory Summary Step|Allocated:|Reserved:|Gap:)" | \
    awk '
        /Memory Summary Step/ {step=$4; gsub(/\]/, "", step)}
        /Allocated:/ {alloc=$2}
        /Reserved:/ {reserved=$2}
        /Gap:/ {
            gap=$2
            pct=$4
            gsub(/\(/, "", pct)
            gsub(/%\)/, "", pct)
            printf "%6s | %8s | %8s | %8s | %6s\n", step, alloc, reserved, gap, pct
        }
    ' | head -50

echo ""

# ================================================================
# 2. Growth Pattern Detection
# ================================================================
echo "========================================================================"
echo "2. MEMORY GROWTH PATTERN"
echo "========================================================================"
echo ""

echo "First 10 measurements:"
grep "Gap:" "$LOG" | awk '{print $2}' | head -10 | nl

echo ""
echo "Last 10 measurements:"
grep "Gap:" "$LOG" | awk '{print $2}' | tail -10 | nl

echo ""

# Calculate growth rate
FIRST_GAP=$(grep "Gap:" "$LOG" | head -1 | awk '{print $2}')
LAST_GAP=$(grep "Gap:" "$LOG" | tail -1 | awk '{print $2}')
NUM_MEASUREMENTS=$(grep -c "Gap:" "$LOG")

if [ -n "$FIRST_GAP" ] && [ -n "$LAST_GAP" ] && [ "$NUM_MEASUREMENTS" -gt 1 ]; then
    TOTAL_GROWTH=$(echo "$LAST_GAP - $FIRST_GAP" | bc 2>/dev/null)
    AVG_GROWTH=$(echo "scale=2; $TOTAL_GROWTH / $NUM_MEASUREMENTS" | bc 2>/dev/null)

    echo "Growth Statistics:"
    echo "  Initial gap:       ${FIRST_GAP} MB"
    echo "  Final gap:         ${LAST_GAP} MB"
    echo "  Total growth:      ${TOTAL_GROWTH} MB"
    echo "  Measurements:      ${NUM_MEASUREMENTS}"
    echo "  Avg growth/step:   ${AVG_GROWTH} MB"
    echo ""

    # Fragmentation verdict
    if (( $(echo "$LAST_GAP > 1000" | bc -l 2>/dev/null) )); then
        echo "  ❌ SEVERE FRAGMENTATION: Gap > 1 GB"
    elif (( $(echo "$LAST_GAP > 500" | bc -l 2>/dev/null) )); then
        echo "  ⚠️  MODERATE FRAGMENTATION: Gap > 500 MB"
    else
        echo "  ✅ HEALTHY: Gap < 500 MB"
    fi
fi

echo ""

# ================================================================
# 3. Memory Summary Files Detection
# ================================================================
echo "========================================================================"
echo "3. MEMORY SUMMARY FILES (Fragmentation Evidence)"
echo "========================================================================"
echo ""

echo "PyTorch Allocated Memory (first 10 reports):"
grep "PyTorch Allocated:" "$LOG" | awk '{print $4}' | head -10

if [ -n "$SUMMARY_FILES" ]; then
    NUM_FILES=$(echo "$SUMMARY_FILES" | wc -l)
    echo "✅ Found $NUM_FILES memory summary files (gap exceeded 500 MB threshold)"
    echo ""
    echo "Summary files generated at steps:"
    echo "$SUMMARY_FILES" | sed 's/.*memory_summary_step_//;s/.txt$//' | sort -n | column -c 80
    echo ""

    echo "Analyzing inactive split blocks (fragmentation indicator):"
    echo "Format: Step | Inactive Splits"
    echo ""
    for file in $SUMMARY_FILES; do
        step=$(basename "$file" | sed 's/memory_summary_step_//;s/.txt$//')
        inactive=$(grep -i "inactive_split_bytes" "$file" 2>/dev/null | head -1 | awk '{print $(NF-1), $NF}')
        if [ -n "$inactive" ]; then
            echo "  Step $step: $inactive"
        else
            # Try alternative format
            inactive=$(grep -i "inactive_split" "$file" 2>/dev/null | grep -v "num_alloc" | head -1)
            if [ -n "$inactive" ]; then
                echo "  Step $step: $inactive"
            fi
        fi
    done
    echo ""

    echo "To view detailed summary for a specific step:"
    echo "  cat ../ghop_production_chunked_results/memory_summary_step_<STEP>.txt"

else
    echo "✅ No memory summary files found"
    echo "   Gap never exceeded 500 MB threshold"
    echo "   This indicates NO SEVERE FRAGMENTATION"
fi

echo ""

# ================================================================
# 4. Fragmentation Warning Detection
# ================================================================
echo "========================================================================"
echo "4. FRAGMENTATION WARNINGS"
echo "========================================================================"
echo ""

WARNINGS=$(grep -c "Large memory gap detected" "$LOG" 2>/dev/null)

if [ "$WARNINGS" -gt 0 ]; then
    echo "⚠️  Found $WARNINGS fragmentation warnings"
    echo ""
    echo "Warning details:"
    grep "Large memory gap detected" "$LOG" | head -10
else
    echo "✅ No fragmentation warnings logged"
fi

echo ""

# ================================================================
# 5. GPU Monitor Data Correlation
# ================================================================
echo "========================================================================"
echo "5. GPU MEMORY MONITOR (nvidia-smi)"
echo "========================================================================"
echo ""

# Find corresponding GPU log
GPU_LOG=$(ls -t ../ghop_production_chunked_results/diagnostic_gpu_*.log 2>/dev/null | head -1)

if [ -z "$GPU_LOG" ]; then
    GPU_LOG=$(ls -t ../ghop_production_chunked_results/diagnostic_gpu_*.csv 2>/dev/null | head -1)
fi

if [ -n "$GPU_LOG" ]; then
    echo "GPU monitor log: $(basename "$GPU_LOG")"
    echo ""

    echo "First 5 GPU measurements:"
    head -7 "$GPU_LOG" | tail -5
    echo ""

    echo "Last 5 GPU measurements:"
    tail -5 "$GPU_LOG"
    echo ""

    # Calculate total GPU growth
    FIRST_GPU=$(head -7 "$GPU_LOG" | tail -1 | cut -d',' -f2)
    LAST_GPU=$(tail -1 "$GPU_LOG" | cut -d',' -f2)

    if [ -n "$FIRST_GPU" ] && [ -n "$LAST_GPU" ]; then
        GPU_GROWTH=$((LAST_GPU - FIRST_GPU))
        echo "GPU Memory Growth:"
        echo "  Initial: ${FIRST_GPU} MiB"
        echo "  Final:   ${LAST_GPU} MiB"
        echo "  Growth:  ${GPU_GROWTH} MiB"
        echo ""

        if [ "$GPU_GROWTH" -gt 1000 ]; then
            echo "  ❌ SEVERE: GPU growth > 1 GB indicates fragmentation"
        elif [ "$GPU_GROWTH" -gt 500 ]; then
            echo "  ⚠️  MODERATE: GPU growth > 500 MB may indicate fragmentation"
        else
            echo "  ✅ HEALTHY: GPU growth < 500 MB is normal"
        fi
    fi
else
    echo "⚠️  No GPU monitor log found"
fi

echo ""

# ================================================================
# 6. OOM Error Detection
# ================================================================
echo "========================================================================"
echo "6. OOM ERROR DETECTION"
echo "========================================================================"
echo ""

if grep -q "CUDA out of memory" "$LOG"; then
    echo "❌ OOM Error detected!"
    echo ""
    echo "Last memory measurements before OOM:"
    grep -B 30 "CUDA out of memory" "$LOG" | grep -A 3 "\[Memory Summary Step" | tail -20
    echo ""
    echo "OOM error message:"
    grep -A 5 "CUDA out of memory" "$LOG" | head -10
else
    echo "✅ No OOM error detected"
    echo "   Training completed successfully or still running"
fi

echo ""

# ================================================================
# 7. Timeline Comparison (PyTorch vs GPU)
# ================================================================
echo "========================================================================"
echo "7. PYTORCH vs GPU MEMORY TIMELINE"
echo "========================================================================"
echo ""

echo "Comparing PyTorch allocated with GPU total at key points:"
echo "Format: Measurement | PyTorch Allocated | PyTorch Reserved | GPU Total (if available)"
echo ""

# Extract measurements every 5th report
grep -A 3 "\[Memory Summary Step" "$LOG" | \
    awk '
        /Memory Summary Step/ {
            step=$4
            gsub(/\]/, "", step)
            count++
        }
        /Allocated:/ {alloc=$2}
        /Reserved:/ {reserved=$2}
        /Gap:/ {
            gap=$2
            if (count % 5 == 0 || count <= 5 || gap > 500) {
                printf "#%-3d | %8s MB | %8s MB | Gap: %8s MB\n", count, alloc, reserved, gap
            }
        }
    '

echo ""

# ================================================================
# 8. Summary Statistics
# ================================================================
echo "========================================================================"
echo "8. SUMMARY STATISTICS"
echo "========================================================================"
echo ""

TOTAL_REPORTS=$(grep -c "\[Memory Summary Step" "$LOG")
echo "Total memory reports: $TOTAL_REPORTS (every 10 steps)"

if [ "$TOTAL_REPORTS" -gt 0 ]; then
    EXPECTED_STEPS=$((TOTAL_REPORTS * 10))
    echo "Estimated training steps: ~$EXPECTED_STEPS"
fi

# Average gap
AVG_GAP=$(grep "Gap:" "$LOG" | awk '{sum+=$2; count++} END {printf "%.1f", sum/count}' 2>/dev/null)
if [ -n "$AVG_GAP" ]; then
    echo "Average memory gap: ${AVG_GAP} MB"
fi

# Peak gap
PEAK_GAP=$(grep "Gap:" "$LOG" | awk '{print $2}' | sort -n | tail -1)
if [ -n "$PEAK_GAP" ]; then
    echo "Peak memory gap: ${PEAK_GAP} MB"
fi

# Summary file count
SUMMARY_COUNT=$(ls -1 ../ghop_production_chunked_results/memory_summary_step_*.txt 2>/dev/null | wc -l)
echo "Memory summary files: ${SUMMARY_COUNT}"

echo ""

# ================================================================
# 9. FINAL DIAGNOSIS
# ================================================================
echo "========================================================================"
echo "9. FINAL DIAGNOSIS"
echo "========================================================================"
echo ""

HAS_OOM=$(grep -c "CUDA out of memory" "$LOG" 2>/dev/null)
HAS_SUMMARIES=$(ls -1 ../ghop_production_chunked_results/memory_summary_step_*.txt 2>/dev/null | wc -l)
PEAK_GAP_NUM=$(grep "Gap:" "$LOG" | awk '{print $2}' | sort -n | tail -1 | cut -d'.' -f1)

if [ "$HAS_OOM" -gt 0 ]; then
    echo "❌ STATUS: Training crashed with OOM"
    echo "   VERDICT: SEVERE FRAGMENTATION CONFIRMED"
    echo "   ACTION: Apply allocator fixes immediately"
elif [ "$HAS_SUMMARIES" -gt 5 ]; then
    echo "⚠️  STATUS: Multiple fragmentation warnings"
    echo "   VERDICT: MODERATE TO SEVERE FRAGMENTATION"
    echo "   ACTION: Apply allocator fixes recommended"
elif [ -n "$PEAK_GAP_NUM" ] && [ "$PEAK_GAP_NUM" -gt 500 ]; then
    echo "⚠️  STATUS: Gap exceeded 500 MB threshold"
    echo "   VERDICT: FRAGMENTATION DETECTED"
    echo "   ACTION: Monitor closely, consider fixes"
else
    echo "✅ STATUS: Memory behavior healthy"
    echo "   VERDICT: NO SIGNIFICANT FRAGMENTATION"
    echo "   ACTION: Previous fixes may have resolved issue"
fi

echo ""
echo "========================================================================"
echo "ANALYSIS COMPLETE"
echo "========================================================================"
