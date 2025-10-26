#!/bin/bash
# scripts/train_diagnostic_chunk4.sh
# ================================================================
# Run chunk 4 (epochs 40-50) with PERIODIC RESTARTS
# WORKAROUND: Restarts Python process every 5 epochs to clear memory leak
# ================================================================

cd ~/Projects/holdse/code

# ================================================================
# CONFIGURATION
# ================================================================
export PYTORCH_CUDA_ALLOC_CONF="backend:native"

RESULTS_DIR="../ghop_production_chunked_results"
mkdir -p "$RESULTS_DIR"

MAIN_LOG="$RESULTS_DIR/diagnostic_chunk4_chunked_$(date +%Y%m%d_%H%M%S).log"
GPU_LOG="$RESULTS_DIR/diagnostic_gpu_$(date +%Y%m%d_%H%M%S).csv"

# Training parameters
STARTING_EPOCH=40
ENDING_EPOCH=100
CHUNK_SIZE=5  # Restart every 5 epochs (prevents OOM at epoch 49)

# ================================================================
# Start GPU monitor (runs throughout all chunks)
# ================================================================
echo "Starting continuous GPU monitor..."

{
    echo "# GPU Memory Monitor - Diagnostic Chunk 4 (with periodic restarts)"
    echo "# Started: $(date)"
    echo "# Interval: 2 seconds"
    echo "timestamp,memory_used_MiB,memory_free_MiB,memory_total_MiB"
} > "$GPU_LOG"

(
    while true; do
        nvidia-smi --query-gpu=timestamp,memory.used,memory.free,memory.total \
                   --format=csv,noheader,nounits 2>/dev/null || break
        sleep 2
    done
) >> "$GPU_LOG" 2>&1 &

GPU_MONITOR_PID=$!
sleep 2

if ps -p $GPU_MONITOR_PID > /dev/null 2>&1; then
    echo "  ✓ GPU monitor running (PID: $GPU_MONITOR_PID)"
else
    echo "  ⚠️  GPU monitor failed to start"
fi
echo ""

# ================================================================
# Main training loop with periodic restarts
# ================================================================
{
    echo "========================================================================"
    echo "DIAGNOSTIC TRAINING: Chunk 4 with Periodic Restarts"
    echo "========================================================================"
    echo ""
    echo "Strategy: Train in $CHUNK_SIZE-epoch chunks to work around memory leak"
    echo "  Total range: Epochs $STARTING_EPOCH → $ENDING_EPOCH"
    echo "  Chunk size: $CHUNK_SIZE epochs"
    echo "  Expected chunks: $(( (ENDING_EPOCH - STARTING_EPOCH) / CHUNK_SIZE ))"
    echo ""

    # Find initial checkpoint and config
    CHECKPOINT_DIR="logs"
    LATEST_CHECKPOINT=$(find "$CHECKPOINT_DIR" -name "last.ckpt" -type f | xargs ls -t | head -1)

    if [ -z "$LATEST_CHECKPOINT" ]; then
        echo "❌ No checkpoint found"
        exit 1
    fi

    CONFIG_FILE=$(ls -t confs/ghop_production_chunked_*.yaml | head -1)

    if [ -z "$CONFIG_FILE" ]; then
        echo "❌ No config file found"
        exit 1
    fi

    echo "Initial checkpoint: $LATEST_CHECKPOINT"
    echo "Config file: $CONFIG_FILE"
    echo ""

    # Extract case name
    CASE=$(grep -m1 "^case:" "$CONFIG_FILE" | awk '{print $2}' | tr -d '"' | tr -d "'")

    if [ -z "$CASE" ]; then
        # Fallback: try to find from logs or use default
        CASE="ghop_bottle_1"
        echo "⚠️  Using default case: $CASE"
    fi

    echo "Dataset case: $CASE"
    echo ""

    # ================================================================
    # Train in chunks
    # ================================================================
    CURRENT_EPOCH=$STARTING_EPOCH
    CHUNK_NUM=1
    TOTAL_CHUNKS=$(( (ENDING_EPOCH - STARTING_EPOCH) / CHUNK_SIZE ))

    while [ $CURRENT_EPOCH -lt $ENDING_EPOCH ]; do
        NEXT_EPOCH=$((CURRENT_EPOCH + CHUNK_SIZE))
        if [ $NEXT_EPOCH -gt $ENDING_EPOCH ]; then
            NEXT_EPOCH=$ENDING_EPOCH
        fi

        ACTUAL_EPOCHS=$((NEXT_EPOCH - CURRENT_EPOCH))

        echo "========================================================================"
        echo "CHUNK $CHUNK_NUM/$TOTAL_CHUNKS: Epochs $CURRENT_EPOCH → $NEXT_EPOCH ($ACTUAL_EPOCHS epochs)"
        echo "========================================================================"
        echo ""
        echo "Checkpoint: $LATEST_CHECKPOINT"
        echo "Expected to train $ACTUAL_EPOCHS epochs"
        echo ""

        # Create chunk-specific log
        CHUNK_LOG="$RESULTS_DIR/chunk${CHUNK_NUM}_epochs_${CURRENT_EPOCH}_to_${NEXT_EPOCH}.log"

        # Run training chunk
        python -u train.py \
            --config "$CONFIG_FILE" \
            --case "$CASE" \
            --num_epoch $ACTUAL_EPOCHS \
            --load_ckpt "$LATEST_CHECKPOINT" \
            --no-comet \
            --gpu_id 0 \
            2>&1 | tee "$CHUNK_LOG"

        CHUNK_EXIT_CODE=${PIPESTATUS[0]}

        if [ $CHUNK_EXIT_CODE -ne 0 ]; then
            echo ""
            echo "❌ Chunk $CHUNK_NUM FAILED with exit code $CHUNK_EXIT_CODE"

            if [ $CHUNK_EXIT_CODE -eq 137 ]; then
                echo "   Exit 137 = OOM (killed by system)"
                echo "   Even with chunking, memory leak is too severe"
                echo "   Recommendation: Reduce CHUNK_SIZE from $CHUNK_SIZE to 3"
            fi

            EXIT_CODE=$CHUNK_EXIT_CODE
            break
        fi

        echo ""
        echo "✅ Chunk $CHUNK_NUM completed successfully"

        # Update checkpoint for next chunk
        LATEST_CHECKPOINT=$(find "$CHECKPOINT_DIR" -name "last.ckpt" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)

        # Memory check between chunks
        if [ -n "$LATEST_CHECKPOINT" ]; then
            echo "   Updated checkpoint: $LATEST_CHECKPOINT"
        else
            echo "   ⚠️  Warning: Could not find updated checkpoint"
        fi

        # Show GPU memory state before restart
        CURRENT_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
        echo "   GPU memory before restart: $CURRENT_MEM MiB"
        echo ""
        echo "🔄 Restarting Python process to clear accumulated memory..."
        echo ""

        # Small delay to ensure clean process termination
        sleep 3

        CURRENT_EPOCH=$NEXT_EPOCH
        CHUNK_NUM=$((CHUNK_NUM + 1))
    done

    # Final status
    if [ $CHUNK_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "========================================================================"
        echo "✅ ALL CHUNKS COMPLETED SUCCESSFULLY"
        echo "========================================================================"
        echo ""
        echo "Total chunks: $TOTAL_CHUNKS"
        echo "Epochs trained: $STARTING_EPOCH → $ENDING_EPOCH"
        EXIT_CODE=0
    fi

} 2>&1 | tee "$MAIN_LOG"

# Capture overall exit code
if [ -z "$EXIT_CODE" ]; then
    EXIT_CODE=${PIPESTATUS[0]}
fi

# ================================================================
# Cleanup GPU monitor
# ================================================================
kill $GPU_MONITOR_PID 2>/dev/null
wait $GPU_MONITOR_PID 2>/dev/null
echo ""
echo "GPU monitor stopped"

# ================================================================
# ANALYSIS: Memory growth across chunks
# ================================================================
echo ""
echo "========================================================================"
echo "MEMORY GROWTH ANALYSIS (Across All Chunks)"
echo "========================================================================"
echo ""

# Analyze GPU memory from start to finish
FIRST_MEM=$(head -7 "$GPU_LOG" | tail -1 | cut -d',' -f2 2>/dev/null)
LAST_MEM=$(tail -1 "$GPU_LOG" | cut -d',' -f2 2>/dev/null)

if [ -n "$FIRST_MEM" ] && [ -n "$LAST_MEM" ]; then
    TOTAL_GROWTH=$((LAST_MEM - FIRST_MEM))

    echo "GPU Memory (nvidia-smi):"
    echo "  Initial:  $FIRST_MEM MiB"
    echo "  Final:    $LAST_MEM MiB"
    echo "  Growth:   $TOTAL_GROWTH MiB"
    echo ""

    if [ "$TOTAL_GROWTH" -gt 5000 ]; then
        echo "  ❌ SEVERE: Growth > 5 GB even with chunking"
        echo "     Recommendation: Reduce CHUNK_SIZE to 3 epochs"
    elif [ "$TOTAL_GROWTH" -gt 2000 ]; then
        echo "  ⚠️  MODERATE: Growth 2-5 GB"
        echo "     Chunking helped but leak still significant"
    else
        echo "  ✅ SUCCESS: Growth < 2 GB"
        echo "     Periodic restarts effectively managed memory leak"
    fi
fi

echo ""

# Analyze memory patterns per chunk
echo "Memory patterns per chunk:"
for chunk_log in "$RESULTS_DIR"/chunk*_epochs_*.log; do
    if [ -f "$chunk_log" ]; then
        chunk_name=$(basename "$chunk_log" .log)
        first_gap=$(grep "Gap:" "$chunk_log" 2>/dev/null | head -1 | awk '{print $2}')
        last_gap=$(grep "Gap:" "$chunk_log" 2>/dev/null | tail -1 | awk '{print $2}')

        if [ -n "$first_gap" ] && [ -n "$last_gap" ]; then
            gap_growth=$(echo "$last_gap - $first_gap" | bc 2>/dev/null)
            echo "  $chunk_name: Gap $first_gap → $last_gap MB (growth: $gap_growth MB)"
        fi
    fi
done

echo ""

# ================================================================
# Final Report
# ================================================================
echo ""
echo "========================================================================"
echo "DIAGNOSTIC TRAINING COMPLETE"
echo "========================================================================"
echo "Exit code: $EXIT_CODE"
echo ""
echo "Logs:"
echo "  Main log: $MAIN_LOG"
echo "  GPU log:  $GPU_LOG"
echo "  Chunk logs: $RESULTS_DIR/chunk*_epochs_*.log"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully with periodic restarts"
    echo ""
    echo "VERDICT:"
    echo "  Periodic restart strategy successfully worked around memory leak"
    echo "  Training can proceed with this approach for full 100-epoch run"
else
    echo "❌ Training failed with exit code $EXIT_CODE"
    echo ""
    echo "NEXT STEPS:"
    echo "  1. Check chunk logs for failure point: ls -lth $RESULTS_DIR/chunk*.log"
    echo "  2. If OOM still occurs, reduce CHUNK_SIZE from $CHUNK_SIZE to 3"
    echo "  3. Consider reducing batch size as additional measure"
fi

echo ""
echo "========================================================================"
echo "ANALYSIS COMMANDS"
echo "========================================================================"
echo ""
echo "1. View memory growth across all chunks:"
echo "   grep -h 'Gap:' $RESULTS_DIR/chunk*.log | awk '{print \$2}' | head -30"
echo ""
echo "2. View GPU memory timeline:"
echo "   awk -F',' 'NR>1 {print NR-1, \$2}' $GPU_LOG | head -50"
echo ""
echo "3. Check for OOM in any chunk:"
echo "   grep -h 'out of memory' $RESULTS_DIR/chunk*.log"
echo ""
echo "4. View chunk completion status:"
echo "   tail -5 $RESULTS_DIR/chunk*.log"
echo ""

#Usage:
## Run the chunked training
#bash scripts/train_diagnostic_chunk4.sh
#
## Monitor in another terminal
#watch -n 5 'tail -20 ../ghop_production_chunked_results/diagnostic_chunk4_chunked_*.log'
