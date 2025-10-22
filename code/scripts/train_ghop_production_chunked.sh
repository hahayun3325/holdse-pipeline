#!/bin/bash
# File: code/scripts/train_ghop_production_chunked.sh
# PURPOSE: Train in 12-epoch chunks with proper checkpoint tracking
# USAGE: bash scripts/train_ghop_production_chunked.sh

set -e

cd ~/Projects/holdse/code

# ================================================================
# CRITICAL: Set CUDA memory configuration FIRST
# ================================================================
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# ================================================================
# CONFIGURATION
# ================================================================
CHUNK_SIZE=12  # Safe margin below epoch 17
TOTAL_EPOCHS=100
NUM_CHUNKS=$((TOTAL_EPOCHS / CHUNK_SIZE + 1))

RESULTS_DIR="../ghop_production_chunked_results"
mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$RESULTS_DIR/chunked_training_$TIMESTAMP.txt"
SUMMARY_FILE="$RESULTS_DIR/TRAINING_SUMMARY_$TIMESTAMP.txt"

# ================================================================
# NEW: TRAINING RUN ID TRACKING
# ================================================================
RUN_ID_FILE="$RESULTS_DIR/current_run_id_$TIMESTAMP.txt"
ALL_RUN_IDS_FILE="$RESULTS_DIR/all_run_ids_$TIMESTAMP.txt"

echo "========================================================================" | tee "$MASTER_LOG"
echo "CHUNKED TRAINING: $NUM_CHUNKS chunks of $CHUNK_SIZE epochs each" | tee -a "$MASTER_LOG"
echo "=========================================================================" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "Configuration:" | tee -a "$MASTER_LOG"
echo "  Chunk size: $CHUNK_SIZE epochs" | tee -a "$MASTER_LOG"
echo "  Target total: $TOTAL_EPOCHS epochs" | tee -a "$MASTER_LOG"
echo "  Number of chunks: $NUM_CHUNKS" | tee -a "$MASTER_LOG"
echo "  CUDA alloc config: $PYTORCH_CUDA_ALLOC_CONF" | tee -a "$MASTER_LOG"
echo "  Training session: $TIMESTAMP" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# Use the latest production config or create one
BASE_CONFIG="confs/ghop_quick_bottle_1.yaml"
TARGET_CONFIG="confs/ghop_production_chunked_$TIMESTAMP.yaml"

# Copy and modify base config (only once)
cp "$BASE_CONFIG" "$TARGET_CONFIG"

# ================================================================
# APPLY PHASE 5 FIX (UPDATED)
# ================================================================
echo "Applying Phase 5 memory and ordering fix..." | tee -a "$MASTER_LOG"

# Fix 1: Delay Phase 5 start to epoch 20
if grep -q "phase5_start_iter: 100" "$TARGET_CONFIG"; then
    sed -i 's/phase5_start_iter: 100/phase5_start_iter: 1100/' "$TARGET_CONFIG"
    echo "  ✓ Delayed Phase 5: epoch 2 → epoch 20 (step 1100)" | tee -a "$MASTER_LOG"
elif grep -q "phase5_start_iter: 600" "$TARGET_CONFIG"; then
    sed -i 's/phase5_start_iter: 600/phase5_start_iter: 1100/' "$TARGET_CONFIG"
    echo "  ✓ Delayed Phase 5: epoch 11 → epoch 20 (step 1100)" | tee -a "$MASTER_LOG"
else
    echo "  ⚠️  phase5_start_iter not found or already set" | tee -a "$MASTER_LOG"
fi

# Fix 2: Reduce temporal window
if grep -q "temporal_window: 5" "$TARGET_CONFIG"; then
    sed -i 's/temporal_window: 5/temporal_window: 3/' "$TARGET_CONFIG"
    echo "  ✓ Reduced temporal window: 5 → 3 frames" | tee -a "$MASTER_LOG"
fi

# Fix 3: Increase total_iterations to accommodate longer training
if grep -q "total_iterations: 1000" "$TARGET_CONFIG"; then
    sed -i 's/total_iterations: 1000/total_iterations: 6000/' "$TARGET_CONFIG"
    echo "  ✓ Increased total_iterations: 1000 → 6000" | tee -a "$MASTER_LOG"
fi

# Fix 4: Move finetune_start AFTER phase5_start
if grep -q "finetune_start_iter: 800" "$TARGET_CONFIG"; then
    sed -i 's/finetune_start_iter: 800/finetune_start_iter: 1200/' "$TARGET_CONFIG"
    echo "  ✓ Moved finetune_start: 800 → 1200 (after phase5)" | tee -a "$MASTER_LOG"
fi

echo "" | tee -a "$MASTER_LOG"

# ================================================================
# END PHASE 5 FIX
# ================================================================

# Apply all the usual fixes (KEEP THESE)
sed -i 's/num_epochs: 1/num_epochs: 100/' "$TARGET_CONFIG"
sed -i 's/max_steps: 20/max_steps: -1/' "$TARGET_CONFIG"
sed -i '/^dataset:/,/^[a-z_]*:/ s/batch_size: [0-9]\+/batch_size: 1/' "$TARGET_CONFIG"
sed -i 's/pixel_per_batch: [0-9]\+/pixel_per_batch: 512/' "$TARGET_CONFIG"

echo "✓ Created config: $TARGET_CONFIG" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# ================================================================
# STORE TRAINING START TIME
# ================================================================
TRAINING_START_TIME=$(date)
TRAINING_START_EPOCH_SECONDS=$(date +%s)

# ================================================================
# HELPER FUNCTION: Find latest checkpoint with highest epoch
# ================================================================
find_latest_checkpoint() {
    local session_start_time="$1"

    # Find all run directories created during this session
    # (modified after session start minus 1 minute for safety)
    local cutoff_time=$(date -d "$session_start_time - 1 minute" +%s 2>/dev/null || \
                       date -v-1M -j -f "%a %b %d %H:%M:%S %Z %Y" "$session_start_time" +%s 2>/dev/null)

    local best_ckpt=""
    local best_epoch=-1
    local best_run_id=""

    # Search all run directories for checkpoints
    for run_dir in logs/*/; do
        local run_id=$(basename "$run_dir")

        # Skip if not a valid run ID format (9 hex chars)
        if ! [[ "$run_id" =~ ^[a-f0-9]{9}$ ]]; then
            continue
        fi

        # Check if checkpoint exists
        local ckpt="$run_dir/checkpoints/last.ckpt"
        if [ ! -f "$ckpt" ]; then
            continue
        fi

        # Get checkpoint modification time
        local ckpt_mtime=$(stat -c %Y "$ckpt" 2>/dev/null || stat -f %m "$ckpt" 2>/dev/null)

        # Skip if checkpoint is older than session start
        if [ -n "$cutoff_time" ] && [ "$ckpt_mtime" -lt "$cutoff_time" ]; then
            continue
        fi

        # Get checkpoint epoch
        local ckpt_epoch=$(python -c "
import torch
try:
    ckpt = torch.load('$ckpt', map_location='cpu')
    print(ckpt.get('epoch', -1))
except Exception:
    print(-1)
" 2>/dev/null)

        # Update best if this epoch is higher
        if [ "$ckpt_epoch" -gt "$best_epoch" ]; then
            best_epoch=$ckpt_epoch
            best_ckpt=$ckpt
            best_run_id=$run_id
        fi
    done

    # Return results via echo (bash function return)
    if [ -n "$best_ckpt" ]; then
        echo "$best_ckpt|$best_epoch|$best_run_id"
        return 0
    else
        return 1
    fi
}

# ================================================================
# TRAIN IN CHUNKS
# ================================================================
for ((chunk=0; chunk<NUM_CHUNKS; chunk++)); do
    START_EPOCH=$((chunk * CHUNK_SIZE))
    END_EPOCH=$(((chunk + 1) * CHUNK_SIZE))

    if [ $END_EPOCH -gt $TOTAL_EPOCHS ]; then
        END_EPOCH=$TOTAL_EPOCHS
    fi

    echo "========================================================================" | tee -a "$MASTER_LOG"
    echo "CHUNK $((chunk + 1))/$NUM_CHUNKS: Epochs $START_EPOCH → $END_EPOCH" | tee -a "$MASTER_LOG"
    echo "========================================================================" | tee -a "$MASTER_LOG"

    CHUNK_START_TIME=$(date)
    echo "Start time: $CHUNK_START_TIME" | tee -a "$MASTER_LOG"
    echo "" | tee -a "$MASTER_LOG"

    CHUNK_LOG="$RESULTS_DIR/chunk_${chunk}_epochs_${START_EPOCH}_${END_EPOCH}_$(date +%Y%m%d_%H%M%S).log"

    # ================================================================
    # IMPROVED: Find checkpoint with HIGHEST epoch number
    # ================================================================
    if [ $chunk -eq 0 ]; then
        # First chunk: Start fresh
        echo "Starting from scratch (chunk 0)" | tee -a "$MASTER_LOG"
        echo "This will create a new training run" | tee -a "$MASTER_LOG"
        RESUME_ARG=""
    else
        # Later chunks: Find checkpoint with highest epoch
        echo "Searching for checkpoint with highest epoch..." | tee -a "$MASTER_LOG"

        # Use helper function to find best checkpoint
        CKPT_SEARCH_RESULT=$(find_latest_checkpoint "$TRAINING_START_TIME")

        if [ $? -eq 0 ] && [ -n "$CKPT_SEARCH_RESULT" ]; then
            # Parse result
            LAST_CKPT=$(echo "$CKPT_SEARCH_RESULT" | cut -d'|' -f1)
            CKPT_EPOCH=$(echo "$CKPT_SEARCH_RESULT" | cut -d'|' -f2)
            CKPT_RUN_ID=$(echo "$CKPT_SEARCH_RESULT" | cut -d'|' -f3)

            echo "✓ Found checkpoint with highest epoch:" | tee -a "$MASTER_LOG"
            echo "  Checkpoint: $LAST_CKPT" | tee -a "$MASTER_LOG"
            echo "  Epoch: $CKPT_EPOCH" | tee -a "$MASTER_LOG"
            echo "  Run ID: $CKPT_RUN_ID" | tee -a "$MASTER_LOG"

            # Update run ID tracking
            echo "$CKPT_RUN_ID" >> "$ALL_RUN_IDS_FILE"
            echo "$CKPT_RUN_ID" > "$RUN_ID_FILE"
            CURRENT_RUN_ID="$CKPT_RUN_ID"

            # Sanity check: epoch should be close to expected
            EXPECTED_EPOCH=$((chunk * CHUNK_SIZE))
            EPOCH_DIFF=$((CKPT_EPOCH - EXPECTED_EPOCH))

            echo "" | tee -a "$MASTER_LOG"
            echo "Checkpoint validation:" | tee -a "$MASTER_LOG"
            echo "  Expected epoch: ~$EXPECTED_EPOCH" | tee -a "$MASTER_LOG"
            echo "  Checkpoint epoch: $CKPT_EPOCH" | tee -a "$MASTER_LOG"
            echo "  Difference: $EPOCH_DIFF epochs" | tee -a "$MASTER_LOG"

            if [ $EPOCH_DIFF -lt -5 ] || [ $EPOCH_DIFF -gt 5 ]; then
                echo "" | tee -a "$MASTER_LOG"
                echo "⚠️  WARNING: Large epoch mismatch detected!" | tee -a "$MASTER_LOG"
                echo "   This may indicate:" | tee -a "$MASTER_LOG"
                echo "   1. Previous chunk failed before saving checkpoint" | tee -a "$MASTER_LOG"
                echo "   2. Checkpoint corruption" | tee -a "$MASTER_LOG"
                echo "   3. Training configuration changed" | tee -a "$MASTER_LOG"
                echo "" | tee -a "$MASTER_LOG"

                # Check if previous chunk had OOM
                PREV_CHUNK_LOG=$(ls -t "$RESULTS_DIR"/chunk_$((chunk - 1))_*.log 2>/dev/null | head -1)
                if [ -n "$PREV_CHUNK_LOG" ] && grep -qE "out of memory|OutOfMemoryError|OOM" "$PREV_CHUNK_LOG"; then
                    echo "❌ DETECTED: Previous chunk (chunk $((chunk - 1))) failed with OOM" | tee -a "$MASTER_LOG"
                    echo "   The OOM failure prevented checkpoint save" | tee -a "$MASTER_LOG"
                    echo "   Using best available checkpoint (epoch $CKPT_EPOCH)" | tee -a "$MASTER_LOG"
                    echo "" | tee -a "$MASTER_LOG"
                    echo "   ACTION REQUIRED:" | tee -a "$MASTER_LOG"
                    echo "   - This means we lost progress from chunk $((chunk - 1))" | tee -a "$MASTER_LOG"
                    echo "   - Training will resume from epoch $((CKPT_EPOCH + 1))" | tee -a "$MASTER_LOG"
                    echo "   - Chunk $((chunk - 1)) will be re-trained" | tee -a "$MASTER_LOG"
                    echo "" | tee -a "$MASTER_LOG"
                else
                    read -p "Continue anyway? (y/N) " -n 1 -r
                    echo
                    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                        echo "Aborted by user" | tee -a "$MASTER_LOG"
                        exit 1
                    fi
                fi
            fi

            RESUME_ARG="--load_ckpt $LAST_CKPT"
        else
            # No checkpoint found for chunk > 0 is an error
            echo "❌ ERROR: No checkpoint found for chunk $chunk!" | tee -a "$MASTER_LOG"
            echo "   Searched all run directories in logs/" | tee -a "$MASTER_LOG"
            echo "   Created since: $TRAINING_START_TIME" | tee -a "$MASTER_LOG"
            echo "" | tee -a "$MASTER_LOG"
            echo "Available run directories:" | tee -a "$MASTER_LOG"
            ls -lt logs/ | head -10 | tee -a "$MASTER_LOG"
            exit 1
        fi
    fi

    echo "" | tee -a "$MASTER_LOG"

    # ================================================================
    # GPU MEMORY STATUS BEFORE TRAINING
    # ================================================================
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU status before training:" | tee -a "$MASTER_LOG"
        nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits | \
            awk '{print "  Used: "$1" MB, Free: "$2" MB"}' | tee -a "$MASTER_LOG"
        echo "" | tee -a "$MASTER_LOG"
    fi

    # ================================================================
    # Run training - LET IT EXIT NATURALLY
    # ================================================================
    CHUNK_START_SECONDS=$(date +%s)

    python train.py \
        --config "$TARGET_CONFIG" \
        --case ghop_bottle_1 \
        --use_ghop \
        --gpu_id 0 \
        --num_epoch $END_EPOCH \
        --no-pin-memory \
        --no-comet \
        $RESUME_ARG \
        2>&1 | tee "$CHUNK_LOG"

    exitcode=$?

    CHUNK_END_SECONDS=$(date +%s)
    CHUNK_DURATION=$((CHUNK_END_SECONDS - CHUNK_START_SECONDS))
    CHUNK_DURATION_MINS=$((CHUNK_DURATION / 60))

    echo "" | tee -a "$MASTER_LOG"
    echo "Chunk $chunk exit code: $exitcode" | tee -a "$MASTER_LOG"
    echo "End time: $(date)" | tee -a "$MASTER_LOG"
    echo "Duration: $CHUNK_DURATION_MINS minutes ($CHUNK_DURATION seconds)" | tee -a "$MASTER_LOG"

    # ================================================================
    # AFTER EACH CHUNK: Update run ID tracking
    # ================================================================
    echo "" | tee -a "$MASTER_LOG"
    echo "Post-chunk analysis:" | tee -a "$MASTER_LOG"

    # Find newest run directory
    NEWEST_RUN_ID=$(ls -t logs/ | grep -E '^[a-f0-9]{9}$' | head -1)

    if [ -n "$NEWEST_RUN_ID" ]; then
        echo "  Newest run directory: $NEWEST_RUN_ID" | tee -a "$MASTER_LOG"
        echo "$NEWEST_RUN_ID" >> "$ALL_RUN_IDS_FILE"

        # Check if checkpoint exists in newest directory
        if [ -f "logs/$NEWEST_RUN_ID/checkpoints/last.ckpt" ]; then
            NEWEST_CKPT_EPOCH=$(python -c "
import torch
try:
    ckpt = torch.load('logs/$NEWEST_RUN_ID/checkpoints/last.ckpt', map_location='cpu')
    print(ckpt.get('epoch', -1))
except Exception:
    print(-1)
" 2>/dev/null)

            echo "  Checkpoint found: epoch $NEWEST_CKPT_EPOCH" | tee -a "$MASTER_LOG"

            # Update current run ID to newest with checkpoint
            echo "$NEWEST_RUN_ID" > "$RUN_ID_FILE"
            CURRENT_RUN_ID="$NEWEST_RUN_ID"
        else
            echo "  ⚠️  WARNING: No checkpoint in newest directory!" | tee -a "$MASTER_LOG"
            echo "     This indicates the chunk may have failed" | tee -a "$MASTER_LOG"
        fi
    fi

    echo "" | tee -a "$MASTER_LOG"

    # ================================================================
    # Check success/failure - IMPROVED OOM detection
    # ================================================================
    CHUNK_FAILED=false

    if [ $exitcode -ne 0 ]; then
        echo "❌ Chunk $chunk failed with non-zero exit code!" | tee -a "$MASTER_LOG"
        CHUNK_FAILED=true
    fi

    # Check for OOM even if exit code is 0
    if grep -qE "out of memory|OutOfMemoryError|OOM" "$CHUNK_LOG"; then
        failed_epoch=$(grep -oP 'Epoch \K[0-9]+' "$CHUNK_LOG" | tail -1)
        echo "❌ Chunk $chunk failed with OOM!" | tee -a "$MASTER_LOG"
        echo "  Failed at epoch: $failed_epoch" | tee -a "$MASTER_LOG"
        echo "  This should not happen with 12-epoch chunks!" | tee -a "$MASTER_LOG"
        echo "  Check if CUDA alloc config was applied" | tee -a "$MASTER_LOG"
        CHUNK_FAILED=true

        # Special case: OOM at epoch 17
        if [ "$failed_epoch" == "17" ]; then
            echo "" | tee -a "$MASTER_LOG"
            echo "⚠️  CRITICAL: OOM at epoch 17 despite all fixes!" | tee -a "$MASTER_LOG"
            echo "   Verification checklist:" | tee -a "$MASTER_LOG"
            echo "   1. CUDA config: $PYTORCH_CUDA_ALLOC_CONF" | tee -a "$MASTER_LOG"
            echo "   2. Check torch.zeros fix in src/model/mano/server.py" | tee -a "$MASTER_LOG"
            echo "   3. Verify no other processes using GPU" | tee -a "$MASTER_LOG"
            echo "" | tee -a "$MASTER_LOG"
        fi
    fi

    if [ "$CHUNK_FAILED" = true ]; then
        echo "" | tee -a "$MASTER_LOG"
        echo "Training stopped due to chunk failure" | tee -a "$MASTER_LOG"
        break
    fi

    echo "✅ Chunk $chunk complete" | tee -a "$MASTER_LOG"
    echo "" | tee -a "$MASTER_LOG"

    # ================================================================
    # PROPER inter-chunk cleanup (NO force kill)
    # ================================================================
    if [ $chunk -lt $((NUM_CHUNKS - 1)) ]; then
        echo "Cleaning up before next chunk..." | tee -a "$MASTER_LOG"

        python -c "
import torch
import gc

gc.collect()
torch.cuda.empty_cache()

if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**2
    print(f'✓ Memory cleanup: {allocated:.2f} MB allocated')
" | tee -a "$MASTER_LOG"

        sleep 5

        echo "Ready for next chunk" | tee -a "$MASTER_LOG"
        echo "" | tee -a "$MASTER_LOG"
    fi

    # Check if reached target
    if [ $END_EPOCH -ge $TOTAL_EPOCHS ]; then
        echo "" | tee -a "$MASTER_LOG"
        echo "========================================================================" | tee -a "$MASTER_LOG"
        echo "✅ REACHED TARGET: $TOTAL_EPOCHS epochs" | tee -a "$MASTER_LOG"
        echo "========================================================================" | tee -a "$MASTER_LOG"
        break
    fi
done

# ================================================================
# CALCULATE TOTAL TRAINING TIME
# ================================================================
TRAINING_END_TIME=$(date)
TRAINING_END_EPOCH_SECONDS=$(date +%s)
TOTAL_DURATION=$((TRAINING_END_EPOCH_SECONDS - TRAINING_START_EPOCH_SECONDS))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINS=$(((TOTAL_DURATION % 3600) / 60))
TOTAL_SECS=$((TOTAL_DURATION % 60))

# ================================================================
# FINAL SUMMARY IN MASTER LOG
# ================================================================
echo "" | tee -a "$MASTER_LOG"
echo "========================================================================" | tee -a "$MASTER_LOG"
echo "CHUNKED TRAINING COMPLETE" | tee -a "$MASTER_LOG"
echo "========================================================================" | tee -a "$MASTER_LOG"
echo "End: $TRAINING_END_TIME" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# Count successful chunks
successful_chunks=0
for ((i=0; i<NUM_CHUNKS; i++)); do
    chunk_log=$(ls -t "$RESULTS_DIR"/chunk_${i}_*.log 2>/dev/null | head -1)
    if [ -n "$chunk_log" ] && ! grep -qE "out of memory|OutOfMemoryError|OOM" "$chunk_log"; then
        successful_chunks=$((successful_chunks + 1))
    else
        break
    fi
done

echo "Summary:" | tee -a "$MASTER_LOG"
echo "  Total chunks: $NUM_CHUNKS" | tee -a "$MASTER_LOG"
echo "  Successful: $successful_chunks" | tee -a "$MASTER_LOG"
echo "  Epochs completed: $((successful_chunks * CHUNK_SIZE))" | tee -a "$MASTER_LOG"
echo "  Failed: $((NUM_CHUNKS - successful_chunks))" | tee -a "$MASTER_LOG"

if [ -f "$RUN_ID_FILE" ]; then
    FINAL_RUN_ID=$(cat "$RUN_ID_FILE")
    echo "  Training run ID: $FINAL_RUN_ID" | tee -a "$MASTER_LOG"
    echo "  Checkpoint location: logs/$FINAL_RUN_ID/checkpoints/" | tee -a "$MASTER_LOG"
fi

echo "" | tee -a "$MASTER_LOG"

if [ $successful_chunks -eq $NUM_CHUNKS ]; then
    TRAINING_STATUS="SUCCESS"
    echo "✅ ALL CHUNKS COMPLETED SUCCESSFULLY!" | tee -a "$MASTER_LOG"
elif [ $((successful_chunks * CHUNK_SIZE)) -ge $TOTAL_EPOCHS ]; then
    TRAINING_STATUS="SUCCESS"
    echo "✅ TARGET EPOCHS REACHED!" | tee -a "$MASTER_LOG"
else
    TRAINING_STATUS="INCOMPLETE"
    echo "⚠️  Some chunks failed" | tee -a "$MASTER_LOG"
    echo "   Completed: $((successful_chunks * CHUNK_SIZE)) epochs" | tee -a "$MASTER_LOG"
    echo "   Can resume from last checkpoint" | tee -a "$MASTER_LOG"
fi

echo "" | tee -a "$MASTER_LOG"
echo "📁 Master log: $MASTER_LOG" | tee -a "$MASTER_LOG"
echo "📁 Chunk logs: $RESULTS_DIR/chunk_*.log" | tee -a "$MASTER_LOG"
echo "========================================================================" | tee -a "$MASTER_LOG"

# ================================================================
# NEW: CREATE COMPREHENSIVE TRAINING SUMMARY FILE
# ================================================================
echo "Creating training summary file..." | tee -a "$MASTER_LOG"

cat > "$SUMMARY_FILE" << EOF
================================================================================
                    HOLDSE TRAINING SESSION SUMMARY
================================================================================

Generated: $(date)
Training Session ID: $TIMESTAMP

================================================================================
QUICK REFERENCE
================================================================================

Training Run ID:        ${CURRENT_RUN_ID:-"Not available"}
Training Status:        $TRAINING_STATUS
Epochs Completed:       $((successful_chunks * CHUNK_SIZE)) / $TOTAL_EPOCHS
Successful Chunks:      $successful_chunks / $NUM_CHUNKS

Checkpoint Location:    logs/${CURRENT_RUN_ID:-"unknown"}/checkpoints/
Final Checkpoint:       logs/${CURRENT_RUN_ID:-"unknown"}/checkpoints/last.ckpt

Config File:            $TARGET_CONFIG
Master Log:             $MASTER_LOG

================================================================================
TRAINING CONFIGURATION
================================================================================

Strategy:               12-epoch chunked training
Chunk Size:             $CHUNK_SIZE epochs per chunk
Total Chunks:           $NUM_CHUNKS
Target Epochs:          $TOTAL_EPOCHS

CUDA Configuration:     PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF
Memory Fixes Applied:
  ✓ CUDA fragmentation prevention (max_split_size_mb:128)
  ✓ torch.zeros explicit allocation
  ✓ 12-epoch safe chunking
  ✓ Natural process exit (no force-kill)

Dataset:                ghop_bottle_1
Batch Size:             1
Pixels Per Batch:       512
GPU ID:                 0

================================================================================
TRAINING TIMELINE
================================================================================

Start Time:             $TRAINING_START_TIME
End Time:               $TRAINING_END_TIME
Total Duration:         ${TOTAL_HOURS}h ${TOTAL_MINS}m ${TOTAL_SECS}s

EOF

# Add per-chunk breakdown
echo "" >> "$SUMMARY_FILE"
echo "Chunk-by-Chunk Breakdown:" >> "$SUMMARY_FILE"
echo "-------------------------" >> "$SUMMARY_FILE"

for ((i=0; i<NUM_CHUNKS; i++)); do
    chunk_log=$(ls -t "$RESULTS_DIR"/chunk_${i}_*.log 2>/dev/null | head -1)

    if [ -n "$chunk_log" ]; then
        # Extract timing info if available
        chunk_start=$(grep "Start time:" "$chunk_log" | head -1 | cut -d':' -f2- | xargs)
        chunk_end=$(grep "End time:" "$chunk_log" | head -1 | cut -d':' -f2- | xargs)

        # Check for OOM
        if grep -qE "out of memory|OutOfMemoryError|OOM" "$chunk_log"; then
            status="❌ FAILED (OOM)"
        elif grep -q "exit code: 0" "$chunk_log"; then
            status="✅ SUCCESS"
        else
            status="❌ FAILED"
        fi

        echo "  Chunk $i: Epochs $((i * CHUNK_SIZE)) → $(((i + 1) * CHUNK_SIZE))" >> "$SUMMARY_FILE"
        echo "    Status: $status" >> "$SUMMARY_FILE"
        if [ -n "$chunk_start" ]; then
            echo "    Started: $chunk_start" >> "$SUMMARY_FILE"
        fi
        if [ -n "$chunk_end" ]; then
            echo "    Ended: $chunk_end" >> "$SUMMARY_FILE"
        fi
        echo "    Log: $chunk_log" >> "$SUMMARY_FILE"
        echo "" >> "$SUMMARY_FILE"
    else
        echo "  Chunk $i: Not started" >> "$SUMMARY_FILE"
        echo "" >> "$SUMMARY_FILE"
    fi
done

# Add checkpoint information
cat >> "$SUMMARY_FILE" << EOF

================================================================================
CHECKPOINT INFORMATION
================================================================================

Run ID Tracking File:   $RUN_ID_FILE
Training Run ID:        ${CURRENT_RUN_ID:-"Not determined"}

EOF

if [ -n "$CURRENT_RUN_ID" ] && [ -d "logs/$CURRENT_RUN_ID" ]; then
    echo "Checkpoint Directory:   logs/$CURRENT_RUN_ID/" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"

    echo "Available Checkpoints:" >> "$SUMMARY_FILE"
    if [ -d "logs/$CURRENT_RUN_ID/checkpoints" ]; then
        ls -lh "logs/$CURRENT_RUN_ID/checkpoints/" >> "$SUMMARY_FILE" 2>/dev/null || echo "  (none found)" >> "$SUMMARY_FILE"
    else
        echo "  Checkpoint directory not found" >> "$SUMMARY_FILE"
    fi

    echo "" >> "$SUMMARY_FILE"

    # Try to read checkpoint info
    if [ -f "logs/$CURRENT_RUN_ID/checkpoints/last.ckpt" ]; then
        FINAL_CKPT_INFO=$(python -c "
import torch
try:
    ckpt = torch.load('logs/$CURRENT_RUN_ID/checkpoints/last.ckpt', map_location='cpu')
    print(f\"  Checkpoint Epoch: {ckpt.get('epoch', 'N/A')}\")
    print(f\"  Global Step: {ckpt.get('global_step', 'N/A')}\")

    # Get model state dict size
    if 'state_dict' in ckpt:
        num_params = len(ckpt['state_dict'])
        print(f\"  Model Parameters: {num_params}\")
except Exception as e:
    print(f\"  Could not read checkpoint: {e}\")
" 2>/dev/null)

        echo "Final Checkpoint Info:" >> "$SUMMARY_FILE"
        echo "$FINAL_CKPT_INFO" >> "$SUMMARY_FILE"
    fi
else
    echo "Training run ID not available or directory not found" >> "$SUMMARY_FILE"
fi

# Add GPU information
cat >> "$SUMMARY_FILE" << EOF

================================================================================
GPU INFORMATION
================================================================================

EOF

if command -v nvidia-smi &> /dev/null; then
    echo "GPU Configuration:" >> "$SUMMARY_FILE"
    nvidia-smi --query-gpu=name,memory.total,driver_version,cuda_version --format=csv,noheader | \
        awk '{print "  GPU: "$1"\n  Total Memory: "$2"\n  Driver Version: "$3"\n  CUDA Version: "$4}' >> "$SUMMARY_FILE"

    echo "" >> "$SUMMARY_FILE"
    echo "Final GPU Memory Status:" >> "$SUMMARY_FILE"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits | \
        awk '{print "  Used: "$1" MB\n  Free: "$2" MB"}' >> "$SUMMARY_FILE"
else
    echo "nvidia-smi not available" >> "$SUMMARY_FILE"
fi

# Add file locations
cat >> "$SUMMARY_FILE" << EOF

================================================================================
OUTPUT FILES AND LOGS
================================================================================

Results Directory:      $RESULTS_DIR/

Master Log:             $MASTER_LOG
Summary File:           $SUMMARY_FILE (this file)
Run ID File:            $RUN_ID_FILE

Chunk Logs:
EOF

for ((i=0; i<NUM_CHUNKS; i++)); do
    chunk_log=$(ls -t "$RESULTS_DIR"/chunk_${i}_*.log 2>/dev/null | head -1)
    if [ -n "$chunk_log" ]; then
        echo "  Chunk $i: $chunk_log" >> "$SUMMARY_FILE"
    fi
done

# Add training outcome and next steps
cat >> "$SUMMARY_FILE" << EOF

================================================================================
TRAINING OUTCOME
================================================================================

Status:                 $TRAINING_STATUS
Epochs Completed:       $((successful_chunks * CHUNK_SIZE)) / $TOTAL_EPOCHS
Success Rate:           $successful_chunks / $NUM_CHUNKS chunks ($((successful_chunks * 100 / NUM_CHUNKS))%)

EOF

if [ "$TRAINING_STATUS" == "SUCCESS" ]; then
    cat >> "$SUMMARY_FILE" << EOF
✅ TRAINING COMPLETED SUCCESSFULLY

Next Steps:
  1. Validate the trained model on test set
  2. Run inference on new data
  3. Backup the checkpoint to permanent storage

  Model checkpoint: logs/$CURRENT_RUN_ID/checkpoints/last.ckpt

EOF
else
    cat >> "$SUMMARY_FILE" << EOF
⚠️  TRAINING INCOMPLETE

To Resume Training:
  1. Check the last successful chunk log for errors
  2. Ensure GPU has sufficient memory
  3. Resume from last checkpoint:

     bash scripts/train_ghop_production_chunked.sh

  The script will automatically detect and resume from:
     logs/$CURRENT_RUN_ID/checkpoints/last.ckpt

EOF
fi

# Add troubleshooting section
cat >> "$SUMMARY_FILE" << EOF

================================================================================
TROUBLESHOOTING
================================================================================

If training failed:

1. Check GPU memory usage:
   nvidia-smi

2. Review the last chunk log for errors:
   tail -100 $RESULTS_DIR/chunk_$((successful_chunks))_*.log

3. Verify CUDA configuration:
   echo \$PYTORCH_CUDA_ALLOC_CONF

4. Check checkpoint integrity:
   python -c "import torch; ckpt = torch.load('logs/$CURRENT_RUN_ID/checkpoints/last.ckpt'); print(ckpt.keys())"

5. View master log:
   less $MASTER_LOG

Common Issues:
  - OOM at epoch 17: CUDA config not set (should be max_split_size_mb:128)
  - Wrong checkpoint loaded: Check run ID in $RUN_ID_FILE
  - Checkpoint not found: Verify logs/$CURRENT_RUN_ID/checkpoints/ exists

================================================================================
SYSTEM INFORMATION
================================================================================

Hostname:               $(hostname)
User:                   $(whoami)
Working Directory:      $(pwd)
Python Version:         $(python --version 2>&1)
PyTorch Version:        $(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "Not available")

Environment Variables:
  PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF
  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-"Not set"}

================================================================================
END OF SUMMARY
================================================================================

For detailed logs, see: $MASTER_LOG
For per-chunk logs, see: $RESULTS_DIR/chunk_*.log

Generated at: $(date)
================================================================================
EOF

echo "✅ Training summary saved to: $SUMMARY_FILE" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# Display summary file location prominently
echo "========================================================================" | tee -a "$MASTER_LOG"
echo "📄 QUICK ACCESS TO TRAINING INFO:" | tee -a "$MASTER_LOG"
echo "   Summary file: $SUMMARY_FILE" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "   View with: cat $SUMMARY_FILE" | tee -a "$MASTER_LOG"
echo "   Or:        less $SUMMARY_FILE" | tee -a "$MASTER_LOG"
echo "========================================================================" | tee -a "$MASTER_LOG"
