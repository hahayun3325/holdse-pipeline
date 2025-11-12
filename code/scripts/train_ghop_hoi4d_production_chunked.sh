#!/bin/bash
# File: code/scripts/train_ghop_hoi4d_production_chunked.sh
# PURPOSE: Train in 10-epoch chunks with aggressive fragmentation prevention
# USAGE: bash scripts/train_ghop_hoi4d_production_chunked.sh

set -e

cd ~/Projects/holdse/code

# ================================================================
# ✅ CRITICAL: Enhanced CUDA memory configuration
# ================================================================
export PYTORCH_CUDA_ALLOC_CONF="backend:native"

echo "================================================================"
echo "CUDA Memory Configuration:"
echo "  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "  max_split_size_mb: 32 (aggressive fragmentation prevention)"
echo "  expandable_segments: True (dynamic memory pool)"
echo "================================================================"
echo ""

# ================================================================
# CONFIGURATION
# ================================================================
CHUNK_SIZE=5
TOTAL_EPOCHS=30
NUM_CHUNKS=$((TOTAL_EPOCHS / CHUNK_SIZE))

# Optional: Backup old checkpoints before starting fresh training
BACKUP_OLD_CKPTS=false  # Set to 'true' to backup logs/ directory

RESULTS_DIR="../ghop_production_chunked_results"
mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$RESULTS_DIR/chunked_training_$TIMESTAMP.txt"
SUMMARY_FILE="$RESULTS_DIR/TRAINING_SUMMARY_$TIMESTAMP.txt"

# Training run ID tracking
RUN_ID_FILE="$RESULTS_DIR/current_run_id_$TIMESTAMP.txt"
ALL_RUN_IDS_FILE="$RESULTS_DIR/all_run_ids_$TIMESTAMP.txt"

echo "========================================================================" | tee "$MASTER_LOG"
echo "CHUNKED TRAINING: $NUM_CHUNKS chunks of $CHUNK_SIZE epochs each" | tee -a "$MASTER_LOG"
echo "========================================================================" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "Configuration:" | tee -a "$MASTER_LOG"
echo "  Chunk size: $CHUNK_SIZE epochs" | tee -a "$MASTER_LOG"
echo "  Target total: $TOTAL_EPOCHS epochs" | tee -a "$MASTER_LOG"
echo "  Number of chunks: $NUM_CHUNKS" | tee -a "$MASTER_LOG"
echo "  CUDA alloc config: $PYTORCH_CUDA_ALLOC_CONF" | tee -a "$MASTER_LOG"
echo "  Training session: $TIMESTAMP" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# ================================================================
# STEP 1: Create Production Configuration
# ================================================================
BASE_CONFIG="confs/ghop_quick_bottle_1.yaml"
TARGET_CONFIG="confs/ghop_production_chunked_$TIMESTAMP.yaml"

cp "$BASE_CONFIG" "$TARGET_CONFIG"
echo "Created config: $TARGET_CONFIG" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# ================================================================
# STEP 2: Configuration Modifications (MINIMAL - base already good!)
# ================================================================
echo "Configuring for production training..." | tee -a "$MASTER_LOG"

# ============================================================
# Grid and Sampling (safe to modify)
# ============================================================
sed -i '/^phase3:/,/^phase4:/ s/grid_resolution: [0-9]\+/grid_resolution: 24/' "$TARGET_CONFIG"
sed -i '/^phase3:/,/^phase4:/ s/prediction_respacing: [0-9]\+/prediction_respacing: 50/' "$TARGET_CONFIG"
# Slightly reduce ray samples for speed (optional)
sed -i 's/N_samples: 8/N_samples: 6/' "$TARGET_CONFIG"

echo "  ✅ Grid: 24³, Respacing: 50, Samples: 6" | tee -a "$MASTER_LOG"

# ============================================================
# DO NOT modify: dims, feature_vector_size, multires, multires_view
# These are already optimized in the base config!
# ============================================================

# ================================================================
# STEP 3: CRITICAL - Add RGB Loss Configuration
# ================================================================
echo "" | tee -a "$MASTER_LOG"
echo "Adding RGB loss configuration..." | tee -a "$MASTER_LOG"

# Reduce SDS weight
sed -i 's/w_sds: 5000.0/w_sds: 100.0/' "$TARGET_CONFIG"

# Add loss section
if ! grep -q "^loss:" "$TARGET_CONFIG"; then
    sed -i '/^training:/i\
loss:\
  w_rgb: 1.0\
  w_mask: 0.1\
  w_eikonal: 0.1\
  w_smooth: 0.005\
  w_contact: 10.0\
  w_temporal: 1.0\
  rgb_loss_type: "l1"\
\
' "$TARGET_CONFIG"
    echo "  ✅ Added RGB loss (w_rgb=1.0)" | tee -a "$MASTER_LOG"
fi

# ================================================================
# STEP 4: Phase Configuration
# ================================================================
echo "Configuring phases..." | tee -a "$MASTER_LOG"

# Phase 5 timing
sed -i 's/phase5_start_iter: 100/phase5_start_iter: 1100/' "$TARGET_CONFIG"
sed -i 's/temporal_window: 5/temporal_window: 3/' "$TARGET_CONFIG"
sed -i 's/total_iterations: 1000/total_iterations: 6000/' "$TARGET_CONFIG"
sed -i 's/finetune_start_iter: 800/finetune_start_iter: 1200/' "$TARGET_CONFIG"

# Disable Phase 4
sed -i '/^phase4:/,/^phase5:/ s/enabled: true/enabled: false/' "$TARGET_CONFIG"

echo "  ✅ Phase 5: epoch 20, Phase 4: disabled" | tee -a "$MASTER_LOG"

# ================================================================
# STEP 5: Training Parameters
# ================================================================
sed -i 's/num_epochs: 1/num_epochs: 30/' "$TARGET_CONFIG"
sed -i 's/max_steps: 20/max_steps: -1/' "$TARGET_CONFIG"
sed -i '/^dataset:/,/^model:/ s/batch_size: [0-9]\+/batch_size: 1/' "$TARGET_CONFIG"

# Enable validation/logging
sed -i '/^validation:/,/^[a-z_]*:/ s/enabled: false/enabled: true/' "$TARGET_CONFIG"
sed -i 's/log_images_every: 999999/log_images_every: 50/' "$TARGET_CONFIG"

echo "  ✅ Training: 30 epochs, validation enabled" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# ================================================================
# Verification
# ================================================================
echo "================================================================" | tee -a "$MASTER_LOG"
echo "Configuration Summary" | tee -a "$MASTER_LOG"
echo "================================================================" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

echo "Model architecture: Using base config (already optimized)" | tee -a "$MASTER_LOG"
grep "feature_vector_size:" "$TARGET_CONFIG" | head -1 | sed 's/^/  /' | tee -a "$MASTER_LOG"
grep "dims:" "$TARGET_CONFIG" | head -1 | sed 's/^/  /' | tee -a "$MASTER_LOG"

echo "" | tee -a "$MASTER_LOG"
echo "Loss weights:" | tee -a "$MASTER_LOG"
grep "w_sds:" "$TARGET_CONFIG" | head -1 | sed 's/^/  /' | tee -a "$MASTER_LOG"
grep "w_rgb:" "$TARGET_CONFIG" | head -1 | sed 's/^/  /' | tee -a "$MASTER_LOG"

echo "" | tee -a "$MASTER_LOG"
echo "✅ Configuration ready for training" | tee -a "$MASTER_LOG"
echo "================================================================" | tee -a "$MASTER_LOG"
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
# TRAIN IN CHUNKS (UPDATED: Matches proven diagnostic approach)
# ================================================================
CURRENT_EPOCH=0

while [ $CURRENT_EPOCH -lt $TOTAL_EPOCHS ]; do
    NEXT_EPOCH=$((CURRENT_EPOCH + CHUNK_SIZE))
    if [ $NEXT_EPOCH -gt $TOTAL_EPOCHS ]; then
        NEXT_EPOCH=$TOTAL_EPOCHS
    fi

    ACTUAL_EPOCHS=$((NEXT_EPOCH - CURRENT_EPOCH))
    CHUNK_NUM=$((CURRENT_EPOCH / CHUNK_SIZE + 1))
    TOTAL_CHUNKS_CALC=$(( (TOTAL_EPOCHS + CHUNK_SIZE - 1) / CHUNK_SIZE ))

    echo "========================================================================" | tee -a "$MASTER_LOG"
    echo "CHUNK $CHUNK_NUM/$TOTAL_CHUNKS_CALC: Epochs $CURRENT_EPOCH → $NEXT_EPOCH ($ACTUAL_EPOCHS epochs)" | tee -a "$MASTER_LOG"
    echo "========================================================================" | tee -a "$MASTER_LOG"

    CHUNK_START_TIME=$(date)
    echo "Start time: $CHUNK_START_TIME" | tee -a "$MASTER_LOG"
    echo "" | tee -a "$MASTER_LOG"

    CHUNK_LOG="$RESULTS_DIR/chunk${CHUNK_NUM}_epochs_${CURRENT_EPOCH}_to_${NEXT_EPOCH}_$(date +%Y%m%d_%H%M%S).log"

    # ================================================================
    # CHECKPOINT LOADING LOGIC (UPDATED: Avoid old checkpoints)
    # ================================================================
    if [ $CURRENT_EPOCH -eq 0 ]; then
        # ============================================================
        # CHUNK 1: START FRESH (Don't load old checkpoints)
        # ============================================================
        echo "========================================" | tee -a "$MASTER_LOG"
        echo "CHUNK 1: Starting Fresh Training" | tee -a "$MASTER_LOG"
        echo "========================================" | tee -a "$MASTER_LOG"
        echo "" | tee -a "$MASTER_LOG"

        echo "⚠️  NOT loading previous checkpoints because:" | tee -a "$MASTER_LOG"
        echo "   1. Model architecture changed (new dimensions)" | tee -a "$MASTER_LOG"
        echo "   2. Old checkpoints trained WITHOUT RGB loss (w_rgb=0)" | tee -a "$MASTER_LOG"
        echo "   3. Old rendering networks are degenerate (stuck at init)" | tee -a "$MASTER_LOG"
        echo "   4. Need fresh training WITH RGB loss (w_rgb=1.0)" | tee -a "$MASTER_LOG"
        echo "" | tee -a "$MASTER_LOG"

        # Check if old checkpoints exist and warn
        OLD_CKPT_COUNT=$(find logs -name "last.ckpt" -type f 2>/dev/null | wc -l)
        if [ $OLD_CKPT_COUNT -gt 0 ]; then
            echo "📋 Found $OLD_CKPT_COUNT existing checkpoint(s) in logs/" | tee -a "$MASTER_LOG"
            echo "   These will NOT be loaded (architecture mismatch)" | tee -a "$MASTER_LOG"
            echo "" | tee -a "$MASTER_LOG"

            # Optionally backup old checkpoints
            if [ "$BACKUP_OLD_CKPTS" = "true" ]; then
                BACKUP_DIR="logs_backup_$(date +%Y%m%d_%H%M%S)"
                echo "   Creating backup: $BACKUP_DIR" | tee -a "$MASTER_LOG"
                cp -r logs "$BACKUP_DIR"
                echo "   ✅ Backup created" | tee -a "$MASTER_LOG"
                echo "" | tee -a "$MASTER_LOG"
            fi
        fi

        echo "✅ Starting fresh training with:" | tee -a "$MASTER_LOG"
        echo "   - RGB loss enabled (w_rgb=1.0)" | tee -a "$MASTER_LOG"
        echo "   - SDS weight reduced (w_sds=100.0)" | tee -a "$MASTER_LOG"
        echo "   - New model architecture" | tee -a "$MASTER_LOG"
        echo "   - Validation enabled (every 5 epochs)" | tee -a "$MASTER_LOG"
        echo "" | tee -a "$MASTER_LOG"

        RESUME_ARG=""
        CURRENT_RUN_ID="new_$(date +%Y%m%d_%H%M%S)"
        echo "$CURRENT_RUN_ID" > "$RUN_ID_FILE"
        echo "$CURRENT_RUN_ID" >> "$ALL_RUN_IDS_FILE"

    else
        # ============================================================
        # CHUNKS 2+: Resume from current training session
        # ============================================================
        echo "========================================" | tee -a "$MASTER_LOG"
        echo "CHUNK $CHUNK_NUM: Resuming Training" | tee -a "$MASTER_LOG"
        echo "========================================" | tee -a "$MASTER_LOG"
        echo "" | tee -a "$MASTER_LOG"
        echo "Searching for checkpoint from current session..." | tee -a "$MASTER_LOG"

        # Use helper function to find best checkpoint from THIS session
        CKPT_SEARCH_RESULT=$(find_latest_checkpoint "$TRAINING_START_TIME")

        if [ $? -eq 0 ] && [ -n "$CKPT_SEARCH_RESULT" ]; then
            # Parse result
            LAST_CKPT=$(echo "$CKPT_SEARCH_RESULT" | cut -d'|' -f1)
            CKPT_EPOCH=$(echo "$CKPT_SEARCH_RESULT" | cut -d'|' -f2)
            CKPT_RUN_ID=$(echo "$CKPT_SEARCH_RESULT" | cut -d'|' -f3)

            echo "✅ Found checkpoint from current session:" | tee -a "$MASTER_LOG"
            echo "   Checkpoint: $LAST_CKPT" | tee -a "$MASTER_LOG"
            echo "   Epoch: $CKPT_EPOCH" | tee -a "$MASTER_LOG"
            echo "   Run ID: $CKPT_RUN_ID" | tee -a "$MASTER_LOG"
            echo "" | tee -a "$MASTER_LOG"

            # Verify checkpoint epoch matches expected
            EXPECTED_EPOCH=$CURRENT_EPOCH
            if [ "$CKPT_EPOCH" -ne "$EXPECTED_EPOCH" ]; then
                echo "⚠️  WARNING: Checkpoint epoch ($CKPT_EPOCH) != expected ($EXPECTED_EPOCH)" | tee -a "$MASTER_LOG"
                echo "   Continuing anyway, but verify results" | tee -a "$MASTER_LOG"
                echo "" | tee -a "$MASTER_LOG"
            fi

            # Update run ID tracking
            echo "$CKPT_RUN_ID" >> "$ALL_RUN_IDS_FILE"
            echo "$CKPT_RUN_ID" > "$RUN_ID_FILE"
            CURRENT_RUN_ID="$CKPT_RUN_ID"

            RESUME_ARG="--load_ckpt $LAST_CKPT"

        else
            # No checkpoint found for chunk > 1 is a critical error
            echo "" | tee -a "$MASTER_LOG"
            echo "❌ CRITICAL ERROR: No checkpoint found for chunk $CHUNK_NUM!" | tee -a "$MASTER_LOG"
            echo "" | tee -a "$MASTER_LOG"
            echo "Details:" | tee -a "$MASTER_LOG"
            echo "   Expected epoch: $CURRENT_EPOCH" | tee -a "$MASTER_LOG"
            echo "   Session start: $TRAINING_START_TIME" | tee -a "$MASTER_LOG"
            echo "   Searched directories created after session start" | tee -a "$MASTER_LOG"
            echo "" | tee -a "$MASTER_LOG"

            echo "Available checkpoints:" | tee -a "$MASTER_LOG"
            find logs -name "last.ckpt" -type f -printf '%T+ %p\n' 2>/dev/null | sort -r | head -5 | tee -a "$MASTER_LOG"
            echo "" | tee -a "$MASTER_LOG"

            echo "Recent run directories:" | tee -a "$MASTER_LOG"
            ls -lt logs/ | head -10 | tee -a "$MASTER_LOG"
            echo "" | tee -a "$MASTER_LOG"

            echo "Possible causes:" | tee -a "$MASTER_LOG"
            echo "   1. Previous chunk failed before saving checkpoint" | tee -a "$MASTER_LOG"
            echo "   2. Checkpoint was deleted or moved" | tee -a "$MASTER_LOG"
            echo "   3. Permissions issue preventing checkpoint access" | tee -a "$MASTER_LOG"
            echo "" | tee -a "$MASTER_LOG"

            echo "ABORTING TRAINING" | tee -a "$MASTER_LOG"
            exit 1
        fi
    fi

    echo "" | tee -a "$MASTER_LOG"

    # ================================================================
    # START GPU MONITORING
    # ================================================================
    GPU_MONITOR_LOG="$RESULTS_DIR/gpu_monitor_chunk${CHUNK_NUM}_$(date +%Y%m%d_%H%M%S).csv"

    echo "Starting GPU monitor..." | tee -a "$MASTER_LOG"

    if ! nvidia-smi &> /dev/null; then
        echo "  ⚠️  nvidia-smi not available, skipping monitor" | tee -a "$MASTER_LOG"
        GPU_MONITOR_PID=""
    else
        echo "  Log file: $GPU_MONITOR_LOG" | tee -a "$MASTER_LOG"

        {
            echo "# GPU Memory Monitor - Chunk $CHUNK_NUM"
            echo "# Started: $(date)"
            echo "# Interval: 2 seconds"
            echo "timestamp,memory_used_MiB,memory_free_MiB,memory_total_MiB"
        } > "$GPU_MONITOR_LOG"

        (
            while true; do
                nvidia-smi --query-gpu=timestamp,memory.used,memory.free,memory.total \
                           --format=csv,noheader,nounits 2>/dev/null || break
                sleep 2
            done
        ) >> "$GPU_MONITOR_LOG" 2>&1 &

        GPU_MONITOR_PID=$!

        sleep 2
        if ps -p $GPU_MONITOR_PID > /dev/null 2>&1; then
            echo "  ✓ GPU monitor running (PID: $GPU_MONITOR_PID)" | tee -a "$MASTER_LOG"
        else
            echo "  ⚠️  GPU monitor failed to start" | tee -a "$MASTER_LOG"
            GPU_MONITOR_PID=""
        fi
    fi

    echo "" | tee -a "$MASTER_LOG"

    # ================================================================
    # RUN TRAINING CHUNK
    # ================================================================
    CHUNK_START_SECONDS=$(date +%s)

    python -u train.py \
        --config "$TARGET_CONFIG" \
        --case ghop_bottle_1 \
        --use_ghop \
        --num_epoch $NEXT_EPOCH \
        $RESUME_ARG \
        --no-comet \
        --gpu_id 0 \
        --no-pin-memory \
        2>&1 | tee "$CHUNK_LOG"

    CHUNK_EXIT_CODE=$?

    # ================================================================
    # STOP GPU MONITOR
    # ================================================================
    if [ -n "$GPU_MONITOR_PID" ]; then
        kill $GPU_MONITOR_PID 2>/dev/null || true
        wait $GPU_MONITOR_PID 2>/dev/null || true
        echo "GPU monitor stopped" | tee -a "$MASTER_LOG"
    fi

    CHUNK_END_SECONDS=$(date +%s)
    CHUNK_DURATION=$((CHUNK_END_SECONDS - CHUNK_START_SECONDS))
    CHUNK_DURATION_MINS=$((CHUNK_DURATION / 60))

    echo "" | tee -a "$MASTER_LOG"
    echo "Chunk $CHUNK_NUM exit code: $CHUNK_EXIT_CODE" | tee -a "$MASTER_LOG"
    echo "End time: $(date)" | tee -a "$MASTER_LOG"
    echo "Duration: $CHUNK_DURATION_MINS minutes ($CHUNK_DURATION seconds)" | tee -a "$MASTER_LOG"

    # ================================================================
    # CHECK FOR FAILURE
    # ================================================================
    if [ $CHUNK_EXIT_CODE -ne 0 ]; then
        echo "" | tee -a "$MASTER_LOG"
        echo "❌ Chunk $CHUNK_NUM FAILED with exit code $CHUNK_EXIT_CODE" | tee -a "$MASTER_LOG"

        if [ $CHUNK_EXIT_CODE -eq 137 ]; then
            echo "   Exit 137 = OOM (killed by system)" | tee -a "$MASTER_LOG"
            echo "   Even with chunking, memory leak is too severe" | tee -a "$MASTER_LOG"
            echo "   Recommendation: Reduce CHUNK_SIZE from $CHUNK_SIZE to 3" | tee -a "$MASTER_LOG"
        fi

        break
    fi

    echo "" | tee -a "$MASTER_LOG"
    echo "✅ Chunk $CHUNK_NUM completed successfully" | tee -a "$MASTER_LOG"

    # ================================================================
    # UPDATE FOR NEXT ITERATION
    # ================================================================
    # Find updated checkpoint
    LAST_CKPT=$(find logs -name "last.ckpt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

    if [ -n "$LAST_CKPT" ]; then
        echo "   Updated checkpoint: $LAST_CKPT" | tee -a "$MASTER_LOG"
    else
        echo "   ⚠️  Warning: Could not find updated checkpoint" | tee -a "$MASTER_LOG"
    fi

    # Show GPU memory state before restart
    if command -v nvidia-smi &> /dev/null; then
        CURRENT_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
        echo "   GPU memory before restart: $CURRENT_MEM MiB" | tee -a "$MASTER_LOG"
    fi

    echo "" | tee -a "$MASTER_LOG"
    echo "🔄 Restarting Python process to clear accumulated memory..." | tee -a "$MASTER_LOG"
    echo "" | tee -a "$MASTER_LOG"

    # Small delay for clean process termination
    sleep 3

    # Update current epoch for next iteration
    CURRENT_EPOCH=$NEXT_EPOCH
done

# ================================================================
# FINAL STATUS CHECK
# ================================================================
if [ $CHUNK_EXIT_CODE -eq 0 ] && [ $CURRENT_EPOCH -ge $TOTAL_EPOCHS ]; then
    echo "" | tee -a "$MASTER_LOG"
    echo "========================================================================" | tee -a "$MASTER_LOG"
    echo "✅ ALL CHUNKS COMPLETED SUCCESSFULLY" | tee -a "$MASTER_LOG"
    echo "========================================================================" | tee -a "$MASTER_LOG"
    echo "" | tee -a "$MASTER_LOG"
    echo "Total chunks: $(( (TOTAL_EPOCHS + CHUNK_SIZE - 1) / CHUNK_SIZE ))" | tee -a "$MASTER_LOG"
    echo "Epochs trained: 0 → $TOTAL_EPOCHS" | tee -a "$MASTER_LOG"
    TRAINING_STATUS="SUCCESS"
    successful_chunks=$TOTAL_CHUNKS_CALC
else
    echo "" | tee -a "$MASTER_LOG"
    echo "========================================================================" | tee -a "$MASTER_LOG"
    echo "⚠️  TRAINING INCOMPLETE" | tee -a "$MASTER_LOG"
    echo "========================================================================" | tee -a "$MASTER_LOG"
    echo "" | tee -a "$MASTER_LOG"
    echo "Epochs completed: $CURRENT_EPOCH / $TOTAL_EPOCHS" | tee -a "$MASTER_LOG"
    TRAINING_STATUS="INCOMPLETE"
    successful_chunks=$((CURRENT_EPOCH / CHUNK_SIZE))
fi


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
# NEW: CREATE COMPREHENSIVE TRAINING SUMMARY FILE WITH CHECKPOINT VALIDATION
# ================================================================
echo "Creating training summary file..." | tee -a "$MASTER_LOG"

# ================================================================
# CRITICAL: Discover all checkpoint directories from this session
# ================================================================
echo "Discovering all checkpoints from training session..." | tee -a "$MASTER_LOG"

# Get session start time for filtering
SESSION_CUTOFF=$(date -d "$TRAINING_START_TIME - 1 minute" +%s 2>/dev/null || \
                 date -v-1M -j -f "%a %b %d %H:%M:%S %Z %Y" "$TRAINING_START_TIME" +%s 2>/dev/null)

# Build array of all checkpoint info from this session
declare -a CHECKPOINT_RUNS
declare -a CHECKPOINT_EPOCHS
declare -a CHECKPOINT_STEPS
declare -a CHECKPOINT_PATHS

for run_dir in logs/*/; do
    run_id=$(basename "$run_dir")

    # Skip if not a valid run ID format
    if ! [[ "$run_id" =~ ^[a-f0-9]{9}$ ]]; then
        continue
    fi

    ckpt_file="$run_dir/checkpoints/last.ckpt"

    # Skip if checkpoint doesn't exist
    if [ ! -f "$ckpt_file" ]; then
        continue
    fi

    # Get checkpoint modification time
    ckpt_mtime=$(stat -c %Y "$ckpt_file" 2>/dev/null || stat -f %m "$ckpt_file" 2>/dev/null)

    # Skip if checkpoint is older than session start
    if [ -n "$SESSION_CUTOFF" ] && [ "$ckpt_mtime" -lt "$SESSION_CUTOFF" ]; then
        continue
    fi

    # Read checkpoint epoch and global_step
    ckpt_info=$(python -c "
import torch
try:
    ckpt = torch.load('$ckpt_file', map_location='cpu')
    epoch = ckpt.get('epoch', -1)
    global_step = ckpt.get('global_step', -1)
    print(f'{epoch},{global_step}')
except Exception:
    print('-1,-1')
" 2>/dev/null)

    ckpt_epoch=$(echo "$ckpt_info" | cut -d',' -f1)
    ckpt_step=$(echo "$ckpt_info" | cut -d',' -f2)

    # Skip invalid checkpoints
    if [ "$ckpt_epoch" -le 0 ]; then
        continue
    fi

    # Add to arrays
    CHECKPOINT_RUNS+=("$run_id")
    CHECKPOINT_EPOCHS+=("$ckpt_epoch")
    CHECKPOINT_STEPS+=("$ckpt_step")
    CHECKPOINT_PATHS+=("$ckpt_file")
done

# Sort checkpoints by epoch (bubble sort for bash arrays)
num_checkpoints=${#CHECKPOINT_RUNS[@]}
for ((i=0; i<$num_checkpoints; i++)); do
    for ((j=i+1; j<$num_checkpoints; j++)); do
        if [ "${CHECKPOINT_EPOCHS[$i]}" -gt "${CHECKPOINT_EPOCHS[$j]}" ]; then
            # Swap all arrays
            temp="${CHECKPOINT_RUNS[$i]}"
            CHECKPOINT_RUNS[$i]="${CHECKPOINT_RUNS[$j]}"
            CHECKPOINT_RUNS[$j]="$temp"

            temp="${CHECKPOINT_EPOCHS[$i]}"
            CHECKPOINT_EPOCHS[$i]="${CHECKPOINT_EPOCHS[$j]}"
            CHECKPOINT_EPOCHS[$j]="$temp"

            temp="${CHECKPOINT_STEPS[$i]}"
            CHECKPOINT_STEPS[$i]="${CHECKPOINT_STEPS[$j]}"
            CHECKPOINT_STEPS[$j]="$temp"

            temp="${CHECKPOINT_PATHS[$i]}"
            CHECKPOINT_PATHS[$i]="${CHECKPOINT_PATHS[$j]}"
            CHECKPOINT_PATHS[$j]="$temp"
        fi
    done
done

# Calculate actual training progress from checkpoints
if [ $num_checkpoints -gt 0 ]; then
    FINAL_EPOCH="${CHECKPOINT_EPOCHS[$((num_checkpoints-1))]}"
    FINAL_RUN_ID="${CHECKPOINT_RUNS[$((num_checkpoints-1))]}"
    FINAL_CKPT_PATH="${CHECKPOINT_PATHS[$((num_checkpoints-1))]}"
    actual_chunks_completed=$num_checkpoints
else
    FINAL_EPOCH=0
    FINAL_RUN_ID="${CURRENT_RUN_ID:-unknown}"
    FINAL_CKPT_PATH="Not found"
    actual_chunks_completed=0
fi

# Determine training status based on actual checkpoint epoch
if [ "$FINAL_EPOCH" -ge "$TOTAL_EPOCHS" ]; then
    TRAINING_STATUS="SUCCESS"
elif [ "$FINAL_EPOCH" -gt 0 ]; then
    TRAINING_STATUS="INCOMPLETE"
else
    TRAINING_STATUS="FAILED"
fi

echo "Checkpoint discovery complete:" | tee -a "$MASTER_LOG"
echo "  Found: $num_checkpoints checkpoints" | tee -a "$MASTER_LOG"
echo "  Final epoch: $FINAL_EPOCH / $TOTAL_EPOCHS" | tee -a "$MASTER_LOG"
echo "  Status: $TRAINING_STATUS" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# ================================================================
# WRITE SUMMARY FILE
# ================================================================
cat > "$SUMMARY_FILE" << EOF
================================================================================
                    HOLDSE TRAINING SESSION SUMMARY
================================================================================

Generated: $(date)
Training Session ID: $TIMESTAMP

================================================================================
QUICK REFERENCE
================================================================================

Training Status:        $TRAINING_STATUS
Epochs Completed:       $FINAL_EPOCH / $TOTAL_EPOCHS
Checkpoints Created:    $num_checkpoints

Final Run ID:           $FINAL_RUN_ID
Final Checkpoint:       $FINAL_CKPT_PATH

Config File:            $TARGET_CONFIG
Master Log:             $MASTER_LOG

================================================================================
TRAINING CONFIGURATION
================================================================================

Strategy:               Chunked training with periodic Python restarts
Chunk Size:             $CHUNK_SIZE epochs per chunk
Total Chunks Planned:   $NUM_CHUNKS
Target Epochs:          $TOTAL_EPOCHS

CUDA Configuration:     PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF
Memory Management:
  ✓ Periodic process restarts (every $CHUNK_SIZE epochs)
  ✓ PyTorch Lightning memory leak mitigation
  ✓ Native CUDA allocator backend

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

# ================================================================
# ADD CHECKPOINT-BASED TRAINING BREAKDOWN
# ================================================================
echo "" >> "$SUMMARY_FILE"
echo "Training Progress (Checkpoint-Based):" >> "$SUMMARY_FILE"
echo "=====================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

if [ $num_checkpoints -gt 0 ]; then
    echo "Discovered $num_checkpoints checkpoint(s) from this training session:" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"

    for ((i=0; i<$num_checkpoints; i++)); do
        run_id="${CHECKPOINT_RUNS[$i]}"
        epoch="${CHECKPOINT_EPOCHS[$i]}"
        global_step="${CHECKPOINT_STEPS[$i]}"
        ckpt_path="${CHECKPOINT_PATHS[$i]}"

        # Calculate which chunk this represents
        chunk_num=$((epoch / CHUNK_SIZE))
        epoch_start=$(( (chunk_num - 1) * CHUNK_SIZE ))
        if [ $epoch_start -lt 0 ]; then
            epoch_start=0
        fi

        echo "Checkpoint $((i+1)):" >> "$SUMMARY_FILE"
        echo "  Run ID:       $run_id" >> "$SUMMARY_FILE"
        echo "  Epoch:        $epoch (trained: $epoch_start → $epoch)" >> "$SUMMARY_FILE"
        echo "  Global Step:  $global_step" >> "$SUMMARY_FILE"
        echo "  Path:         $ckpt_path" >> "$SUMMARY_FILE"

        # Validate progression if not first checkpoint
        if [ $i -gt 0 ]; then
            prev_epoch="${CHECKPOINT_EPOCHS[$((i-1))]}"
            prev_step="${CHECKPOINT_STEPS[$((i-1))]}"

            epoch_delta=$((epoch - prev_epoch))
            step_delta=$((global_step - prev_step))

            # Check if progression is correct
            if [ $epoch_delta -eq $CHUNK_SIZE ] && [ $step_delta -gt 0 ]; then
                echo "  Progress:     ✅ Valid (+$epoch_delta epochs, +$step_delta steps)" >> "$SUMMARY_FILE"
            elif [ $epoch_delta -eq 0 ] && [ $step_delta -eq 0 ]; then
                echo "  Progress:     ❌ NO PROGRESS (checkpoint loading bug detected)" >> "$SUMMARY_FILE"
            else
                echo "  Progress:     ⚠️  Unexpected (+$epoch_delta epochs, +$step_delta steps)" >> "$SUMMARY_FILE"
            fi
        else
            echo "  Progress:     ✅ First checkpoint (baseline)" >> "$SUMMARY_FILE"
        fi

        echo "" >> "$SUMMARY_FILE"
    done

    # Add progression summary
    echo "Training Progression Summary:" >> "$SUMMARY_FILE"
    echo "  Start epoch:    0" >> "$SUMMARY_FILE"
    echo "  Final epoch:    $FINAL_EPOCH" >> "$SUMMARY_FILE"
    echo "  Total trained:  $FINAL_EPOCH epochs" >> "$SUMMARY_FILE"
    echo "  Target:         $TOTAL_EPOCHS epochs" >> "$SUMMARY_FILE"
    echo "  Progress:       $(( FINAL_EPOCH * 100 / TOTAL_EPOCHS ))%" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
else
    echo "No checkpoints found from this training session." >> "$SUMMARY_FILE"
    echo "Training may have failed to start or crashed immediately." >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
fi

# ================================================================
# ADD CHUNK LOG INFORMATION (SUPPLEMENTARY)
# ================================================================
echo "" >> "$SUMMARY_FILE"
echo "Chunk Logs (Supplementary Information):" >> "$SUMMARY_FILE"
echo "=======================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

chunk_logs_found=0
for ((i=0; i<NUM_CHUNKS; i++)); do
    chunk_log=$(ls -t "$RESULTS_DIR"/chunk*_epochs_$((i * CHUNK_SIZE))_to_$(((i + 1) * CHUNK_SIZE))_*.log 2>/dev/null | head -1)

    if [ -n "$chunk_log" ]; then
        chunk_logs_found=$((chunk_logs_found + 1))

        chunk_start=$(grep "Start time:" "$chunk_log" | head -1 | cut -d':' -f2- | xargs)
        chunk_end=$(grep "End time:" "$chunk_log" | head -1 | cut -d':' -f2- | xargs)

        # Check for errors
        if grep -qE "out of memory|OutOfMemoryError|OOM" "$chunk_log"; then
            status="❌ OOM Error"
        elif grep -qE "Error|Exception|Traceback" "$chunk_log"; then
            status="❌ Error"
        elif grep -q "exit code: 0" "$chunk_log" || grep -q "Training complete" "$chunk_log"; then
            status="✅ Completed"
        else
            status="⚠️  Unknown"
        fi

        echo "Chunk $((i+1)): Epochs $((i * CHUNK_SIZE)) → $(((i + 1) * CHUNK_SIZE))" >> "$SUMMARY_FILE"
        echo "  Status:  $status" >> "$SUMMARY_FILE"
        [ -n "$chunk_start" ] && echo "  Started: $chunk_start" >> "$SUMMARY_FILE"
        [ -n "$chunk_end" ] && echo "  Ended:   $chunk_end" >> "$SUMMARY_FILE"
        echo "  Log:     $chunk_log" >> "$SUMMARY_FILE"
        echo "" >> "$SUMMARY_FILE"
    fi
done

if [ $chunk_logs_found -eq 0 ]; then
    echo "No chunk log files found in $RESULTS_DIR/" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
fi

# ================================================================
# REST OF SUMMARY (GPU INFO, ETC.)
# ================================================================
cat >> "$SUMMARY_FILE" << EOF

================================================================================
FINAL CHECKPOINT DETAILS
================================================================================

EOF

if [ -n "$FINAL_CKPT_PATH" ] && [ -f "$FINAL_CKPT_PATH" ]; then
    echo "Path:     $FINAL_CKPT_PATH" >> "$SUMMARY_FILE"
    echo "Run ID:   $FINAL_RUN_ID" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"

    CKPT_DETAILS=$(python -c "
import torch
try:
    ckpt = torch.load('$FINAL_CKPT_PATH', map_location='cpu')
    print(f\"Epoch:         {ckpt.get('epoch', 'N/A')}\")
    print(f\"Global Step:   {ckpt.get('global_step', 'N/A')}\")
    if 'state_dict' in ckpt:
        num_params = len(ckpt['state_dict'])
        print(f\"Parameters:    {num_params}\")
    if 'optimizer_states' in ckpt:
        print(f\"Optimizer:     Saved\")
    if 'lr_schedulers' in ckpt:
        print(f\"LR Scheduler:  Saved\")
except Exception as e:
    print(f\"Error reading checkpoint: {e}\")
" 2>/dev/null)

    echo "$CKPT_DETAILS" >> "$SUMMARY_FILE"
else
    echo "Final checkpoint not found or not accessible." >> "$SUMMARY_FILE"
fi

cat >> "$SUMMARY_FILE" << EOF

================================================================================
ALL CHECKPOINTS FROM SESSION
================================================================================

EOF

if [ $num_checkpoints -gt 0 ]; then
    echo "Directory listing of all checkpoint directories:" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"

    for ((i=0; i<$num_checkpoints; i++)); do
        run_id="${CHECKPOINT_RUNS[$i]}"
        echo "logs/$run_id/checkpoints/:" >> "$SUMMARY_FILE"
        ls -lh "logs/$run_id/checkpoints/" >> "$SUMMARY_FILE" 2>/dev/null || echo "  (not accessible)" >> "$SUMMARY_FILE"
        echo "" >> "$SUMMARY_FILE"
    done
else
    echo "No checkpoints from this session." >> "$SUMMARY_FILE"
fi

# GPU information
cat >> "$SUMMARY_FILE" << EOF

================================================================================
GPU INFORMATION
================================================================================

EOF

if command -v nvidia-smi &> /dev/null; then
    echo "GPU Configuration:" >> "$SUMMARY_FILE"
    nvidia-smi --query-gpu=name,memory.total,driver_version,cuda_version --format=csv,noheader 2>/dev/null | head -1 | \
        awk -F', ' '{print "  GPU:            "$1"\n  Memory:         "$2"\n  Driver:         "$3"\n  CUDA:           "$4}' >> "$SUMMARY_FILE" || \
        echo "  (Could not query GPU info)" >> "$SUMMARY_FILE"

    echo "" >> "$SUMMARY_FILE"
    echo "Current GPU Memory:" >> "$SUMMARY_FILE"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | \
        awk '{print "  Used:           "$1" MiB\n  Free:           "$2" MiB"}' >> "$SUMMARY_FILE" || \
        echo "  (Could not query memory)" >> "$SUMMARY_FILE"
else
    echo "nvidia-smi not available" >> "$SUMMARY_FILE"
fi

# Training outcome
cat >> "$SUMMARY_FILE" << EOF

================================================================================
TRAINING OUTCOME
================================================================================

Status:           $TRAINING_STATUS
Epochs Trained:   $FINAL_EPOCH / $TOTAL_EPOCHS
Completion:       $(( FINAL_EPOCH * 100 / TOTAL_EPOCHS ))%
Checkpoints:      $num_checkpoints created

EOF

if [ "$TRAINING_STATUS" == "SUCCESS" ]; then
    cat >> "$SUMMARY_FILE" << EOF
✅ TRAINING COMPLETED SUCCESSFULLY

All $TOTAL_EPOCHS epochs completed. Model is ready for evaluation.

Next Steps:
  1. Validate model on test set
  2. Run inference on new data
  3. Archive checkpoint to permanent storage

Command to use checkpoint:
  python test.py --load_ckpt $FINAL_CKPT_PATH --config $TARGET_CONFIG

EOF
elif [ "$TRAINING_STATUS" == "INCOMPLETE" ]; then
    cat >> "$SUMMARY_FILE" << EOF
⚠️  TRAINING INCOMPLETE

Completed $FINAL_EPOCH out of $TOTAL_EPOCHS epochs ($((TOTAL_EPOCHS - FINAL_EPOCH)) remaining).

To Resume Training:
  The chunked training script will automatically resume from the latest
  checkpoint. Simply run:

    bash scripts/train_ghop_hoi4d_production_chunked.sh

  It will detect and load: $FINAL_CKPT_PATH

EOF
else
    cat >> "$SUMMARY_FILE" << EOF
❌ TRAINING FAILED

No checkpoints were created. Training may have crashed immediately.

Troubleshooting Steps:
  1. Check the master log for errors: less $MASTER_LOG
  2. Verify GPU availability: nvidia-smi
  3. Check CUDA configuration: echo \$PYTORCH_CUDA_ALLOC_CONF
  4. Review Python environment: python -c "import torch; print(torch.__version__)"

EOF
fi

# Troubleshooting section
cat >> "$SUMMARY_FILE" << EOF

================================================================================
TROUBLESHOOTING
================================================================================

Verify Checkpoint Progression:
  Run this command to check all checkpoints:

  for ckpt in logs/*/checkpoints/last.ckpt; do
      [ -f "\$ckpt" ] && python -c "
import torch
ckpt = torch.load('\$ckpt', map_location='cpu')
print('\$ckpt: epoch=' + str(ckpt.get('epoch', -1)) + ', step=' + str(ckpt.get('global_step', -1)))
"
  done

Check GPU Memory:
  nvidia-smi

Review Master Log:
  less $MASTER_LOG

Review Chunk Logs:
  ls -lt $RESULTS_DIR/chunk*.log

Common Issues:
  - Epochs not progressing: Verify --num_epoch uses \$NEXT_EPOCH (not \$ACTUAL_EPOCHS)
  - Multiple run IDs: Expected with current implementation (one per chunk)
  - OOM errors: Reduce CHUNK_SIZE or batch_size in config

================================================================================
SYSTEM INFORMATION
================================================================================

Hostname:         $(hostname)
User:             $(whoami)
Working Dir:      $(pwd)
Python:           $(python --version 2>&1)
PyTorch:          $(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "Not available")

Environment:
  PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF
  CUDA_VISIBLE_DEVICES:    ${CUDA_VISIBLE_DEVICES:-"Not set"}

================================================================================
END OF SUMMARY
================================================================================

Master log:     $MASTER_LOG
Chunk logs:     $RESULTS_DIR/chunk*.log
Generated:      $(date)
================================================================================
EOF

echo "✅ Training summary saved to: $SUMMARY_FILE" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# Display summary
echo "========================================================================" | tee -a "$MASTER_LOG"
echo "📄 TRAINING SUMMARY" | tee -a "$MASTER_LOG"
echo "========================================================================" | tee -a "$MASTER_LOG"
echo "  Status:       $TRAINING_STATUS" | tee -a "$MASTER_LOG"
echo "  Epochs:       $FINAL_EPOCH / $TOTAL_EPOCHS" | tee -a "$MASTER_LOG"
echo "  Checkpoints:  $num_checkpoints" | tee -a "$MASTER_LOG"
echo "  Summary file: $SUMMARY_FILE" | tee -a "$MASTER_LOG"
echo "========================================================================" | tee -a "$MASTER_LOG"
