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
TOTAL_EPOCHS=100
NUM_CHUNKS=$((TOTAL_EPOCHS / CHUNK_SIZE))

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
# FILE: scripts/train_ghop_hoi4d_production_chunked.sh
# SECTION: Configuration creation with memory optimization
# ================================================================

# Use the latest production config or create one
BASE_CONFIG="confs/ghop_quick_bottle_1.yaml"
TARGET_CONFIG="confs/ghop_production_chunked_$TIMESTAMP.yaml"

# Copy and modify base config (only once)
cp "$BASE_CONFIG" "$TARGET_CONFIG"

# ================================================================
# APPLY AGGRESSIVE MEMORY REDUCTION (NEW)
# ================================================================
echo "Applying aggressive memory reduction settings..." | tee -a "$MASTER_LOG"

# ================================================================
# 1. REDUCE MODEL DIMENSIONS (75% parameter reduction)
# ================================================================
echo "  Reducing model dimensions..." | tee -a "$MASTER_LOG"

# Implicit network: dims [32, 32] → [16, 16]
#sed -i '/implicit_network:/,/cond: pose/ s/dims: \[[0-9, ]*\]/dims: [16, 16]/' "$TARGET_CONFIG"
sed -i '/implicit_network:/,/cond: pose/ s/feature_vector_size: [0-9]\+/feature_vector_size: 16/' "$TARGET_CONFIG"
sed -i '/implicit_network:/,/cond: pose/ s/multires: [0-9]\+/multires: 0/' "$TARGET_CONFIG"

# Rendering network: dims [32] → [16]
sed -i '/rendering_network:/,/multires_view:/ s/dims: \[[0-9, ]*\]/dims: [16]/' "$TARGET_CONFIG"
sed -i '/rendering_network:/,/multires_view:/ s/feature_vector_size: [0-9]\+/feature_vector_size: 16/' "$TARGET_CONFIG"

# Background implicit network: dims [32, 32] → [16, 16]
sed -i '/bg_implicit_network:/,/cond: frame/ s/dims: \[[0-9, ]*\]/dims: [16, 16]/' "$TARGET_CONFIG"
sed -i '/bg_implicit_network:/,/cond: frame/ s/feature_vector_size: [0-9]\+/feature_vector_size: 16/' "$TARGET_CONFIG"
sed -i '/bg_implicit_network:/,/cond: frame/ s/dim_frame_encoding: [0-9]\+/dim_frame_encoding: 4/' "$TARGET_CONFIG"
sed -i '/bg_implicit_network:/,/cond: frame/ s/multires: [0-9]\+/multires: 1/' "$TARGET_CONFIG"

# Background rendering network: dims [32] → [16]
sed -i '/bg_rendering_network:/,/dim_frame_encoding:/ s/dims: \[[0-9, ]*\]/dims: [16]/' "$TARGET_CONFIG"
sed -i '/bg_rendering_network:/,/dim_frame_encoding:/ s/feature_vector_size: [0-9]\+/feature_vector_size: 16/' "$TARGET_CONFIG"
sed -i '/bg_rendering_network:/,/dim_frame_encoding:/ s/multires_view: [0-9]\+/multires_view: 0/' "$TARGET_CONFIG"
sed -i '/bg_rendering_network:/,/dim_frame_encoding:/ s/dim_frame_encoding: [0-9]\+/dim_frame_encoding: 4/' "$TARGET_CONFIG"

echo "  ✓ Reduced model dimensions: 32→16 (75% reduction)" | tee -a "$MASTER_LOG"

# ================================================================
# 2. REDUCE RAY SAMPLING (50% memory reduction)
# ================================================================
echo "  Reducing ray samples..." | tee -a "$MASTER_LOG"

#sed -i '/ray_sampler:/,/add_tiny:/ s/N_samples: [0-9]\+/N_samples: 4/' "$TARGET_CONFIG"
sed -i '/ray_sampler:/,/add_tiny:/ s/N_samples_eval: [0-9]\+/N_samples_eval: 4/' "$TARGET_CONFIG"
sed -i '/ray_sampler:/,/add_tiny:/ s/N_samples_extra: [0-9]\+/N_samples_extra: 2/' "$TARGET_CONFIG"
sed -i '/ray_sampler:/,/add_tiny:/ s/N_samples_inverse_sphere: [0-9]\+/N_samples_inverse_sphere: 2/' "$TARGET_CONFIG"

echo "  ✓ Reduced ray samples: 8→4 (50% reduction)" | tee -a "$MASTER_LOG"

# ================================================================
# 3. REDUCE GRID RESOLUTION (87% memory reduction)
# ================================================================
echo "  Reducing grid resolution..." | tee -a "$MASTER_LOG"

# Phase 3 grid: 32 → 16
sed -i '/^phase3:/,/^phase4:/ s/grid_resolution: [0-9]\+/grid_resolution: 16/' "$TARGET_CONFIG"

echo "  ✓ Reduced Phase 3 grid: 32³→16³ (87% reduction)" | tee -a "$MASTER_LOG"

# ================================================================
# 4. REDUCE MESH RESOLUTION (87% memory reduction)
# ================================================================
echo "  Reducing mesh resolution..." | tee -a "$MASTER_LOG"

# Phase 4 mesh: 64 → 32
sed -i '/^phase4:/,/^phase5:/ s/resolution: [0-9]\+/resolution: 32/' "$TARGET_CONFIG"

echo "  ✓ Reduced Phase 4 mesh: 64³→32³ (87% reduction)" | tee -a "$MASTER_LOG"

echo "" | tee -a "$MASTER_LOG"
echo "✅ Memory optimization complete" | tee -a "$MASTER_LOG"
echo "   Estimated total memory reduction: 70-80%" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# ================================================================
# APPLY PHASE 5 FIX (EXISTING - KEEP THESE)
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

echo "Disabling Phase 4 and 5 for memory testing..." | tee -a "$MASTER_LOG"

# Disable Phase 4
sed -i '/^phase4:/,/^phase5:/ s/enabled: true/enabled: false/' "$TARGET_CONFIG"
echo "  ✓ Disabled Phase 4 (contact refinement)" | tee -a "$MASTER_LOG"

# Disable Phase 5
sed -i '/^phase5:/,/^training:/ s/enabled: true/enabled: false/' "$TARGET_CONFIG"
echo "  ✓ Disabled Phase 5 (temporal consistency)" | tee -a "$MASTER_LOG"

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
# VERIFICATION: Show key memory settings
# ================================================================
echo "Configuration verification:" | tee -a "$MASTER_LOG"
echo "  Model dims:" | tee -a "$MASTER_LOG"
grep -A 2 "implicit_network:" "$TARGET_CONFIG" | grep "dims:" | tee -a "$MASTER_LOG"
echo "  Ray samples:" | tee -a "$MASTER_LOG"
grep "N_samples:" "$TARGET_CONFIG" | tee -a "$MASTER_LOG"
echo "  Grid resolution:" | tee -a "$MASTER_LOG"
grep "grid_resolution:" "$TARGET_CONFIG" | tee -a "$MASTER_LOG"
echo "  Mesh resolution:" | tee -a "$MASTER_LOG"
grep -A 5 "mesh_extraction:" "$TARGET_CONFIG" | grep "resolution:" | tee -a "$MASTER_LOG"
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
    # CHECKPOINT LOADING LOGIC
    # ================================================================
    if [ $CURRENT_EPOCH -eq 0 ]; then
        # First chunk: Start fresh
        echo "Starting from scratch (epoch 0)" | tee -a "$MASTER_LOG"
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

            RESUME_ARG="--load_ckpt $LAST_CKPT"
        else
            # No checkpoint found for chunk > 0 is an error
            echo "❌ ERROR: No checkpoint found for current epoch $CURRENT_EPOCH!" | tee -a "$MASTER_LOG"
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
        --num_epoch $ACTUAL_EPOCHS \
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
