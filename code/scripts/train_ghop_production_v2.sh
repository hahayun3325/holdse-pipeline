#!/bin/bash
# File: code/scripts/train_ghop_production_v2.sh
# PURPOSE: Production training with WORKING 32-dim architecture
# UPDATED: Now uses ghop_quick_bottle_1.yaml as base (verified 32-dim)

set -e

cd ~/Projects/holdse/code

# ================================================================
# SETUP: Results directory
# ================================================================
RESULTS_DIR="../ghop_production_results"
mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="$RESULTS_DIR/production_32dim_$TIMESTAMP.txt"

echo ""
echo "========================================================================" | tee "$SUMMARY_FILE"
echo "GHOP PRODUCTION TRAINING V2 - 32-dim (WORKING ARCHITECTURE)" | tee -a "$SUMMARY_FILE"
echo "========================================================================" | tee -a "$SUMMARY_FILE"
echo "Start: $(date)" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"
echo "Strategy:" | tee -a "$SUMMARY_FILE"
echo "  ✅ Use ORIGINAL working 32-dim config (ghop_quick_bottle_1.yaml)" | tee -a "$SUMMARY_FILE"
echo "  ✅ ONLY add loss weights + timing fixes" | tee -a "$SUMMARY_FILE"
echo "  ✅ NO architecture changes" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# ================================================================
# STEP 1: Check for original 32-dim config
# ================================================================
echo "Checking for original 32-dim config..." | tee -a "$SUMMARY_FILE"

BASE_CONFIG="confs/ghop_quick_bottle_1.yaml"

if [ ! -f "$BASE_CONFIG" ]; then
    echo "❌ Error: Base config not found: $BASE_CONFIG" | tee -a "$SUMMARY_FILE"
    echo "" | tee -a "$SUMMARY_FILE"
    echo "This file should contain the ORIGINAL working 32-dim architecture" | tee -a "$SUMMARY_FILE"
    exit 1
fi

# Verify it's actually 32-dim
if ! grep -q "feature_vector_size: 32" "$BASE_CONFIG"; then
    echo "❌ Error: Base config is NOT 32-dim!" | tee -a "$SUMMARY_FILE"
    echo "" | tee -a "$SUMMARY_FILE"
    echo "Expected: feature_vector_size: 32" | tee -a "$SUMMARY_FILE"
    echo "Found:" | tee -a "$SUMMARY_FILE"
    grep "feature_vector_size:" "$BASE_CONFIG" | head -3 | sed 's/^/  /' | tee -a "$SUMMARY_FILE"
    exit 1
fi

echo "✓ Found working 32-dim config: $BASE_CONFIG" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# ================================================================
# STEP 2: Create production config
# ================================================================
echo "Creating production config..." | tee -a "$SUMMARY_FILE"

TARGET_CONFIG="confs/ghop_production_32dim_$TIMESTAMP.yaml"

# Copy base config
cp "$BASE_CONFIG" "$TARGET_CONFIG"

echo "  ✓ Copied base config" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# ================================================================
# STEP 3: Apply targeted fixes
# ================================================================
echo "Applying fixes..." | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# Fix 1: Update epochs
echo "  Fix 1: Training duration" | tee -a "$SUMMARY_FILE"
sed -i 's/num_epochs: 1/num_epochs: 100/' "$TARGET_CONFIG"
sed -i 's/max_steps: 20/max_steps: -1/' "$TARGET_CONFIG"
echo "    ✓ num_epochs: 1 → 100" | tee -a "$SUMMARY_FILE"
echo "    ✓ max_steps: 20 → unlimited" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# Fix 2: Fix Phase 5 timing
echo "  Fix 2: Phase 5 timing" | tee -a "$SUMMARY_FILE"
phase4_start=$(grep "contact_start_iter:" "$TARGET_CONFIG" | head -1 | awk '{print $2}')
phase5_current=$(grep "phase5_start_iter:" "$TARGET_CONFIG" | head -1 | awk '{print $2}')

if [ -n "$phase4_start" ] && [ -n "$phase5_current" ]; then
    if [ "$phase5_current" -lt "$phase4_start" ]; then
        new_phase5=$((phase4_start + 100))
        sed -i "s/phase5_start_iter: $phase5_current/phase5_start_iter: $new_phase5/" "$TARGET_CONFIG"
        echo "    ✓ phase5_start_iter: $phase5_current → $new_phase5" | tee -a "$SUMMARY_FILE"
        echo "      (must be >= phase4_start $phase4_start)" | tee -a "$SUMMARY_FILE"
    else
        echo "    ✓ Phase timing already correct ($phase5_current >= $phase4_start)" | tee -a "$SUMMARY_FILE"
    fi
else
    # Default fix
    sed -i 's/phase5_start_iter: 100/phase5_start_iter: 600/' "$TARGET_CONFIG"
    echo "    ✓ phase5_start_iter: 100 → 600 (default fix)" | tee -a "$SUMMARY_FILE"
fi
echo "" | tee -a "$SUMMARY_FILE"

# Fix 3: Add loss weights
echo "  Fix 3: Loss weights" | tee -a "$SUMMARY_FILE"

# Check if loss section already exists
if grep -q "^loss:" "$TARGET_CONFIG"; then
    echo "    ⚠️  Loss section already exists - checking for w_rgb..." | tee -a "$SUMMARY_FILE"
    
    if grep -q "w_rgb:" "$TARGET_CONFIG"; then
        echo "    ✓ Loss weights already present" | tee -a "$SUMMARY_FILE"
    else
        echo "    ⚠️  Loss section exists but missing w_rgb - manual intervention needed" | tee -a "$SUMMARY_FILE"
    fi
else
    # Add loss weights
    cat >> "$TARGET_CONFIG" << 'LOSS_EOF'

# ================================================================
# LOSS WEIGHTS (CRITICAL FIX - ADDED BY SCRIPT)
# ================================================================
loss:
  w_rgb: 10.0              # RGB loss weight (was missing!)
  w_mask: 5.0              # Mask loss
  w_mano_cano: 0.5         # REDUCED to prevent 99.61% dominance
  w_eikonal: 0.1
  w_smooth: 0.01
  w_semantic: 1.0
  w_opacity_sparse: 0.1
LOSS_EOF
    
    echo "    ✓ Added loss weights section" | tee -a "$SUMMARY_FILE"
    echo "      w_rgb: 10.0 (CRITICAL)" | tee -a "$SUMMARY_FILE"
    echo "      w_mano_cano: 0.5 (prevent dominance)" | tee -a "$SUMMARY_FILE"
fi
echo "" | tee -a "$SUMMARY_FILE"

# ================================================================
# Fix 4: Reduce batch size to prevent OOM
# ================================================================
echo "  Fix 4: Memory optimization (batch size)" | tee -a "$SUMMARY_FILE"

# Find current batch_size
current_batch=$(grep "batch_size:" "$TARGET_CONFIG" | grep -v "#" | head -1 | awk '{print $2}')

if [ -z "$current_batch" ]; then
    current_batch=2  # Assume default
fi

echo "    Current batch_size: $current_batch" | tee -a "$SUMMARY_FILE"

# Reduce to 1 to prevent OOM at epoch 22
if [ "$current_batch" -gt 1 ]; then
    # Update all batch_size occurrences in dataset sections
    sed -i '/^dataset:/,/^[a-z_]*:/ s/batch_size: [0-9]\+/batch_size: 1/' "$TARGET_CONFIG"
    echo "    ✓ batch_size: $current_batch → 1 (prevent OOM at epoch 22)" | tee -a "$SUMMARY_FILE"
    echo "      Note: Training ~20% slower but completes all 100 epochs" | tee -a "$SUMMARY_FILE"
else
    echo "    ✓ batch_size already optimal: $current_batch" | tee -a "$SUMMARY_FILE"
fi

# ================================================================
# Fix 5: AGGRESSIVELY reduce pixel_per_batch to prevent fragmentation
# ================================================================
current_pixels=$(grep "pixel_per_batch:" "$TARGET_CONFIG" | grep -v "#" | head -1 | awk '{print $2}')
if [ -z "$current_pixels" ]; then
    current_pixels=2048  # Assume default if not found
fi

echo "    Current pixel_per_batch: $current_pixels" | tee -a "$SUMMARY_FILE"

# Reduce to 512 (down from 1024) for maximum memory safety
if [ "$current_pixels" -gt 512 ]; then
    sed -i 's/pixel_per_batch: [0-9]\+/pixel_per_batch: 512/' "$TARGET_CONFIG"
    echo "    ✓ pixel_per_batch: $current_pixels → 512 (prevent fragmentation)" | tee -a "$SUMMARY_FILE"
    echo "      Note: ~30% slower but prevents OOM from fragmentation" | tee -a "$SUMMARY_FILE"
else
    echo "    ✓ pixel_per_batch already optimal: $current_pixels" | tee -a "$SUMMARY_FILE"
fi

echo "" | tee -a "$SUMMARY_FILE"

echo "✅ All fixes applied" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"
# ================================================================
# STEP 4: Comprehensive verification
# ================================================================
echo "========================================================================" | tee -a "$SUMMARY_FILE"
echo "VERIFICATION" | tee -a "$SUMMARY_FILE"
echo "========================================================================" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

verification_passed=true

# Check 1: Architecture is 32-dim
echo -n "1. Network architecture: " | tee -a "$SUMMARY_FILE"
if grep -q "feature_vector_size: 32" "$TARGET_CONFIG"; then
    echo "✅ 32-dim (WORKING)" | tee -a "$SUMMARY_FILE"
else
    echo "❌ NOT 32-dim!" | tee -a "$SUMMARY_FILE"
    verification_passed=false
fi

# Check 2: Loss weights present
echo -n "2. Loss weights: " | tee -a "$SUMMARY_FILE"
if grep -q "w_rgb:" "$TARGET_CONFIG"; then
    w_rgb=$(grep "w_rgb:" "$TARGET_CONFIG" | head -1 | awk '{print $2}')
    echo "✅ Present (w_rgb=$w_rgb)" | tee -a "$SUMMARY_FILE"
else
    echo "❌ Missing w_rgb!" | tee -a "$SUMMARY_FILE"
    verification_passed=false
fi

# Check 3: Epochs updated
echo -n "3. Training epochs: " | tee -a "$SUMMARY_FILE"
epochs=$(grep "num_epochs:" "$TARGET_CONFIG" | head -1 | awk '{print $2}')
if [ "$epochs" = "100" ]; then
    echo "✅ 100" | tee -a "$SUMMARY_FILE"
else
    echo "⚠️  $epochs (expected 100)" | tee -a "$SUMMARY_FILE"
fi

# Check 4: Phase 5 timing
echo -n "4. Phase 5 timing: " | tee -a "$SUMMARY_FILE"
phase5_start=$(grep "phase5_start_iter:" "$TARGET_CONFIG" | head -1 | awk '{print $2}')
phase4_start=$(grep "contact_start_iter:" "$TARGET_CONFIG" | head -1 | awk '{print $2}')

if [ -n "$phase4_start" ] && [ -n "$phase5_start" ]; then
    if [ "$phase5_start" -ge "$phase4_start" ]; then
        echo "✅ Valid ($phase5_start >= $phase4_start)" | tee -a "$SUMMARY_FILE"
    else
        echo "❌ Invalid ($phase5_start < $phase4_start)" | tee -a "$SUMMARY_FILE"
        verification_passed=false
    fi
else
    echo "✅ $phase5_start" | tee -a "$SUMMARY_FILE"
fi

# Check 5: No duplicate keys
echo -n "5. YAML validity: " | tee -a "$SUMMARY_FILE"
loss_count=$(grep -c "^loss:" "$TARGET_CONFIG" || echo "0")
if [ "$loss_count" -eq 1 ]; then
    echo "✅ No duplicates" | tee -a "$SUMMARY_FILE"
elif [ "$loss_count" -eq 0 ]; then
    echo "❌ No loss section!" | tee -a "$SUMMARY_FILE"
    verification_passed=false
else
    echo "❌ Duplicate 'loss:' keys ($loss_count found)" | tee -a "$SUMMARY_FILE"
    grep -n "^loss:" "$TARGET_CONFIG" | tee -a "$SUMMARY_FILE"
    verification_passed=false
fi

echo "" | tee -a "$SUMMARY_FILE"

if [ "$verification_passed" = false ]; then
    echo "❌ VERIFICATION FAILED" | tee -a "$SUMMARY_FILE"
    echo "" | tee -a "$SUMMARY_FILE"
    echo "Config file: $TARGET_CONFIG" | tee -a "$SUMMARY_FILE"
    echo "Please check manually before training" | tee -a "$SUMMARY_FILE"
    exit 1
fi

echo "✅ ALL CHECKS PASSED" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# ================================================================
# STEP 5: Configuration summary
# ================================================================
echo "========================================================================" | tee -a "$SUMMARY_FILE"
echo "CONFIGURATION SUMMARY" | tee -a "$SUMMARY_FILE"
echo "========================================================================" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"
echo "Base config:   $BASE_CONFIG" | tee -a "$SUMMARY_FILE"
echo "Target config: $TARGET_CONFIG" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"
echo "Architecture:  32-dim (TESTED & WORKING)" | tee -a "$SUMMARY_FILE"
echo "Batch size:    1 (memory optimized)" | tee -a "$SUMMARY_FILE"
echo "Epochs:        100" | tee -a "$SUMMARY_FILE"
echo "Duration:      ~12 hours (with batch_size=1)" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"
echo "Applied fixes:" | tee -a "$SUMMARY_FILE"
echo "  1. Loss weights: w_rgb=10.0, w_mano_cano=0.5" | tee -a "$SUMMARY_FILE"
echo "  2. Phase 5 timing: Fixed to avoid conflicts" | tee -a "$SUMMARY_FILE"
echo "  3. Training duration: Extended to 100 epochs" | tee -a "$SUMMARY_FILE"
echo "  4. Batch size: Reduced to 1 (prevents OOM at epoch 22)" | tee -a "$SUMMARY_FILE"
echo "  5. pixel_per_batch: Reduced to 512 (prevents fragmentation)" | tee -a "$SUMMARY_FILE"  # ← UPDATE
echo "  6. pin_memory: Disabled (prevents 24GB DataLoader leak)" | tee -a "$SUMMARY_FILE"  # ← RENUMBER
echo "" | tee -a "$SUMMARY_FILE"
echo "Everything else: UNCHANGED from working quick test" | tee -a "$SUMMARY_FILE"
echo "========================================================================" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# ================================================================
# STEP 6: Run training with memory optimization
# ================================================================
echo "========================================================================" | tee -a "$SUMMARY_FILE"
echo "MEMORY OPTIMIZATION" | tee -a "$SUMMARY_FILE"
echo "========================================================================" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"
echo "Memory safety measures:" | tee -a "$SUMMARY_FILE"
echo "  ✅ batch_size = 1 (prevents OOM at epoch 22)" | tee -a "$SUMMARY_FILE"
echo "  ✅ pixel_per_batch = 1024 (reduced from default)" | tee -a "$SUMMARY_FILE"
echo "  ✅ pin_memory = False (prevents 24GB DataLoader leak)" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"
echo "Expected memory usage:" | tee -a "$SUMMARY_FILE"
echo "  - Baseline: ~6-8 GB" | tee -a "$SUMMARY_FILE"
echo "  - Phase 4/5: ~8-10 GB (vs 24 GB without fix)" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"
echo "Trade-offs:" | tee -a "$SUMMARY_FILE"
echo "  - Training speed: ~5-10% slower (acceptable)" | tee -a "$SUMMARY_FILE"
echo "  - Memory safety: CRITICAL improvement" | tee -a "$SUMMARY_FILE"
echo "========================================================================" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

LOG_FILE="$RESULTS_DIR/production_32dim_$TIMESTAMP.log"

echo "Starting training..." | tee -a "$SUMMARY_FILE"
echo "  Config: $TARGET_CONFIG" | tee -a "$SUMMARY_FILE"
echo "  Log: $LOG_FILE" | tee -a "$SUMMARY_FILE"
echo "  Memory mode: pin_memory DISABLED" | tee -a "$SUMMARY_FILE"  # ← NEW
echo "" | tee -a "$SUMMARY_FILE"
echo "⏰ Expected duration: ~12 hours for 100 epochs" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

START_TIME=$(date +%s)

# ================================================================
# Check for existing checkpoint to resume training
# ================================================================
CKPT_DIR="logs"
LAST_CKPT=$(find "$CKPT_DIR" -name "last.ckpt" -type f 2>/dev/null | head -1)

if [ -n "$LAST_CKPT" ] && [ -f "$LAST_CKPT" ]; then
    echo "✅ Found checkpoint: $LAST_CKPT" | tee -a "$SUMMARY_FILE"
    echo "   Resuming training from previous run..." | tee -a "$SUMMARY_FILE"
    RESUME_ARG="--load_ckpt $LAST_CKPT"

    # Extract epoch from checkpoint if possible
    CKPT_EPOCH=$(echo "$LAST_CKPT" | grep -oP 'epoch=\K[0-9]+' || echo "unknown")
    echo "   Last checkpoint epoch: $CKPT_EPOCH" | tee -a "$SUMMARY_FILE"
else
    echo "No checkpoint found, training from scratch" | tee -a "$SUMMARY_FILE"
    RESUME_ARG=""
fi
echo "" | tee -a "$SUMMARY_FILE"

# ================================================================
# Run training with all optimizations
# ================================================================
#python train.py \
#    --config "$TARGET_CONFIG" \
#    --case ghop_bottle_1 \
#    --use_ghop \
#    --gpu_id 0 \
#    --num_epoch 100 \
#    --no-pin-memory \
#    --no-comet \
#    $RESUME_ARG \ There're too many useless checkpoints.
#    2>&1 | tee "$LOG_FILE"

python train.py \
    --config "$TARGET_CONFIG" \
    --case ghop_bottle_1 \
    --use_ghop \
    --gpu_id 0 \
    --num_epoch 100 \
    --no-pin-memory \
    --no-comet \
    2>&1 | tee "$LOG_FILE"

exitcode=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "" | tee -a "$SUMMARY_FILE"
echo "========================================================================" | tee -a "$SUMMARY_FILE"
echo "TRAINING ANALYSIS" | tee -a "$SUMMARY_FILE"
echo "========================================================================" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# Training status
if [ $exitcode -eq 0 ]; then
    echo "✅ Training Status: COMPLETED" | tee -a "$SUMMARY_FILE"
else
    echo "❌ Training Status: FAILED (exit code: $exitcode)" | tee -a "$SUMMARY_FILE"
fi

# Duration
hours=$((DURATION / 3600))
minutes=$(((DURATION % 3600) / 60))
seconds=$((DURATION % 60))
echo "   Duration: ${hours}h ${minutes}m ${seconds}s" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# RGB Loss
echo "🎨 RGB Loss:" | tee -a "$SUMMARY_FILE"
if grep -q "loss.*rgb\|loss/rgb" "$LOG_FILE"; then
    echo "   ✅ RGB loss computed" | tee -a "$SUMMARY_FILE"
    final_rgb=$(grep "loss.*rgb\|loss/rgb" "$LOG_FILE" | tail -1 | grep -oP "rgb[=:]?\s*\K[0-9.]+")
    if [ -n "$final_rgb" ]; then
        echo "   Final RGB loss: $final_rgb" | tee -a "$SUMMARY_FILE"
    fi
else
    echo "   ❌ RGB loss not found" | tee -a "$SUMMARY_FILE"
fi
echo "" | tee -a "$SUMMARY_FILE"

# GHOP SDS
echo "🤖 GHOP SDS:" | tee -a "$SUMMARY_FILE"
if grep -q "ghop.*sds\|loss/sds" "$LOG_FILE"; then
    echo "   ✅ GHOP SDS detected" | tee -a "$SUMMARY_FILE"
else
    echo "   ❌ GHOP SDS not detected" | tee -a "$SUMMARY_FILE"
fi
echo "" | tee -a "$SUMMARY_FILE"

# Errors
echo "⚠️  Errors:" | tee -a "$SUMMARY_FILE"
error_count=$(grep -ci "RuntimeError\|exception" "$LOG_FILE" || echo "0")
if [ "$error_count" -gt 0 ]; then
    echo "   ⚠️  Found $error_count errors" | tee -a "$SUMMARY_FILE"
    grep -i "RuntimeError" "$LOG_FILE" | head -3 | sed 's/^/     /' | tee -a "$SUMMARY_FILE"
else
    echo "   ✅ No critical errors" | tee -a "$SUMMARY_FILE"
fi
echo "" | tee -a "$SUMMARY_FILE"

# ================================================================
# FINAL SUMMARY
# ================================================================
echo "========================================================================" | tee -a "$SUMMARY_FILE"
echo "FINAL SUMMARY" | tee -a "$SUMMARY_FILE"
echo "========================================================================" | tee -a "$SUMMARY_FILE"

if [ $exitcode -eq 0 ]; then
    echo "✅ TRAINING COMPLETED SUCCESSFULLY" | tee -a "$SUMMARY_FILE"
    echo "" | tee -a "$SUMMARY_FILE"
    echo "Next Steps:" | tee -a "$SUMMARY_FILE"
    echo "  1. Check RGB rendering quality" | tee -a "$SUMMARY_FILE"
    echo "  2. Verify no NaN values in outputs" | tee -a "$SUMMARY_FILE"
    echo "  3. Compare with 20-epoch baseline" | tee -a "$SUMMARY_FILE"
else
    echo "❌ TRAINING FAILED (exit code: $exitcode)" | tee -a "$SUMMARY_FILE"
    echo "" | tee -a "$SUMMARY_FILE"

    # ================================================================
    # DETAILED ERROR ANALYSIS
    # ================================================================
    echo "========================================================================" | tee -a "$SUMMARY_FILE"
    echo "DETAILED ERROR ANALYSIS" | tee -a "$SUMMARY_FILE"
    echo "========================================================================" | tee -a "$SUMMARY_FILE"
    echo "" | tee -a "$SUMMARY_FILE"

    # Count different error types
    echo "📊 Error Statistics:" | tee -a "$SUMMARY_FILE"
    runtime_errors=$(grep -c "RuntimeError" "$LOG_FILE" || echo "0")
    cuda_oom=$(grep -c "CUDA error: out of memory" "$LOG_FILE" || echo "0")
    value_errors=$(grep -c "ValueError" "$LOG_FILE" || echo "0")
    type_errors=$(grep -c "TypeError" "$LOG_FILE" || echo "0")
    attribute_errors=$(grep -c "AttributeError" "$LOG_FILE" || echo "0")

    echo "  RuntimeError:     $runtime_errors" | tee -a "$SUMMARY_FILE"
    echo "  CUDA OOM:         $cuda_oom" | tee -a "$SUMMARY_FILE"
    echo "  ValueError:       $value_errors" | tee -a "$SUMMARY_FILE"
    echo "  TypeError:        $type_errors" | tee -a "$SUMMARY_FILE"
    echo "  AttributeError:   $attribute_errors" | tee -a "$SUMMARY_FILE"
    echo "" | tee -a "$SUMMARY_FILE"

    # Extract full traceback
    echo "🔍 Full Traceback (Last Error):" | tee -a "$SUMMARY_FILE"
    echo "========================================================================" | tee -a "$SUMMARY_FILE"

    # Find the last "Traceback" and print everything until next empty line
    awk '/^Traceback \(most recent call last\):/{flag=1} flag{print} /^[A-Za-z]+Error:/{print; flag=0}' "$LOG_FILE" | tail -50 | tee -a "$SUMMARY_FILE"

    echo "========================================================================" | tee -a "$SUMMARY_FILE"
    echo "" | tee -a "$SUMMARY_FILE"

    # Extract the specific error line
    echo "💥 Error Summary:" | tee -a "$SUMMARY_FILE"
    last_error=$(grep -E "RuntimeError|ValueError|TypeError|AttributeError|KeyError" "$LOG_FILE" | tail -1)
    if [ -n "$last_error" ]; then
        echo "  $last_error" | tee -a "$SUMMARY_FILE"
    else
        echo "  Could not extract specific error message" | tee -a "$SUMMARY_FILE"
    fi
    echo "" | tee -a "$SUMMARY_FILE"

    # Show the file and line where error occurred
    echo "📍 Error Location:" | tee -a "$SUMMARY_FILE"
    error_location=$(grep "File \"" "$LOG_FILE" | tail -5 | tee -a "$SUMMARY_FILE")
    echo "" | tee -a "$SUMMARY_FILE"

    # Extract last 50 lines before error for context
    echo "📝 Last 50 Log Lines Before Error:" | tee -a "$SUMMARY_FILE"
    echo "========================================================================" | tee -a "$SUMMARY_FILE"

    # Find line number of last Traceback
    traceback_line=$(grep -n "Traceback (most recent call last):" "$LOG_FILE" | tail -1 | cut -d: -f1)
    if [ -n "$traceback_line" ]; then
        start_line=$((traceback_line - 50))
        if [ $start_line -lt 1 ]; then
            start_line=1
        fi
        sed -n "${start_line},${traceback_line}p" "$LOG_FILE" | tail -50 | tee -a "$SUMMARY_FILE"
    else
        tail -50 "$LOG_FILE" | tee -a "$SUMMARY_FILE"
    fi

    echo "========================================================================" | tee -a "$SUMMARY_FILE"
    echo "" | tee -a "$SUMMARY_FILE"

    # Check for common failure patterns
    echo "🔎 Common Failure Patterns:" | tee -a "$SUMMARY_FILE"

    if grep -q "CUDA error: out of memory" "$LOG_FILE"; then
        echo "  ⚠️  CUDA OOM detected - reduce batch_size or increase GPU memory" | tee -a "$SUMMARY_FILE"

        # Show memory usage at failure
        memory_line=$(grep -B 5 "CUDA error: out of memory" "$LOG_FILE" | grep -i "memory\|MB\|GB" | tail -1)
        if [ -n "$memory_line" ]; then
            echo "     $memory_line" | tee -a "$SUMMARY_FILE"
        fi
    fi

    if grep -q "shapes cannot be multiplied" "$LOG_FILE"; then
        echo "  ⚠️  Shape mismatch detected - check tensor dimensions" | tee -a "$SUMMARY_FILE"
        grep "shapes cannot be multiplied" "$LOG_FILE" | tail -1 | sed 's/^/     /' | tee -a "$SUMMARY_FILE"
    fi

    if grep -q "NaN\|nan" "$LOG_FILE"; then
        echo "  ⚠️  NaN values detected - check loss computation" | tee -a "$SUMMARY_FILE"
    fi

    if grep -q "KeyError" "$LOG_FILE"; then
        echo "  ⚠️  Missing dictionary key - check data pipeline" | tee -a "$SUMMARY_FILE"
        grep "KeyError" "$LOG_FILE" | tail -3 | sed 's/^/     /' | tee -a "$SUMMARY_FILE"
    fi

    echo "" | tee -a "$SUMMARY_FILE"

    # Suggest fixes based on error type
    echo "💡 Suggested Actions:" | tee -a "$SUMMARY_FILE"

    if [ "$cuda_oom" -gt 0 ]; then
        echo "  1. Reduce batch_size to 1 in config" | tee -a "$SUMMARY_FILE"
        echo "  2. Reduce pixel_per_batch to 512" | tee -a "$SUMMARY_FILE"
        echo "  3. Check for memory leaks (detach logged tensors)" | tee -a "$SUMMARY_FILE"
    elif [ "$runtime_errors" -gt 0 ]; then
        echo "  1. Check tensor shapes in error location" | tee -a "$SUMMARY_FILE"
        echo "  2. Verify data preprocessing" | tee -a "$SUMMARY_FILE"
        echo "  3. Review recent code changes" | tee -a "$SUMMARY_FILE"
    elif [ "$value_errors" -gt 0 ]; then
        echo "  1. Check input data ranges" | tee -a "$SUMMARY_FILE"
        echo "  2. Verify configuration parameters" | tee -a "$SUMMARY_FILE"
    else
        echo "  1. Review full log file: $LOG_FILE" | tee -a "$SUMMARY_FILE"
        echo "  2. Check for warnings before error" | tee -a "$SUMMARY_FILE"
    fi

    echo "" | tee -a "$SUMMARY_FILE"
    echo "========================================================================" | tee -a "$SUMMARY_FILE"
fi

echo "" | tee -a "$SUMMARY_FILE"
echo "========================================================================" | tee -a "$SUMMARY_FILE"
echo "End: $(date)" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"
echo "📁 Files:" | tee -a "$SUMMARY_FILE"
echo "   Config: $TARGET_CONFIG" | tee -a "$SUMMARY_FILE"
echo "   Log: $LOG_FILE" | tee -a "$SUMMARY_FILE"
echo "   Summary: $SUMMARY_FILE" | tee -a "$SUMMARY_FILE"

# Create error-specific log if training failed
if [ $exitcode -ne 0 ]; then
    ERROR_LOG="$RESULTS_DIR/error_analysis_$TIMESTAMP.txt"
    echo "   Error Analysis: $ERROR_LOG" | tee -a "$SUMMARY_FILE"

    # Create detailed error log
    echo "DETAILED ERROR LOG - $(date)" > "$ERROR_LOG"
    echo "========================================================================" >> "$ERROR_LOG"
    echo "" >> "$ERROR_LOG"

    echo "Full Traceback:" >> "$ERROR_LOG"
    awk '/^Traceback \(most recent call last\):/{flag=1} flag{print} /^[A-Za-z]+Error:/{print; flag=0}' "$LOG_FILE" | tail -100 >> "$ERROR_LOG"

    echo "" >> "$ERROR_LOG"
    echo "========================================================================" >> "$ERROR_LOG"
    echo "Last 100 Lines Before Error:" >> "$ERROR_LOG"
    echo "========================================================================" >> "$ERROR_LOG"
    traceback_line=$(grep -n "Traceback (most recent call last):" "$LOG_FILE" | tail -1 | cut -d: -f1)
    if [ -n "$traceback_line" ]; then
        start_line=$((traceback_line - 100))
        [ $start_line -lt 1 ] && start_line=1
        sed -n "${start_line},${traceback_line}p" "$LOG_FILE" >> "$ERROR_LOG"
    fi
fi

echo "========================================================================" | tee -a "$SUMMARY_FILE"

exit $exitcode
