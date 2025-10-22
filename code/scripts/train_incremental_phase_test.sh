#!/bin/bash
# File: code/scripts/train_incremental_phase_test.sh
# PURPOSE: Systematically isolate memory leak (FIXED VERSION)
# FIX: Properly handle enabling specific phases, not just disabling all

set -e

cd ~/Projects/holdse/code

# ================================================================
# CONFIGURATION: Which phases to test
# ================================================================
# Set test mode: "baseline", "phase4", "phase5", "all_disabled"
TEST_MODE="${1:-baseline}"

# ================================================================
# SETUP: Results directory
# ================================================================
RESULTS_DIR="../phase_isolation_results"
mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="$RESULTS_DIR/test_${TEST_MODE}_${TIMESTAMP}.txt"

echo ""
echo "========================================================================" | tee "$SUMMARY_FILE"
echo "INCREMENTAL PHASE ISOLATION TEST (FIXED)" | tee -a "$SUMMARY_FILE"
echo "========================================================================" | tee -a "$SUMMARY_FILE"
echo "Start: $(date)" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"
echo "Test Mode: $TEST_MODE" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# ================================================================
# Define test configurations
# ================================================================
case "$TEST_MODE" in
    "baseline")
        echo "Strategy:" | tee -a "$SUMMARY_FILE"
        echo "  🎯 Test BASELINE HOLD ONLY" | tee -a "$SUMMARY_FILE"
        echo "  ❌ Phase 3 (GHOP): DISABLED" | tee -a "$SUMMARY_FILE"
        echo "  ❌ Phase 4 (Contact): DISABLED" | tee -a "$SUMMARY_FILE"
        echo "  ❌ Phase 5 (Temporal): DISABLED" | tee -a "$SUMMARY_FILE"

        PHASE3_ENABLED="false"
        PHASE4_ENABLED="false"
        PHASE5_ENABLED="false"
        NUM_EPOCHS=10
        ;;

    "phase4")
        echo "Strategy:" | tee -a "$SUMMARY_FILE"
        echo "  🎯 Test BASELINE + PHASE 4" | tee -a "$SUMMARY_FILE"
        echo "  ❌ Phase 3 (GHOP): DISABLED" | tee -a "$SUMMARY_FILE"
        echo "  ✅ Phase 4 (Contact): ENABLED" | tee -a "$SUMMARY_FILE"
        echo "  ❌ Phase 5 (Temporal): DISABLED" | tee -a "$SUMMARY_FILE"

        PHASE3_ENABLED="false"
        PHASE4_ENABLED="true"
        PHASE5_ENABLED="false"
        NUM_EPOCHS=100
        ;;

    "phase5")
        echo "Strategy:" | tee -a "$SUMMARY_FILE"
        echo "  🎯 Test BASELINE + PHASE 5" | tee -a "$SUMMARY_FILE"
        echo "  ❌ Phase 3 (GHOP): DISABLED" | tee -a "$SUMMARY_FILE"
        echo "  ❌ Phase 4 (Contact): DISABLED" | tee -a "$SUMMARY_FILE"
        echo "  ✅ Phase 5 (Temporal): ENABLED" | tee -a "$SUMMARY_FILE"

        PHASE3_ENABLED="false"
        PHASE4_ENABLED="false"
        PHASE5_ENABLED="true"
        NUM_EPOCHS=100
        ;;

    "phase4and5")
        echo "Strategy:" | tee -a "$SUMMARY_FILE"
        echo "  🎯 Test BASELINE + PHASE 4 + PHASE 5" | tee -a "$SUMMARY_FILE"
        echo "  ❌ Phase 3 (GHOP): DISABLED" | tee -a "$SUMMARY_FILE"
        echo "  ✅ Phase 4 (Contact): ENABLED" | tee -a "$SUMMARY_FILE"
        echo "  ✅ Phase 5 (Temporal): ENABLED" | tee -a "$SUMMARY_FILE"

        PHASE3_ENABLED="false"
        PHASE4_ENABLED="true"
        PHASE5_ENABLED="true"
        NUM_EPOCHS=100
        ;;

    *)
        echo "❌ Error: Unknown test mode: $TEST_MODE" | tee -a "$SUMMARY_FILE"
        echo "" | tee -a "$SUMMARY_FILE"
        echo "Usage: $0 [baseline|phase4|phase5|phase4and5]" | tee -a "$SUMMARY_FILE"
        exit 1
        ;;
esac

echo "" | tee -a "$SUMMARY_FILE"

# ================================================================
# STEP 1: Check for base config
# ================================================================
echo "Checking for base config..." | tee -a "$SUMMARY_FILE"

BASE_CONFIG="confs/ghop_quick_bottle_1.yaml"

if [ ! -f "$BASE_CONFIG" ]; then
    echo "❌ Error: Base config not found: $BASE_CONFIG" | tee -a "$SUMMARY_FILE"
    exit 1
fi

# Verify 32-dim
if ! grep -q "feature_vector_size: 32" "$BASE_CONFIG"; then
    echo "❌ Error: Base config is NOT 32-dim!" | tee -a "$SUMMARY_FILE"
    exit 1
fi

echo "✓ Found base config: $BASE_CONFIG" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# ================================================================
# STEP 2: Create test-specific config
# ================================================================
echo "Creating test config..." | tee -a "$SUMMARY_FILE"

TARGET_CONFIG="confs/phase_test_${TEST_MODE}_${TIMESTAMP}.yaml"

# Copy base config
cp "$BASE_CONFIG" "$TARGET_CONFIG"

echo "  ✓ Copied base config" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# ================================================================
# STEP 3: FIXED phase enable/disable logic
# ================================================================
echo "Applying phase-specific modifications..." | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# ================================================================
# FIXED: Set each phase's enabled status explicitly
# ================================================================
set_phase_enabled() {
    local phase_name=$1
    local enabled_value=$2  # "true" or "false"
    local config_file=$3

    echo "  🔧 Setting ${phase_name}.enabled = $enabled_value..." | tee -a "$SUMMARY_FILE"

    # Find the phase section and the FIRST "enabled:" line within it
    # Use awk for precise control
    awk -v phase="^${phase_name}:" -v val="$enabled_value" '
    BEGIN { in_section=0; done=0 }
    {
        if ($0 ~ phase) {
            in_section=1
            print
            next
        }

        if (in_section && !done && /^  enabled:/) {
            print "  enabled: " val
            done=1
            next
        }

        if (in_section && /^[a-z_]+:/) {
            in_section=0
        }

        print
    }
    ' "$config_file" > "${config_file}.tmp" && mv "${config_file}.tmp" "$config_file"

    # Verify the change
    phase_status=$(awk -v phase="^${phase_name}:" '
        BEGIN { in_section=0 }
        {
            if ($0 ~ phase) in_section=1
            if (in_section && /^  enabled:/) {
                print $2
                exit
            }
            if (in_section && /^[a-z_]+:/) exit
        }
    ' "$config_file")

    if [ "$phase_status" = "$enabled_value" ]; then
        echo "    ✓ ${phase_name}.enabled: $phase_status" | tee -a "$SUMMARY_FILE"
        return 0
    else
        echo "    ❌ Failed: ${phase_name}.enabled = $phase_status (expected: $enabled_value)" | tee -a "$SUMMARY_FILE"
        return 1
    fi
}

# Set each phase explicitly
verification_passed=true

if ! set_phase_enabled "phase3" "$PHASE3_ENABLED" "$TARGET_CONFIG"; then
    verification_passed=false
fi

if ! set_phase_enabled "phase4" "$PHASE4_ENABLED" "$TARGET_CONFIG"; then
    verification_passed=false
fi

if ! set_phase_enabled "phase5" "$PHASE5_ENABLED" "$TARGET_CONFIG"; then
    verification_passed=false
fi

echo "" | tee -a "$SUMMARY_FILE"

if [ "$verification_passed" = false ]; then
    echo "❌ Failed to set phase configurations correctly" | tee -a "$SUMMARY_FILE"
    exit 1
fi

# ================================================================
# Fix 1: Training duration
# ================================================================
echo "  Fix 1: Training duration" | tee -a "$SUMMARY_FILE"
sed -i "s/num_epochs: 1/num_epochs: $NUM_EPOCHS/" "$TARGET_CONFIG"
sed -i 's/max_steps: 20/max_steps: -1/' "$TARGET_CONFIG"
echo "    ✓ num_epochs: 1 → $NUM_EPOCHS" | tee -a "$SUMMARY_FILE"
echo "    ✓ max_steps: 20 → unlimited" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# ================================================================
# Fix 2: Loss weights
# ================================================================
echo "  Fix 2: Loss weights" | tee -a "$SUMMARY_FILE"

if ! grep -q "^loss:" "$TARGET_CONFIG"; then
    cat >> "$TARGET_CONFIG" << 'LOSS_EOF'

# ================================================================
# LOSS WEIGHTS
# ================================================================
loss:
  w_rgb: 10.0
  w_mask: 5.0
  w_mano_cano: 0.5
  w_eikonal: 0.1
  w_smooth: 0.01
  w_semantic: 1.0
  w_opacity_sparse: 0.1
LOSS_EOF
    echo "    ✓ Added loss weights section" | tee -a "$SUMMARY_FILE"
else
    echo "    ✓ Loss weights already present" | tee -a "$SUMMARY_FILE"
fi
echo "" | tee -a "$SUMMARY_FILE"

# ================================================================
# Fix 3: Memory optimization
# ================================================================
echo "  Fix 3: Memory optimization" | tee -a "$SUMMARY_FILE"

current_batch=$(grep "batch_size:" "$TARGET_CONFIG" | grep -v "#" | head -1 | awk '{print $2}')
if [ -z "$current_batch" ] || [ "$current_batch" -gt 1 ]; then
    sed -i '/^dataset:/,/^[a-z_]*:/ s/batch_size: [0-9]\+/batch_size: 1/' "$TARGET_CONFIG"
    echo "    ✓ batch_size: ${current_batch:-2} → 1" | tee -a "$SUMMARY_FILE"
fi

sed -i 's/pixel_per_batch: [0-9]\+/pixel_per_batch: 1024/' "$TARGET_CONFIG"
echo "    ✓ pixel_per_batch → 1024" | tee -a "$SUMMARY_FILE"

echo "" | tee -a "$SUMMARY_FILE"
echo "✅ All modifications applied" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# ================================================================
# STEP 4: Final verification
# ================================================================
echo "========================================================================" | tee -a "$SUMMARY_FILE"
echo "FINAL VERIFICATION" | tee -a "$SUMMARY_FILE"
echo "========================================================================" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

final_verification_passed=true

# Verify Phase 3
echo -n "1. Phase 3 (GHOP): " | tee -a "$SUMMARY_FILE"
phase3_status=$(awk '/^phase3:/ {in_p3=1} in_p3 && /^  enabled:/ {print $2; exit} /^phase4:/ {exit}' "$TARGET_CONFIG")
if [ "$phase3_status" = "$PHASE3_ENABLED" ]; then
    echo "✅ $phase3_status (expected: $PHASE3_ENABLED)" | tee -a "$SUMMARY_FILE"
else
    echo "❌ $phase3_status (expected: $PHASE3_ENABLED)" | tee -a "$SUMMARY_FILE"
    final_verification_passed=false
fi

# Verify Phase 4
echo -n "2. Phase 4 (Contact): " | tee -a "$SUMMARY_FILE"
phase4_status=$(awk '/^phase4:/ {in_p4=1} in_p4 && /^  enabled:/ {print $2; exit} /^phase5:/ {exit}' "$TARGET_CONFIG")
if [ "$phase4_status" = "$PHASE4_ENABLED" ]; then
    echo "✅ $phase4_status (expected: $PHASE4_ENABLED)" | tee -a "$SUMMARY_FILE"
else
    echo "❌ $phase4_status (expected: $PHASE4_ENABLED)" | tee -a "$SUMMARY_FILE"
    final_verification_passed=false
fi

# Verify Phase 5
echo -n "3. Phase 5 (Temporal): " | tee -a "$SUMMARY_FILE"
phase5_status=$(awk '/^phase5:/ {in_p5=1} in_p5 && /^  enabled:/ {print $2; exit} /^training:/ {exit}' "$TARGET_CONFIG")
if [ "$phase5_status" = "$PHASE5_ENABLED" ]; then
    echo "✅ $phase5_status (expected: $PHASE5_ENABLED)" | tee -a "$SUMMARY_FILE"
else
    echo "❌ $phase5_status (expected: $PHASE5_ENABLED)" | tee -a "$SUMMARY_FILE"
    final_verification_passed=false
fi

echo -n "4. Architecture: " | tee -a "$SUMMARY_FILE"
if grep -q "feature_vector_size: 32" "$TARGET_CONFIG"; then
    echo "✅ 32-dim" | tee -a "$SUMMARY_FILE"
else
    echo "❌ NOT 32-dim" | tee -a "$SUMMARY_FILE"
    final_verification_passed=false
fi

echo -n "5. Batch size: " | tee -a "$SUMMARY_FILE"
batch_size=$(grep "batch_size:" "$TARGET_CONFIG" | grep -v "#" | head -1 | awk '{print $2}')
if [ "$batch_size" = "1" ]; then
    echo "✅ 1 (optimized)" | tee -a "$SUMMARY_FILE"
else
    echo "⚠️  $batch_size" | tee -a "$SUMMARY_FILE"
fi

echo "" | tee -a "$SUMMARY_FILE"

if [ "$final_verification_passed" = false ]; then
    echo "❌ FINAL VERIFICATION FAILED" | tee -a "$SUMMARY_FILE"
    echo "" | tee -a "$SUMMARY_FILE"
    echo "Config file: $TARGET_CONFIG" | tee -a "$SUMMARY_FILE"
    echo "" | tee -a "$SUMMARY_FILE"
    echo "Review phase settings:" | tee -a "$SUMMARY_FILE"
    grep -A 1 "^phase[345]:" "$TARGET_CONFIG" | grep "enabled:" | tee -a "$SUMMARY_FILE"
    exit 1
fi

echo "✅ ALL VERIFICATIONS PASSED" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# ================================================================
# STEP 5: Configuration summary
# ================================================================
echo "========================================================================" | tee -a "$SUMMARY_FILE"
echo "CONFIGURATION SUMMARY" | tee -a "$SUMMARY_FILE"
echo "========================================================================" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"
echo "Test Mode:     $TEST_MODE" | tee -a "$SUMMARY_FILE"
echo "Base config:   $BASE_CONFIG" | tee -a "$SUMMARY_FILE"
echo "Target config: $TARGET_CONFIG" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"
echo "Phase Configuration:" | tee -a "$SUMMARY_FILE"

if [ "$PHASE3_ENABLED" = "true" ]; then
    echo "  Phase 3 (GHOP):     ✅ ENABLED" | tee -a "$SUMMARY_FILE"
else
    echo "  Phase 3 (GHOP):     ❌ DISABLED" | tee -a "$SUMMARY_FILE"
fi

if [ "$PHASE4_ENABLED" = "true" ]; then
    echo "  Phase 4 (Contact):  ✅ ENABLED" | tee -a "$SUMMARY_FILE"
else
    echo "  Phase 4 (Contact):  ❌ DISABLED" | tee -a "$SUMMARY_FILE"
fi

if [ "$PHASE5_ENABLED" = "true" ]; then
    echo "  Phase 5 (Temporal): ✅ ENABLED" | tee -a "$SUMMARY_FILE"
else
    echo "  Phase 5 (Temporal): ❌ DISABLED" | tee -a "$SUMMARY_FILE"
fi

echo "" | tee -a "$SUMMARY_FILE"
echo "Training Settings:" | tee -a "$SUMMARY_FILE"
echo "  Architecture:  32-dim" | tee -a "$SUMMARY_FILE"
echo "  Batch size:    1" | tee -a "$SUMMARY_FILE"
echo "  Epochs:        $NUM_EPOCHS" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"
echo "========================================================================" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# ================================================================
# STEP 6: Run training
# ================================================================
LOG_FILE="$RESULTS_DIR/test_${TEST_MODE}_${TIMESTAMP}.log"
MEMORY_LOG="$RESULTS_DIR/memory_${TEST_MODE}_${TIMESTAMP}.csv"

echo "Starting training..." | tee -a "$SUMMARY_FILE"
echo "  Config: $TARGET_CONFIG" | tee -a "$SUMMARY_FILE"
echo "  Log: $LOG_FILE" | tee -a "$SUMMARY_FILE"
echo "  Memory Log: $MEMORY_LOG" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# Start memory monitoring
cat > /tmp/monitor_memory_$TIMESTAMP.sh << 'MONITOR_EOF'
#!/bin/bash
LOG_FILE="$1"
echo "timestamp,elapsed_seconds,memory_mb,growth_mb" > "$LOG_FILE"
START_TIME=$(date +%s)
BASELINE_MEM=""

while true; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))

    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)

    if [ -n "$MEM" ]; then
        if [ -z "$BASELINE_MEM" ]; then
            BASELINE_MEM=$MEM
            GROWTH=0
        else
            GROWTH=$((MEM - BASELINE_MEM))
        fi

        TIMESTAMP=$(date +%Y-%m-%d\ %H:%M:%S)
        echo "$TIMESTAMP,$ELAPSED,$MEM,$GROWTH" >> "$LOG_FILE"
    fi

    sleep 30
done
MONITOR_EOF

chmod +x /tmp/monitor_memory_$TIMESTAMP.sh
nohup /tmp/monitor_memory_$TIMESTAMP.sh "$MEMORY_LOG" > /dev/null 2>&1 &
MONITOR_PID=$!

echo "  Memory monitor PID: $MONITOR_PID" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

START_TIME=$(date +%s)

# Train WITHOUT --use_ghop flag
python train.py \
    --config "$TARGET_CONFIG" \
    --case ghop_bottle_1 \
    --gpu_id 0 \
    --num_epoch $NUM_EPOCHS 2>&1 | tee "$LOG_FILE"

exitcode=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Stop memory monitoring
kill $MONITOR_PID 2>/dev/null

echo "" | tee -a "$SUMMARY_FILE"
echo "========================================================================" | tee -a "$SUMMARY_FILE"
echo "TEST RESULTS" | tee -a "$SUMMARY_FILE"
echo "========================================================================" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

if [ $exitcode -eq 0 ]; then
    echo "✅ Training Status: COMPLETED" | tee -a "$SUMMARY_FILE"
else
    echo "❌ Training Status: FAILED (exit code: $exitcode)" | tee -a "$SUMMARY_FILE"
fi

hours=$((DURATION / 3600))
minutes=$(((DURATION % 3600) / 60))
seconds=$((DURATION % 60))
echo "   Duration: ${hours}h ${minutes}m ${seconds}s" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# Memory Analysis
echo "📊 MEMORY ANALYSIS:" | tee -a "$SUMMARY_FILE"
echo "========================================================================" | tee -a "$SUMMARY_FILE"

if [ -f "$MEMORY_LOG" ]; then
    FINAL_MEM=$(tail -1 "$MEMORY_LOG" | cut -d',' -f3)
    FINAL_GROWTH=$(tail -1 "$MEMORY_LOG" | cut -d',' -f4)
    SAMPLES=$(wc -l < "$MEMORY_LOG")

    echo "  Total samples: $((SAMPLES - 1))" | tee -a "$SUMMARY_FILE"
    echo "  Final memory: ${FINAL_MEM} MB" | tee -a "$SUMMARY_FILE"
    echo "  Total growth: ${FINAL_GROWTH} MB" | tee -a "$SUMMARY_FILE"
    echo "" | tee -a "$SUMMARY_FILE"

    # Verdict
    if [ $exitcode -eq 0 ] && [ "$FINAL_MEM" -lt 10000 ]; then
        echo "  ✅ VERDICT: Memory STABLE (< 10 GB)" | tee -a "$SUMMARY_FILE"
        echo "  ✅ CONCLUSION: This configuration is LEAK-FREE" | tee -a "$SUMMARY_FILE"
    elif [ "$FINAL_MEM" -lt 20000 ]; then
        echo "  ⚠️  VERDICT: Memory elevated (10-20 GB)" | tee -a "$SUMMARY_FILE"
        echo "  ⚠️  CONCLUSION: Minor leak or overhead" | tee -a "$SUMMARY_FILE"
    else
        echo "  ❌ VERDICT: Memory excessive (> 20 GB)" | tee -a "$SUMMARY_FILE"
        echo "  ❌ CONCLUSION: Severe memory leak detected" | tee -a "$SUMMARY_FILE"
    fi
else
    echo "  ❌ Memory log not found!" | tee -a "$SUMMARY_FILE"
fi

echo "========================================================================" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# Test-specific conclusions
echo "📋 TEST-SPECIFIC CONCLUSION:" | tee -a "$SUMMARY_FILE"
echo "========================================================================" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

case "$TEST_MODE" in
    "baseline")
        if [ $exitcode -eq 0 ] && [ -n "$FINAL_MEM" ] && [ "$FINAL_MEM" -lt 10000 ]; then
            echo "✅ BASELINE HOLD IS CLEAN" | tee -a "$SUMMARY_FILE"
            echo "   Leak is definitively in Phase 3/4/5" | tee -a "$SUMMARY_FILE"
            echo "" | tee -a "$SUMMARY_FILE"
            echo "Next: Test with Phase 4 enabled:" | tee -a "$SUMMARY_FILE"
            echo "  bash scripts/train_incremental_phase_test_fixed.sh phase4" | tee -a "$SUMMARY_FILE"
        fi
        ;;

    "phase4")
        if [ $exitcode -eq 0 ] && [ -n "$FINAL_MEM" ] && [ "$FINAL_MEM" -lt 10000 ]; then
            echo "✅ PHASE 4 (CONTACT) IS CLEAN" | tee -a "$SUMMARY_FILE"
            echo "   Leak is NOT in contact refinement" | tee -a "$SUMMARY_FILE"
            echo "" | tee -a "$SUMMARY_FILE"
            echo "Next: Test with Phase 5 enabled:" | tee -a "$SUMMARY_FILE"
            echo "  bash scripts/train_incremental_phase_test_fixed.sh phase5" | tee -a "$SUMMARY_FILE"
        else
            echo "❌ PHASE 4 (CONTACT) CAUSES LEAK" | tee -a "$SUMMARY_FILE"
            echo "   Investigate: Mesh extraction, PyTorch3D caching" | tee -a "$SUMMARY_FILE"
        fi
        ;;

    "phase5")
        if [ $exitcode -eq 0 ] && [ -n "$FINAL_MEM" ] && [ "$FINAL_MEM" -lt 10000 ]; then
            echo "✅ PHASE 5 (TEMPORAL) IS CLEAN" | tee -a "$SUMMARY_FILE"
            echo "   Leak is NOT in temporal consistency" | tee -a "$SUMMARY_FILE"
            echo "" | tee -a "$SUMMARY_FILE"
            echo "Next: Test with Phase 4+5 together:" | tee -a "$SUMMARY_FILE"
            echo "  bash scripts/train_incremental_phase_test_fixed.sh phase4and5" | tee -a "$SUMMARY_FILE"
        else
            echo "❌ PHASE 5 (TEMPORAL) CAUSES LEAK" | tee -a "$SUMMARY_FILE"
            echo "   Investigate: History buffers, temporal window" | tee -a "$SUMMARY_FILE"
        fi
        ;;

    "phase4and5")
        if [ $exitcode -eq 0 ] && [ -n "$FINAL_MEM" ] && [ "$FINAL_MEM" -lt 10000 ]; then
            echo "✅ PHASE 4+5 TOGETHER ARE CLEAN" | tee -a "$SUMMARY_FILE"
            echo "   Leak is ONLY in Phase 3 (GHOP)" | tee -a "$SUMMARY_FILE"
        else
            echo "❌ PHASE 4+5 INTERACTION CAUSES LEAK" | tee -a "$SUMMARY_FILE"
            echo "   Phases OK individually but leak together" | tee -a "$SUMMARY_FILE"
        fi
        ;;
esac

echo "========================================================================" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# Final summary
echo "========================================================================" | tee -a "$SUMMARY_FILE"
echo "TEST COMPLETE" | tee -a "$SUMMARY_FILE"
echo "========================================================================" | tee -a "$SUMMARY_FILE"
echo "End: $(date)" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"
echo "📁 Files:" | tee -a "$SUMMARY_FILE"
echo "   Config:      $TARGET_CONFIG" | tee -a "$SUMMARY_FILE"
echo "   Log:         $LOG_FILE" | tee -a "$SUMMARY_FILE"
echo "   Memory Log:  $MEMORY_LOG" | tee -a "$SUMMARY_FILE"
echo "   Summary:     $SUMMARY_FILE" | tee -a "$SUMMARY_FILE"
echo "========================================================================" | tee -a "$SUMMARY_FILE"

exit $exitcode
