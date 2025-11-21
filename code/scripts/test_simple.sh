#!/bin/bash
# Simple test without early exit

echo "=== SIMPLE VALIDATION TEST ==="
echo ""

# Test 1: Files
echo "TEST 1: Required files"
PASS=0
FAIL=0

check() {
    if [ -f "$1" ]; then
        echo "  ✅ $1"
        ((PASS++))
    else
        echo "  ❌ MISSING: $1"
        ((FAIL++))
    fi
}

check "logs/stage2_final.ckpt"
check "logs/stage3_final.ckpt"
check "logs/140dc5c18/checkpoints/last.ckpt"
check "scripts/extract_predictions.py"
check "scripts/evaluate_predictions.py"
check "confs/ghop_stage2_temporal_only.yaml"
check "confs/ghop_stage3_full_pipeline.yaml"
check "data/hold_MC1_ho3d/build/data.npy"

echo ""
echo "Files: $PASS passed, $FAIL failed"
echo ""

# Test 2: Checkpoint loading
echo "TEST 2: Checkpoint loading"
python << 'PYEOF'
import torch
try:
    torch.load('logs/stage2_final.ckpt', map_location='cpu')
    print("  ✅ Stage 2 checkpoint loads")
except Exception as e:
    print(f"  ❌ Stage 2 failed: {e}")

try:
    torch.load('logs/stage3_final.ckpt', map_location='cpu')
    print("  ✅ Stage 3 checkpoint loads")
except Exception as e:
    print(f"  ❌ Stage 3 failed: {e}")
PYEOF

echo ""
echo "TEST 3: Script syntax"
if bash -n scripts/evaluate_stage2_stage3_overnight.sh 2>/dev/null; then
    echo "  ✅ Overnight script syntax OK"
else
    echo "  ❌ Syntax error in overnight script"
fi

echo ""
echo "=== TESTS COMPLETE ==="

if [ $FAIL -eq 0 ]; then
    echo "✅ Ready to launch overnight run"
else
    echo "❌ Fix missing files first"
fi
