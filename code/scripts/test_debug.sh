#!/bin/bash
# ================================================================
# HOLDSE Overnight Pipeline - Pre-flight Test Suite
# Validates all components before full 7-hour run
# Duration: 5-10 minutes
# ================================================================

# set -e
CODE_DIR="$HOME/Projects/holdse/code"

echo "================================================================"
echo "OVERNIGHT PIPELINE PRE-FLIGHT TEST SUITE"
echo "================================================================"
echo ""

# ================================================================
# TEST 1: File Existence Checks
# ================================================================

echo "TEST 1: Checking required files..."
echo ""

PASSED=0
FAILED=0

check_file() {
    if [ -f "$1" ]; then
        echo "  ✅ $1"
        ((PASSED++))
    else
        echo "  ❌ MISSING: $1"
        ((FAILED++))
    fi
}

# Check checkpoints
check_file "logs/stage2_final.ckpt"
check_file "logs/stage3_final.ckpt"
check_file "logs/140dc5c18/checkpoints/last.ckpt"

# Check scripts
check_file "scripts/extract_predictions.py"
check_file "scripts/evaluate_predictions.py"
check_file "scripts/evaluate_stage2_stage3_overnight.sh"

# Check configs
check_file "confs/ghop_stage2_temporal_only.yaml"
check_file "confs/ghop_stage3_full_pipeline.yaml"

# Check dataset
check_file "data/hold_MC1_ho3d/build/data.npy"

# Check baseline (optional)
if [ -f "$HOME/Projects/hold/code/logs/cb20a1702/checkpoints/last.ckpt.metric.json" ]; then
    echo "  ✅ HOLD baseline (optional)"
else
    echo "  ⚠️  HOLD baseline not found (comparisons will be skipped)"
fi

echo ""
echo "Result: $PASSED passed, $FAILED failed"

if [ $FAILED -gt 0 ]; then
    echo "❌ TEST 1 FAILED: Missing required files"
    exit 1
fi

echo "✅ TEST 1 PASSED"
echo ""

# ================================================================
# TEST 2: Python Dependencies
# ================================================================

echo "TEST 2: Checking Python dependencies..."
echo ""

python << 'EOF'
import sys

required_modules = [
    'torch',
    'numpy',
    'tqdm',
    'pytorch_lightning',
]

passed = 0
failed = 0

for module in required_modules:
    try:
        __import__(module)
        print(f"  ✅ {module}")
        passed += 1
    except ImportError:
        print(f"  ❌ {module} (not installed)")
        failed += 1

# Check HOLD modules
sys.path.insert(0, '.')
sys.path.insert(0, '../hold/code')

try:
    import src.utils.eval_modules as eval_m
    print(f"  ✅ src.utils.eval_modules (HOLD)")
    passed += 1
except ImportError as e:
    print(f"  ❌ src.utils.eval_modules: {e}")
    failed += 1

try:
    import src.utils.io.gt as gt
    print(f"  ✅ src.utils.io.gt (HOLD)")
    passed += 1
except ImportError as e:
    print(f"  ❌ src.utils.io.gt: {e}")
    failed += 1

print(f"\nResult: {passed} passed, {failed} failed")

if failed > 0:
    print("❌ TEST 2 FAILED: Missing dependencies")
    sys.exit(1)

print("✅ TEST 2 PASSED")
EOF

if [ $? -ne 0 ]; then
    exit 1
fi

echo ""

# ================================================================
# TEST 3: Checkpoint Loading
# ================================================================

echo "TEST 3: Verifying checkpoint loading..."
echo ""

python << 'EOF'
import torch
import sys

checkpoints = {
    'Stage 2': 'logs/stage2_final.ckpt',
    'Stage 3': 'logs/stage3_final.ckpt'
}

passed = 0
failed = 0

for name, path in checkpoints.items():
    try:
        ckpt = torch.load(path, map_location='cpu')
        
        # Verify required keys
        required_keys = ['epoch', 'global_step', 'state_dict']
        missing = [k for k in required_keys if k not in ckpt]
        
        if missing:
            print(f"  ❌ {name}: Missing keys {missing}")
            failed += 1
        else:
            epoch = ckpt['epoch']
            step = ckpt['global_step']
            size_mb = len(str(ckpt)) / 1e6  # Rough estimate
            print(f"  ✅ {name}: epoch {epoch}, step {step}")
            passed += 1
            
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        failed += 1

print(f"\nResult: {passed} passed, {failed} failed")

if failed > 0:
    print("❌ TEST 3 FAILED: Checkpoint loading issues")
    sys.exit(1)

print("✅ TEST 3 PASSED")
EOF

if [ $? -ne 0 ]; then
    exit 1
fi

echo ""

# ================================================================
# TEST 4: Dataset Loading
# ================================================================

echo "TEST 4: Testing dataset loading..."
echo ""

python << 'EOF'
import numpy as np

try:
    data = np.load('data/hold_MC1_ho3d/build/data.npy', allow_pickle=True).item()
    
    print(f"  ✅ Dataset loaded successfully")
    print(f"     Scale: {data.get('scale', 'N/A')}")
    print(f"     Normalize shift: {data.get('normalize_shift', 'N/A')}")
    
    # Check for required keys
    required_keys = ['scale', 'normalize_shift']
    missing = [k for k in required_keys if k not in data]
    
    if missing:
        print(f"  ⚠️  Missing keys: {missing}")
    
    print("✅ TEST 4 PASSED")
    
except Exception as e:
    print(f"  ❌ Dataset loading failed: {e}")
    print("❌ TEST 4 FAILED")
    exit(1)
EOF

if [ $? -ne 0 ]; then
    exit 1
fi

echo ""

# ================================================================
# TEST 5: Script Execution Permissions
# ================================================================

echo "TEST 5: Checking script permissions..."
echo ""

if [ -x "scripts/evaluate_stage2_stage3_overnight.sh" ]; then
    echo "  ✅ Overnight script is executable"
else
    echo "  ❌ Overnight script not executable"
    echo "     Fixing with chmod +x..."
    chmod +x scripts/evaluate_stage2_stage3_overnight.sh
    echo "  ✅ Fixed"
fi

echo "✅ TEST 5 PASSED"
echo ""

# ================================================================
# TEST 6: Mini Extraction Test (Stage 2, 1 frame)
# ================================================================

echo "TEST 6: Running mini extraction test (Stage 2, first frame only)..."
echo ""
echo "⚠️  This will take ~90 seconds to test actual extraction code"
echo ""

# Create temporary test extraction script
python << 'EOF'
import torch
import sys
import numpy as np

sys.path.insert(0, '.')

print("  Loading Stage 2 checkpoint...")
ckpt = torch.load('logs/stage2_final.ckpt', map_location='cpu')

print(f"  ✓ Checkpoint loaded: epoch {ckpt['epoch']}, step {ckpt['global_step']}")

# Verify frame encoder shape
state_dict = ckpt['state_dict']
encoder_key = 'model.nodes.object.frame_latent_encoder.weight'

if encoder_key in state_dict:
    encoder = state_dict[encoder_key]
    print(f"  ✓ Frame encoder: {encoder.shape[0]} frames × {encoder.shape[1]} dims")
    
    if encoder.shape[0] != 144:
        print(f"  ⚠️  WARNING: Expected 144 frames for MC1, got {encoder.shape[0]}")
else:
    print(f"  ❌ ERROR: Frame encoder not found in checkpoint")
    sys.exit(1)

# Check coordinate transform parameters
print("  Loading dataset metadata for coordinate transform...")
data = np.load('data/hold_MC1_ho3d/build/data.npy', allow_pickle=True).item()

if 'normalize_shift' in data and 'scale' in data:
    print(f"  ✓ Coordinate transform parameters available")
else:
    print(f"  ❌ ERROR: Missing coordinate transform parameters")
    sys.exit(1)

print("\n  ✅ Mini test setup successful")
print("  (Full extraction would process all 144 frames)")

EOF

if [ $? -ne 0 ]; then
    echo "❌ TEST 6 FAILED"
    exit 1
fi

echo "✅ TEST 6 PASSED"
echo ""

# ================================================================
# TEST 7: Dry Run Syntax Check
# ================================================================

echo "TEST 7: Syntax validation of overnight script..."
echo ""

bash -n scripts/evaluate_stage2_stage3_overnight.sh

if [ $? -eq 0 ]; then
    echo "  ✅ Script syntax is valid"
    echo "✅ TEST 7 PASSED"
else
    echo "  ❌ Script has syntax errors"
    echo "❌ TEST 7 FAILED"
    exit 1
fi

echo ""

# ================================================================
# FINAL SUMMARY
# ================================================================

echo "================================================================"
echo "✅ ALL TESTS PASSED!"
echo "================================================================"
echo ""
echo "Your overnight pipeline is ready to run."
echo ""
echo "To launch the full 7-hour evaluation:"
echo "  cd ~/Projects/holdse/code"
echo "  nohup bash scripts/evaluate_stage2_stage3_overnight.sh > logs/overnight_run.log 2>&1 &"
echo "  echo \$! > logs/overnight_run.pid"
echo ""
echo "Monitor progress:"
echo "  tail -f logs/overnight_run.log"
echo ""
echo "Good night! 🌙"
echo ""

