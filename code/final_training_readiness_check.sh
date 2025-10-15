#!/bin/bash
# File: ~/Projects/holdse/code/final_training_readiness_check.sh

echo "======================================================================"
echo "HOLDSE: Final Training Readiness Check"
echo "======================================================================"

cd ~/Projects/holdse/code

ALL_OK=true

# Check 1: Git repository
echo "[1/7] Git repository..."
if git rev-parse HEAD >/dev/null 2>&1; then
    COMMIT=$(git rev-parse --short HEAD)
    echo "✓ Git commit: $COMMIT"
else
    echo "❌ Git not initialized"
    ALL_OK=false
fi

# Check 2: Data symlink
echo "[2/7] Data symlink..."
if [ -L "./data" ] && [ -f "./data/hold_mug1_itw/build/data.npy" ]; then
    echo "✓ Data accessible: hold_mug1_itw"
else
    echo "❌ Data symlink broken"
    ALL_OK=false
fi

# Check 3: Body models symlink
echo "[3/7] Body models symlink..."
if [ -L "./body_models" ] && [ -f "./body_models/MANO_RIGHT.pkl" ]; then
    echo "✓ Body models accessible"
else
    echo "❌ Body models symlink broken"
    ALL_OK=false
fi

# Check 4: Saved models symlink
echo "[4/7] Saved models symlink..."
if [ -L "./saved_models" ] && [ -f "./saved_models/75268d864/checkpoints/last.ckpt" ]; then
    SIZE=$(stat -c%s ./saved_models/75268d864/checkpoints/last.ckpt 2>/dev/null || stat -f%z ./saved_models/75268d864/checkpoints/last.ckpt)
    echo "✓ Checkpoint accessible: $(echo "scale=2; $SIZE / 1024 / 1024" | bc) MB"
else
    echo "❌ Saved models symlink broken"
    ALL_OK=false
fi

# Check 5: Python imports
echo "[5/7] Python imports..."
python -c "
import sys
sys.path.insert(0, '../common')
from src.utils.parser import parser_args
import sys_utils
print('✓ Imports successful')
" 2>/dev/null || {
    echo "❌ Python imports failed"
    ALL_OK=false
}

# Check 6: CUDA fix
echo "[6/7] CUDA fix..."
if grep -q "FIX: Compute inverse on CPU" src/model/mano/server.py; then
    echo "✓ CUDA cuSPARSE fix applied"
else
    echo "❌ CUDA fix not applied"
    ALL_OK=false
fi

# Check 7: PyTorch checkpoint loading
echo "[7/7] PyTorch checkpoint loading..."
python -c "
import torch
ckpt = torch.load('saved_models/75268d864/checkpoints/last.ckpt', map_location='cpu')
assert 'state_dict' in ckpt, 'Invalid checkpoint structure'
print(f'✓ Checkpoint valid (epoch {ckpt.get(\"epoch\", \"N/A\")})')
" 2>/dev/null || {
    echo "❌ Checkpoint loading failed"
    ALL_OK=false
}

echo ""
echo "======================================================================"
if [ "$ALL_OK" = true ]; then
    echo "✅✅✅ ALL CHECKS PASSED - READY FOR TRAINING ✅✅✅"
    echo "======================================================================"
    echo ""
    echo "Run sanity training with:"
    echo "  python sanity_train.py --case hold_mug1_itw --shape_init 75268d864 --gpu_id 0"
    echo ""
    echo "Monitor training with:"
    echo "  tail -f logs/*/train.log"
else
    echo "❌ SOME CHECKS FAILED - REVIEW ERRORS ABOVE"
    echo "======================================================================"
    exit 1
fi