#!/bin/bash
# File: ~/Projects/holdse/code/verify_all_symlinks.sh

echo "======================================================================"
echo "HOLDSE: Complete Symlink Verification"
echo "======================================================================"

cd ~/Projects/holdse/code

# Check 1: Data symlink
echo "[1/3] Checking data symlink..."
if [ -L "./data" ] && [ -d "./data" ]; then
    TARGET=$(readlink -f ./data)
    NUM_SEQS=$(ls -1 ./data/ | grep "^hold_" | wc -l)
    echo "✓ Data symlink OK"
    echo "  Target: $TARGET"
    echo "  Sequences: $NUM_SEQS"
else
    echo "❌ Data symlink missing or broken"
fi

# Check 2: Body models symlink
echo ""
echo "[2/3] Checking body_models symlink..."
if [ -L "./body_models" ] && [ -d "./body_models" ]; then
    TARGET=$(readlink -f ./body_models)
    NUM_FILES=$(ls -1 ./body_models/ | wc -l)
    echo "✓ Body models symlink OK"
    echo "  Target: $TARGET"
    echo "  Files: $NUM_FILES"

    # Verify MANO files
    if [ -f "./body_models/MANO_RIGHT.pkl" ]; then
        SIZE=$(stat -c%s ./body_models/MANO_RIGHT.pkl 2>/dev/null || stat -f%z ./body_models/MANO_RIGHT.pkl)
        echo "  MANO_RIGHT.pkl: $(echo "scale=2; $SIZE / 1024 / 1024" | bc) MB"
    fi
else
    echo "❌ Body models symlink missing or broken"
fi

# Check 3: Saved models symlink
echo ""
echo "[3/3] Checking saved_models symlink..."
if [ -L "./saved_models" ] && [ -d "./saved_models" ]; then
    TARGET=$(readlink -f ./saved_models)
    NUM_CKPTS=$(ls -d ./saved_models/*/ 2>/dev/null | wc -l)
    echo "✓ Saved models symlink OK"
    echo "  Target: $TARGET"
    echo "  Checkpoints: $NUM_CKPTS"

    # Verify critical checkpoint
    if [ -f "./saved_models/75268d864/checkpoints/last.ckpt" ]; then
        SIZE=$(stat -c%s ./saved_models/75268d864/checkpoints/last.ckpt 2>/dev/null || stat -f%z ./saved_models/75268d864/checkpoints/last.ckpt)
        echo "  75268d864 checkpoint: $(echo "scale=2; $SIZE / 1024 / 1024" | bc) MB ✓"
    else
        echo "  ❌ 75268d864 checkpoint not found"
    fi
else
    echo "❌ Saved models symlink missing or broken"
fi

echo ""
echo "======================================================================"
echo "Summary:"
ls -lh ./data ./body_models ./saved_models 2>/dev/null | grep '^l'
echo "======================================================================"