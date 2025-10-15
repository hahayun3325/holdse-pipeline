#!/bin/bash
# File: ~/Projects/holdse/code/setup_saved_models_symlink.sh

set -e

echo "======================================================================"
echo "HOLDSE: Create Symlink to HOLD saved_models"
echo "======================================================================"

cd ~/Projects/holdse/code

# Define paths
HOLD_SAVED_MODELS=~/Projects/hold/code/saved_models
HOLDSE_SAVED_MODELS=./saved_models

echo "[1/5] Checking HOLD saved_models directory..."
if [ ! -d "$HOLD_SAVED_MODELS" ]; then
    echo "❌ ERROR: HOLD saved_models not found at $HOLD_SAVED_MODELS"
    exit 1
fi

echo "✓ Found HOLD saved_models"

# Count checkpoints
NUM_CHECKPOINTS=$(ls -d "$HOLD_SAVED_MODELS"/*/ 2>/dev/null | wc -l)
echo "  Available checkpoints: $NUM_CHECKPOINTS"

# Verify critical checkpoint exists
TARGET_CKPT="$HOLD_SAVED_MODELS/75268d864/checkpoints/last.ckpt"
if [ ! -f "$TARGET_CKPT" ]; then
    echo "❌ ERROR: Required checkpoint not found: $TARGET_CKPT"
    exit 1
fi

CKPT_SIZE=$(stat -c%s "$TARGET_CKPT" 2>/dev/null || stat -f%z "$TARGET_CKPT")
echo "✓ Target checkpoint verified: $(echo "scale=2; $CKPT_SIZE / 1024 / 1024" | bc) MB"

echo ""
echo "[2/5] Handling existing saved_models..."
if [ -L "$HOLDSE_SAVED_MODELS" ]; then
    echo "⚠️  Symlink already exists, removing..."
    rm "$HOLDSE_SAVED_MODELS"
elif [ -d "$HOLDSE_SAVED_MODELS" ]; then
    echo "⚠️  Directory exists, backing up..."
    BACKUP_DIR="saved_models.backup_$(date +%Y%m%d_%H%M%S)"
    mv "$HOLDSE_SAVED_MODELS" "$BACKUP_DIR"
    echo "  Backed up to: $BACKUP_DIR"
fi

echo ""
echo "[3/5] Creating symlink..."
ln -s "$HOLD_SAVED_MODELS" "$HOLDSE_SAVED_MODELS"
echo "✓ Created symlink:"
echo "  $HOLDSE_SAVED_MODELS -> $HOLD_SAVED_MODELS"

echo ""
echo "[4/5] Verifying symlink..."
if [ -L "$HOLDSE_SAVED_MODELS" ] && [ -d "$HOLDSE_SAVED_MODELS" ]; then
    echo "✓ Symlink verification successful"

    # Verify checkpoint accessible through symlink
    if [ -f "$HOLDSE_SAVED_MODELS/75268d864/checkpoints/last.ckpt" ]; then
        SIZE=$(stat -c%s "$HOLDSE_SAVED_MODELS/75268d864/checkpoints/last.ckpt" 2>/dev/null || stat -f%z "$HOLDSE_SAVED_MODELS/75268d864/checkpoints/last.ckpt")
        echo "✓ Checkpoint accessible through symlink"
        echo "  Size: $(echo "scale=2; $SIZE / 1024 / 1024" | bc) MB"
    else
        echo "❌ ERROR: Checkpoint not accessible through symlink"
        exit 1
    fi
else
    echo "❌ ERROR: Symlink creation failed"
    exit 1
fi

echo ""
echo "[5/5] Testing PyTorch loading..."
python -c "
import torch
ckpt_path = 'saved_models/75268d864/checkpoints/last.ckpt'
ckpt = torch.load(ckpt_path, map_location='cpu')
print('✓ PyTorch can load checkpoint through symlink')
print(f'  Keys: {list(ckpt.keys())}')
print(f'  Epoch: {ckpt.get(\"epoch\", \"N/A\")}')
print(f'  Global step: {ckpt.get(\"global_step\", \"N/A\")}')
"

echo ""
echo "======================================================================"
echo "✓✓✓ Symlink Setup Complete ✓✓✓"
echo "======================================================================"
echo ""
echo "Available checkpoints in saved_models:"
ls -1 "$HOLDSE_SAVED_MODELS"
echo ""
echo "You can now run sanity training:"
echo "  python sanity_train.py --case hold_mug1_itw --shape_init 75268d864 --gpu_id 0"
echo "======================================================================"