#!/bin/bash

echo "=================================================================="
echo "CHECKING TRAINING VALIDATION OUTPUTS"
echo "=================================================================="

# Check for validation images from training
echo ""
echo "1. Searching for validation images from training..."

VAL_IMGS=$(find logs -name "*.png" -o -name "*.jpg" | grep -E "val|validation|test" | head -20)

if [ -z "$VAL_IMGS" ]; then
    echo "   ❌ No validation images found"
    echo "   Training may not have saved visualizations"
else
    echo "   ✅ Found validation images:"
    echo "$VAL_IMGS" | head -10
    echo ""
    
    # Check first validation image
    FIRST_IMG=$(echo "$VAL_IMGS" | head -1)
    echo "   Checking first validation image: $FIRST_IMG"
    
    python -c "
import cv2
import numpy as np

img = cv2.imread('$FIRST_IMG')
if img is not None:
    print(f'     Size: {img.shape}')
    print(f'     Mean: {img.mean():.2f}')
    print(f'     Max: {img.max()}')
    
    if img.mean() > 1:
        print(f'     ✅ Has content - training DID produce visible images!')
    else:
        print(f'     ❌ Black - training never produced visible images')
else:
    print(f'     ❌ Failed to load image')
"
fi

# Check TensorBoard logs
echo ""
echo "2. Checking for TensorBoard logs..."

TB_LOGS=$(find logs -name "events.out.tfevents.*" | head -5)

if [ -z "$TB_LOGS" ]; then
    echo "   ❌ No TensorBoard logs found"
else
    echo "   ✅ Found TensorBoard logs:"
    echo "$TB_LOGS" | head -3
    echo ""
    echo "   To view training images:"
    echo "   tensorboard --logdir=logs --port=6006"
fi

# Check for checkpoint metadata
echo ""
echo "3. Checking checkpoint metadata..."

for ckpt in logs/*/checkpoints/last.ckpt logs/*/checkpoints/epoch=*.ckpt; do
    if [ -f "$ckpt" ]; then
        echo "   Checkpoint: $ckpt"
        
        python -c "
import torch

ckpt = torch.load('$ckpt', map_location='cpu')

if 'epoch' in ckpt:
    print(f'     Epoch: {ckpt[\"epoch\"]}')
if 'global_step' in ckpt:
    print(f'     Global step: {ckpt[\"global_step\"]}')

# Check for stored losses
if 'val_loss' in ckpt:
    print(f'     Val loss: {ckpt[\"val_loss\"]:.6f}')
if 'train_loss' in ckpt:
    print(f'     Train loss: {ckpt[\"train_loss\"]:.6f}')
" 2>/dev/null
        
        break  # Only check first checkpoint
    fi
done

echo ""
echo "=================================================================="

