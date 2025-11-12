#!/bin/bash

echo "=================================================================="
echo "PRETRAINED CHECKPOINT TESTING"
echo "=================================================================="

# Check if pretrained checkpoints exist
echo ""
echo "1. Checking for existing pretrained checkpoints..."

PRETRAINED_DIRS=(
    "../deployment"
    "./pretrained"
    "./checkpoints"
    "$HOME/Downloads"
)

FOUND_PRETRAINED=""

for dir in "${PRETRAINED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        CKPTS=$(find "$dir" -name "*.ckpt" -o -name "*.pt" -o -name "*.pth" 2>/dev/null)
        if [ -n "$CKPTS" ]; then
            echo "   ✅ Found in $dir:"
            echo "$CKPTS"
            FOUND_PRETRAINED="$CKPTS"
            break
        fi
    fi
done

if [ -z "$FOUND_PRETRAINED" ]; then
    echo "   ❌ No pretrained checkpoints found"
    echo ""
    echo "2. Where to get pretrained checkpoints:"
    echo "   - Original HOLD paper: https://github.com/zc-alexfan/hold"
    echo "   - Check paper supplementary materials"
    echo "   - Contact authors if publicly available"
    echo ""
    echo "   After downloading, place in:"
    echo "   - ./pretrained/"
    echo "   - ../deployment/"
    exit 1
fi

echo ""
echo "2. Testing pretrained checkpoint..."

PRETRAINED_CKPT=$(echo "$FOUND_PRETRAINED" | head -1)
echo "   Using: $PRETRAINED_CKPT"

# Test rendering with pretrained
python scripts/render_validation_with_arg.py --ckpt "$PRETRAINED_CKPT"

# Check result
TEST_IMG="logs/6aaaf5002/validation_render/rgb/00000.png"

if [ -f "$TEST_IMG" ]; then
    python -c "
import cv2
img = cv2.imread('$TEST_IMG')
print(f'   Result: mean={img.mean():.2f}, max={img.max()}')

if img.mean() > 1:
    print(f'   ✅ Pretrained checkpoint works!')
    print(f'   This confirms your training broke rendering')
else:
    print(f'   ❌ Pretrained also produces black images')
    print(f'   This suggests dataset/config issue')
"
else
    echo "   ❌ No output generated"
fi

echo ""
echo "=================================================================="

