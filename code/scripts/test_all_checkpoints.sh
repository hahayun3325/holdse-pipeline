#!/bin/bash
# Test all available checkpoints to find any that can render

echo "=================================================================="
echo "CHECKPOINT RENDERING TEST"
echo "=================================================================="
echo "Testing all available checkpoints to find working ones"
echo ""

OUTPUT_SUMMARY="checkpoint_test_results.txt"
echo "Checkpoint Testing Results - $(date)" > $OUTPUT_SUMMARY
echo "==================================================================" >> $OUTPUT_SUMMARY

# Function to test a single checkpoint
test_checkpoint() {
    local ckpt_path=$1
    local ckpt_name=$(basename $(dirname $(dirname $ckpt_path)))/$(basename $ckpt_path)
    
    echo ""
    echo "Testing: $ckpt_name"
    echo "  Path: $ckpt_path"
    
    # Check if checkpoint exists and is valid
    if [ ! -f "$ckpt_path" ]; then
        echo "  ❌ File not found"
        echo "$ckpt_name: FILE_NOT_FOUND" >> $OUTPUT_SUMMARY
        return
    fi
    
    local size_mb=$(du -m "$ckpt_path" | cut -f1)
    echo "  Size: ${size_mb}MB"
    
    if [ $size_mb -lt 10 ]; then
        echo "  ❌ Too small (likely corrupted)"
        echo "$ckpt_name: CORRUPTED (${size_mb}MB)" >> $OUTPUT_SUMMARY
        return
    fi
    
    # Quick test: Check for NaN in checkpoint weights
    echo "  Checking for NaN in weights..."
    python -c "
import torch
import sys

ckpt = torch.load('$ckpt_path', map_location='cpu')
sd = ckpt['state_dict']

nan_params = [k for k, v in sd.items() if torch.isnan(v).any()]
inf_params = [k for k, v in sd.items() if torch.isinf(v).any()]

if len(nan_params) > 0 or len(inf_params) > 0:
    print(f'  ❌ Checkpoint has NaN/Inf: {len(nan_params)} NaN, {len(inf_params)} Inf')
    sys.exit(1)
else:
    print(f'  ✅ No NaN/Inf in weights')
    sys.exit(0)
"
    
    if [ $? -ne 0 ]; then
        echo "$ckpt_name: NAN_IN_WEIGHTS" >> $OUTPUT_SUMMARY
        return
    fi
    
    # Run actual rendering test
    echo "  Running render test..."
    
    python scripts/render_validation.py \
        --ckpt "$ckpt_path" \
        > "test_${ckpt_name//\//_}.log" 2>&1
    
    # Check if output has content
    local test_img="logs/6aaaf5002/validation_render/rgb/00000.png"
    
    if [ -f "$test_img" ]; then
        local img_mean=$(python -c "
import cv2
import numpy as np
img = cv2.imread('$test_img')
print(img.mean())
" 2>/dev/null)
        
        if [ -n "$img_mean" ]; then
            echo "  Image mean: $img_mean"
            
            if (( $(echo "$img_mean > 1.0" | bc -l) )); then
                echo "  ✅ SUCCESS! This checkpoint can render!"
                echo "$ckpt_name: SUCCESS (mean=$img_mean)" >> $OUTPUT_SUMMARY
                return 0
            else
                echo "  ❌ Black image (mean=$img_mean)"
                echo "$ckpt_name: BLACK_IMAGE (mean=$img_mean)" >> $OUTPUT_SUMMARY
            fi
        fi
    else
        echo "  ❌ No output generated"
        echo "$ckpt_name: NO_OUTPUT" >> $OUTPUT_SUMMARY
    fi
}

# Find all checkpoints
echo "Searching for checkpoints..."
CHECKPOINTS=$(find logs -name "*.ckpt" -type f | sort)

if [ -z "$CHECKPOINTS" ]; then
    echo "❌ No checkpoints found in logs/"
    exit 1
fi

TOTAL=$(echo "$CHECKPOINTS" | wc -l)
echo "Found $TOTAL checkpoints"
echo ""

# Test each checkpoint
COUNT=0
for ckpt in $CHECKPOINTS; do
    COUNT=$((COUNT+1))
    echo "[$COUNT/$TOTAL] Testing: $ckpt"
    test_checkpoint "$ckpt"
done

echo ""
echo "=================================================================="
echo "TESTING COMPLETE"
echo "=================================================================="
echo "Results summary: $OUTPUT_SUMMARY"
echo ""

# Show summary
echo "Summary:"
grep "SUCCESS" $OUTPUT_SUMMARY || echo "  No working checkpoints found"
grep "BLACK_IMAGE" $OUTPUT_SUMMARY | wc -l | xargs echo "  Black images:"
grep "NAN_IN_WEIGHTS" $OUTPUT_SUMMARY | wc -l | xargs echo "  Corrupted:"

