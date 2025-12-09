#!/bin/bash
# ================================================================
# STAGE 2 CONTINUATION: Resume SDS Training from Stage 2 Checkpoint (MC1)
# ================================================================

set -e

cd ~/Projects/holdse/code

echo "========================================================================"
echo "STAGE 2 CONTINUATION: Resume SDS Training - MC1 (HO3D)"
echo "========================================================================"

# ================================================================
# Configuration
# ================================================================
STAGE2_CKPT="$1"  # Pass checkpoint path as first argument
TARGET_EPOCHS="${2:-60}"  # Default to 60 epochs for Stage 2

# If no checkpoint provided, try to find the latest one
if [ -z "$STAGE2_CKPT" ]; then
    echo "No checkpoint specified, searching for latest..."
    STAGE2_CKPT=$(find logs -name "last.ckpt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

    if [ -z "$STAGE2_CKPT" ]; then
        echo "❌ ERROR: No checkpoint found"
        echo "Usage: $0 heckpoint_path> [target_epochs]"
        echo "Example: $0 logs/xyz123/checkpoints/last.ckpt 60"
        exit 1
    fi

    echo "Found latest checkpoint: $STAGE2_CKPT"
fi

CONFIG="./confs/stage2_hold_MC1_ho3d_sds_from_official.yaml"

echo ""
echo "Configuration:"
echo "  Checkpoint: $STAGE2_CKPT"
echo "  Config: $CONFIG"
echo "  Target epochs: $TARGET_EPOCHS"
echo "  Case: hold_MC1_ho3d"
echo "  Training: SDS-guided optimization (Stage 2)"
echo "========================================================================"
echo ""

# ================================================================
# Verification: Stage 2 Checkpoint
# ================================================================
if [ ! -f "$STAGE2_CKPT" ]; then
    echo "❌ ERROR: Checkpoint not found at $STAGE2_CKPT"
    echo "   Available checkpoints:"
    find logs -name "last.ckpt" -type f -printf '   %p\n' 2>/dev/null | head -10
    exit 1
fi

echo "✅ Stage 2 checkpoint verified: $STAGE2_CKPT"
echo ""

# Display checkpoint information and validate
python3 << EOF
import torch
import sys

try:
    ckpt = torch.load('$STAGE2_CKPT', map_location='cpu')

    current_epoch = ckpt.get('epoch', 'unknown')
    global_step = ckpt.get('global_step', 'unknown')

    print(f"Checkpoint Information:")
    print(f"   Current epoch: {current_epoch}")
    print(f"   Global step: {global_step}")
    print(f"   Has optimizer states: {'optimizer_states' in ckpt}")
    print(f"   Has LR scheduler: {'lr_schedulers' in ckpt}")

    # Validate epoch range
    if isinstance(current_epoch, int) and current_epoch >= $TARGET_EPOCHS:
        print(f"\\n⚠️  WARNING: Checkpoint epoch ({current_epoch}) >= target epochs ($TARGET_EPOCHS)")
        print(f"   Training will complete immediately.")
        print(f"   Increase target epochs (e.g., {current_epoch + 30}) to continue training.")
        sys.exit(1)
    else:
        print(f"\\n✓ Will resume from epoch {current_epoch + 1 if isinstance(current_epoch, int) else 'unknown'} to $TARGET_EPOCHS")

except Exception as e:
    print(f"❌ ERROR: Failed to load checkpoint: {e}")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Checkpoint validation failed"
    exit 1
fi

echo ""

# ================================================================
# Verification: Configuration File
# ================================================================
if [ ! -f "$CONFIG" ]; then
    echo "❌ ERROR: Config not found at $CONFIG"
    exit 1
fi

echo "✅ Configuration file verified: $CONFIG"
echo ""

# ================================================================
# Run Continuation Training
# ================================================================
echo "========================================================================"
echo "Starting Stage 2 continuation training..."
echo "========================================================================"
echo ""

python train.py \
    --config "$CONFIG" \
    --case "hold_MC1_ho3d" \
    --num_epoch $TARGET_EPOCHS \
    --infer_ckpt "$STAGE2_CKPT" \
    --no-comet \
    --gpu_id 0 \
    --no-pin-memory

# ================================================================
# Post-Training: Summary
# ================================================================
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✅ STAGE 2 CONTINUATION COMPLETE (MC1)!"
    echo "========================================================================"

    # Find most recent checkpoint
    NEW_CKPT=$(find logs -name "last.ckpt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

    if [ -n "$NEW_CKPT" ]; then
        echo "  New checkpoint: $NEW_CKPT"

        # Display final epoch info
        python3 << EOF
import torch
try:
    ckpt = torch.load('$NEW_CKPT', map_location='cpu')
    print(f"  Final epoch: {ckpt.get('epoch', 'unknown')}")
    print(f"  Global step: {ckpt.get('global_step', 'unknown')}")
except:
    pass
EOF
    fi

    echo ""
    echo "Next steps:"
    echo "  1. Render results: python scripts/render_validation_with_arg.py --checkpoint $NEW_CKPT"
    echo "  2. Continue training: $0 $NEW_CKPT [higher_epoch_count]"
    echo "  3. Run evaluation pipeline"
    echo "========================================================================"
else
    echo ""
    echo "❌ Stage 2 continuation training failed (MC1)"
    echo "   Check latest log: logs/stage2_*_hold_MC1_ho3d_*.log"
    exit 1
fi

# ================================================================
# USAGE GUIDE FOR STAGE 2 CONTINUATION
# ================================================================
#
# 1. Make it executable:
#    chmod +x scripts/train_MC1_stage2_continue.sh
#
# 2. Usage Options:
#
#    OPTION A: Specify checkpoint explicitly (recommended)
#    -----------------------------------------------------
#    ./scripts/train_MC1_stage2_continue.sh heckpoint_path> [target_epochs]
#
#    Example (continue from epoch 30 to epoch 60):
#    ./scripts/train_MC1_stage2_continue.sh \
#        logs/eb4395048/checkpoints/last.ckpt \
#        70 \
#        2>&1 | tee logs/stage2_30to70_hold_MC1_ho3d_from_official_$(date +%Y%m%d_%H%M%S).log
#
#    OPTION B: Auto-detect latest checkpoint
#    ----------------------------------------
#    ./scripts/train_MC1_stage2_continue.sh [target_epochs]
#
#    Example (continue latest to epoch 100):
#    ./scripts/train_MC1_stage2_continue.sh 100 2>&1 | tee logs/stage2_continuation_$(date +%Y%m%d_%H%M%S).log
#
# 3. Monitor Training:
#
#    # Watch epoch progress
#    tail -f logs/stage2_30to60_hold_MC1_ho3d_from_official_*.log | grep -E "Epoch.*Avg loss"
#
#    # Watch SDS guidance
#    tail -f logs/stage2_*.log | grep -i "sds\|guidance"
#
# 4. Common Scenarios:
#
#    Scenario 1: Stage 2 trained for 30 epochs, continue to 60
#    ----------------------------------------------------------
#    ./scripts/train_MC1_stage2_continue.sh logs/xyz/checkpoints/last.ckpt 60
#
#    Scenario 2: Continue from 60 to 100 epochs
#    -------------------------------------------
#    ./scripts/train_MC1_stage2_continue.sh logs/abc/checkpoints/last.ckpt 100
#
# 5. Key Differences from Stage 1:
#    - Stage 2 uses SDS guidance (slower per epoch)
#    - Typical Stage 2 total: 30-100 epochs
#    - Stage 2 refines geometry from Stage 1