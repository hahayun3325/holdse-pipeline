#!/bin/bash
# ================================================================
# STAGE 1 CONTINUATION: Resume Training from Checkpoint (MC1)
# ================================================================

set -e

cd ~/Projects/holdse/code

echo "========================================================================"
echo "STAGE 1 CONTINUATION: Resume Training - MC1 (HO3D)"
echo "========================================================================"

# ================================================================
# Configuration
# ================================================================
STAGE1_CKPT="$1"  # Pass checkpoint path as first argument
TARGET_EPOCHS="${2:-100}"  # Default to 100 epochs for MC1

# If no checkpoint provided, try to find the latest one
if [ -z "$STAGE1_CKPT" ]; then
    echo "No checkpoint specified, searching for latest..."
    STAGE1_CKPT=$(find logs -name "last.ckpt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

    if [ -z "$STAGE1_CKPT" ]; then
        echo "❌ ERROR: No checkpoint found"
        echo "Usage: $0 <checkpoint_path> [target_epochs]"
        echo "Example: $0 logs/a0419ab35/checkpoints/last.ckpt 20"
        exit 1
    fi

    echo "Found latest checkpoint: $STAGE1_CKPT"
fi

# ✅ KEY CHANGE 1: Config file for MC1
CONFIG="./confs/stage1_hold_MC1_ho3d_8layer_implicit_official_match_fixed.yaml"

echo ""
echo "Configuration:"
echo "  Checkpoint: $STAGE1_CKPT"
echo "  Config: $CONFIG"
echo "  Target epochs: $TARGET_EPOCHS"
echo "  Case: hold_MC1_ho3d"  # ✅ Changed from hold_ABF12_ho3d
echo "  Dataset: HO3D MC1 (144 frames)"  # ✅ Added dataset info
echo "========================================================================"
echo ""

# ================================================================
# Verification: Stage 1 Checkpoint
# ================================================================
if [ ! -f "$STAGE1_CKPT" ]; then
    echo "❌ ERROR: Checkpoint not found at $STAGE1_CKPT"
    echo "   Available checkpoints:"
    find logs -name "last.ckpt" -type f -printf '   %p\n' 2>/dev/null | head -10
    exit 1
fi

echo "✅ Stage 1 checkpoint verified: $STAGE1_CKPT"
echo ""

# Display checkpoint information
python3 << EOF
import torch
import sys

try:
    ckpt = torch.load('$STAGE1_CKPT', map_location='cpu')

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
        print(f"   Increase target epochs (e.g., {current_epoch + 100}) to continue training.")
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
    echo "   Creating from ghop_stage1_rgb_only.yaml..."
    cp ./confs/ghop_stage1_rgb_only.yaml "$CONFIG"
    echo "   ✓ Config created"
fi

echo "✅ Configuration file verified: $CONFIG"
echo ""

# ================================================================
# Run Continuation Training
# ================================================================
echo "========================================================================"
echo "Starting Stage 1 continuation training..."
echo "========================================================================"
echo ""

# ✅ KEY CHANGE 2: Case name for MC1
python train.py \
    --config "$CONFIG" \
    --case "hold_MC1_ho3d" \
    --num_epoch $TARGET_EPOCHS \
    --infer_ckpt "$STAGE1_CKPT" \
    --no-comet \
    --gpu_id 0 \
    --no-pin-memory

# ================================================================
# Post-Training: Summary
# ================================================================
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✅ STAGE 1 CONTINUATION COMPLETE (MC1)!"
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
    echo "  1. Render validation: python scripts/render_validation_with_arg.py --checkpoint $NEW_CKPT"
    echo "  2. Continue training: $0 $NEW_CKPT [higher_epoch_count]"
    echo "  3. Proceed to Stage 2 with this checkpoint"
    echo "========================================================================"
else
    echo ""
    echo "❌ Stage 1 continuation training failed (MC1)"
    echo "   Check latest log: logs/MC1_stage1_continuation_*.log"
    exit 1
fi

#USAGE GUIDE FOR STAGE 1 CONTINUATION
#=====================================
#2. Make it executable:
#   chmod +x scripts/train_MC1_stage1_continue.sh
#
#3. Usage Options:
#
#   OPTION A: Auto-detect latest checkpoint
#   ----------------------------------------
#   ./scripts/train_MC1_stage1_continue.sh [target_epochs]
#
#   Example (continue to epoch 100):
#   ./scripts/train_MC1_stage1_continue.sh 100 2>&1 | tee logs/stage1_20to100_hold_MC1_ho3d_official_match_fixed_mano_$(date +%Y%m%d_%H%M%S).log
#./scripts/train_MC1_stage1_continue.sh \
#    logs/a0419ab35/checkpoints/last.ckpt \
#    100 \
#    2>&1 | tee logs/stage1_20to100_hold_MC1_ho3d_official_match_fixed_mano_$(date +%Y%m%d_%H%M%S).log

#   OPTION B: Specify checkpoint explicitly
#   ----------------------------------------
#   ./scripts/train_MC1_stage1_continue.sh <checkpoint_path> [target_epochs]
#
#   Example (continue specific checkpoint to epoch 100):
#   ./scripts/train_MC1_stage1_continue.sh logs/a0419ab35/checkpoints/last.ckpt 100
#
#4. Monitor Training:
#
#   # Watch loss values
#   tail -f logs/stage1_10to100_hold_MC1_ho3d_8implicit_mano_*.log | grep "Avg loss"
#
#   # Watch epoch progress
#   tail -f logs/stage1_20to100_hold_MC1_ho3d_official_match_fixed_mano_*.log | grep -E "Epoch.*Avg loss"
#
#   # Monitor GPU usage
#   watch -n 5 nvidia-smi
#
#5. Common Scenarios:
#
#   Scenario 1: Your Stage 1 trained for 10 epochs, continue to 100
#   ----------------------------------------------------------------
#   ./scripts/train_MC1_stage1_continue.sh logs/xyz/checkpoints/last.ckpt 100
#
#   Scenario 2: Continue from 100 to 100 epochs
#   --------------------------------------------
#   ./scripts/train_MC1_stage1_continue.sh logs/abc/checkpoints/last.ckpt 100
#
#   Scenario 3: Auto-find latest and train to 150
#   ----------------------------------------------
#   ./scripts/train_MC1_stage1_continue.sh 150
#
#6. Verification:
#
#   Before running, check what will happen:
#
#   python3 << 'EOF'
#   import torch
#   ckpt = torch.load('YOUR_CHECKPOINT_PATH', map_location='cpu')
#   print(f"Current epoch: {ckpt.get('epoch')}")
#   print(f"Target epoch: YOUR_TARGET")
#   print(f"Will train: {YOUR_TARGET - ckpt.get('epoch')} more epochs")
#   EOF
#
#COMPARISON WITH STAGE 2 SCRIPT
#===============================
#
#Stage 1 Script:
#- Config: confs/stage1_hold_MC1_ho3d_8layer_implicit.yaml
#- Case: hold_MC1_ho3d
#- Dataset: HO3D (RGB only, no GHOP)
#- Typical epochs: 80-120
#
#Stage 2 Script:
#- Config: confs/ghop_stage2_hold_MC1_ho3d.yaml
#- Case: hold_MC1_ho3d
#- Dataset: HO3D (RGB only, no GHOP)
#- Typical epochs: 20-30
