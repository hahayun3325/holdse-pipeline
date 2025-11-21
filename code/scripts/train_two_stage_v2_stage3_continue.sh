# Create Stage 3 continuation training script

stage3_continue_script = '''#!/bin/bash
# ================================================================
# STAGE 3 CONTINUATION: Resume Full Pipeline Training
# ================================================================

set -e

cd ~/Projects/holdse/code

echo "========================================================================"
echo "STAGE 3 CONTINUATION: Resume Full Pipeline Training"
echo "========================================================================"

# ================================================================
# Configuration
# ================================================================
STAGE3_CKPT="$1"  # Pass checkpoint path as first argument
TARGET_EPOCHS="${2:-40}"  # Default to 40 epochs if not specified

# If no checkpoint provided, try to find the latest one
if [ -z "$STAGE3_CKPT" ]; then
    echo "No checkpoint specified, searching for latest..."
    STAGE3_CKPT=$(find logs -name "last.ckpt" -type f -printf '%T@ %p\\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

    if [ -z "$STAGE3_CKPT" ]; then
        echo "❌ ERROR: No checkpoint found"
        echo "Usage: $0 <checkpoint_path> [target_epochs]"
        echo "Example: $0 logs/abc123/checkpoints/last.ckpt 40"
        exit 1
    fi

    echo "Found latest checkpoint: $STAGE3_CKPT"
fi

CONFIG="confs/ghop_stage3_full_pipeline.yaml"

echo ""
echo "Configuration:"
echo "  Checkpoint: $STAGE3_CKPT"
echo "  Config: $CONFIG"
echo "  Target epochs: $TARGET_EPOCHS"
echo "  Case: hold_bottle1_itw"
echo ""
echo "Phases Enabled:"
echo "  ✅ Phase 3: GHOP SDS (Object Shape)"
echo "  ✅ Phase 4: Contact Refinement"
echo "  ✅ Phase 5: Temporal Consistency"
echo "========================================================================"
echo ""

# ================================================================
# Verification: Stage 3 Checkpoint
# ================================================================
if [ ! -f "$STAGE3_CKPT" ]; then
    echo "❌ ERROR: Checkpoint not found at $STAGE3_CKPT"
    echo "   Available checkpoints:"
    find logs -name "last.ckpt" -type f -printf '   %p\\n' 2>/dev/null | head -10
    exit 1
fi

echo "✅ Stage 3 checkpoint verified: $STAGE3_CKPT"
echo ""

# Display checkpoint information
python3 << EOF
import torch
import sys

try:
    ckpt = torch.load('$STAGE3_CKPT', map_location='cpu')

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
        print(f"   Increase target epochs (e.g., {current_epoch + 10}) to continue training.")
        sys.exit(1)
    else:
        print(f"\\n✓ Will resume from epoch {current_epoch + 1 if isinstance(current_epoch, int) else 'unknown'} to $TARGET_EPOCHS")

    # Show phase schedule info based on global step
    if isinstance(global_step, int):
        print(f"\\nPhase Schedule (based on global_step={global_step}):")
        print(f"   Phase 3 (GHOP SDS): Active (starts at step 0)")

        if global_step >= 1000:
            print(f"   Phase 4 (Contact): Active (starts at step 1000)")
        else:
            remaining = 1000 - global_step
            print(f"   Phase 4 (Contact): Will activate in {remaining} steps")

        if global_step >= 2000:
            print(f"   Phase 5 (Temporal): Active (starts at step 2000)")
        else:
            remaining = 2000 - global_step
            print(f"   Phase 5 (Temporal): Will activate in {remaining} steps")

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
# Verification: GHOP Checkpoint
# ================================================================
if [ ! -f "checkpoints/ghop/last.ckpt" ]; then
    echo "⚠️  WARNING: GHOP checkpoint not found at checkpoints/ghop/last.ckpt"
    echo "   Phase 3 may use random initialization"
    echo ""
else
    echo "✅ GHOP checkpoint verified: checkpoints/ghop/last.ckpt"
    echo ""
fi

# ================================================================
# Run Continuation Training
# ================================================================
echo "========================================================================"
echo "Starting Stage 3 continuation training..."
echo "========================================================================"
echo ""

python train.py \\
    --config "$CONFIG" \\
    --case "hold_bottle1_itw" \\
    --num_epoch $TARGET_EPOCHS \\
    --infer_ckpt "$STAGE3_CKPT" \\
    --no-comet \\
    --gpu_id 0 \\
    --no-pin-memory \\
    2>&1 | tee logs/stage3_continuation_$(date +%Y%m%d_%H%M%S).log

# ================================================================
# Post-Training: Summary
# ================================================================
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✅ STAGE 3 CONTINUATION COMPLETE!"
    echo "========================================================================"

    # Find most recent checkpoint
    NEW_CKPT=$(find logs -name "last.ckpt" -type f -printf '%T@ %p\\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

    if [ -n "$NEW_CKPT" ]; then
        echo "  New checkpoint: $NEW_CKPT"

        # Display final state
        python3 << EOF
import torch
try:
    ckpt = torch.load('$NEW_CKPT', map_location='cpu')
    final_epoch = ckpt.get('epoch', 'unknown')
    final_step = ckpt.get('global_step', 'unknown')
    print(f"  Final epoch: {final_epoch}")
    print(f"  Final global step: {final_step}")

    # Show which phases are/were active
    if isinstance(final_step, int):
        print(f"\\n  Active phases at completion:")
        print(f"    Phase 3 (GHOP SDS): ✅")
        if final_step >= 1000:
            print(f"    Phase 4 (Contact): ✅")
        else:
            print(f"    Phase 4 (Contact): ❌ (needs step >= 1000)")
        if final_step >= 2000:
            print(f"    Phase 5 (Temporal): ✅")
        else:
            print(f"    Phase 5 (Temporal): ❌ (needs step >= 2000)")
except:
    pass
EOF

        # Offer to save as stage3_final.ckpt
        echo ""
        read -p "Save as logs/stage3_final.ckpt? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            cp "$NEW_CKPT" logs/stage3_final.ckpt
            echo "✅ Saved to logs/stage3_final.ckpt"
        fi
    fi

    echo ""
    echo "Next steps:"
    echo "  1. Render validation:"
    echo "     python scripts/render_validation_with_arg.py --checkpoint $NEW_CKPT"
    echo ""
    echo "  2. Continue training further:"
    echo "     $0 $NEW_CKPT [higher_epoch_count]"
    echo ""
    echo "  3. Extract final meshes:"
    echo "     python scripts/extract_meshes.py --checkpoint $NEW_CKPT"
    echo ""
    echo "  4. Compare stages:"
    echo "     - Stage 1: Basic RGB reconstruction"
    echo "     - Stage 2: + GHOP SDS guidance"
    echo "     - Stage 3: + Contact + Temporal consistency (FINAL)"
    echo ""
    echo "========================================================================"
else
    echo ""
    echo "========================================================================"
    echo "❌ STAGE 3 CONTINUATION FAILED"
    echo "========================================================================"
    echo "   Check logs/stage3_continuation_*.log for details"
    echo ""
    echo "Common issues:"
    echo "  1. Dimension mismatch with checkpoint"
    echo "  2. Phase 4 contact instability"
    echo "  3. Phase 5 temporal consistency divergence"
    echo "  4. Out of memory (reduce batch size)"
    echo ""
    exit 1
fi
'''

# Write the script
with open('/tmp/train_stage3_continue.sh', 'w') as f:
    f.write(stage3_continue_script)

print("Stage 3 Continuation Script Created!")
print("="*70)
print("\n" + stage3_continue_script)
print("\n" + "="*70)

# Create usage guide
usage_guide = """
USAGE GUIDE FOR STAGE 3 CONTINUATION
=====================================

Stage 3 is the most complex stage with 3 active phases that have different
activation schedules. This script handles resume correctly while maintaining
phase scheduling.

1. Save the script:
   --------------------------------------------------------
   Copy to: ~/Projects/holdse/code/scripts/train_two_stage_v2_stage3_continue.sh

2. Make it executable:
   --------------------------------------------------------
   chmod +x scripts/train_two_stage_v2_stage3_continue.sh

3. Usage Options:

   OPTION A: Auto-detect latest checkpoint
   ----------------------------------------
   ./scripts/train_two_stage_v2_stage3_continue.sh [target_epochs]

   Example (continue to epoch 40):
   ./scripts/train_two_stage_v2_stage3_continue.sh 40

   OPTION B: Specify checkpoint explicitly
   ----------------------------------------
   ./scripts/train_two_stage_v2_stage3_continue.sh <checkpoint_path> [target_epochs]

   Example:
   ./scripts/train_two_stage_v2_stage3_continue.sh logs/abc123/checkpoints/last.ckpt 40

4. Understanding Phase Scheduling:
   --------------------------------------------------------
   Stage 3 has 3 phases that activate at different global steps:

   - Phase 3 (GHOP SDS):      Active from step 0 (always on)
   - Phase 4 (Contact):       Active from step 1000+
   - Phase 5 (Temporal):      Active from step 2000+

   The script will show which phases are active based on global_step.

5. Common Scenarios:

   Scenario 1: Continue Stage 3 from 30 to 40 epochs
   --------------------------------------------------
   # Assuming Stage 3 started at epoch 0
   ./scripts/train_two_stage_v2_stage3_continue.sh logs/xyz/checkpoints/last.ckpt 40

   Scenario 2: Small test (1 epoch continuation)
   ----------------------------------------------
   # Test resume works before long training
   ./scripts/train_two_stage_v2_stage3_continue.sh logs/xyz/checkpoints/last.ckpt 31

   Scenario 3: Auto-detect and train to 50 epochs
   -----------------------------------------------
   ./scripts/train_two_stage_v2_stage3_continue.sh 50

6. Monitoring Training:
   --------------------------------------------------------
   # Watch loss values
   tail -f logs/stage3_continuation_*.log | grep "Avg loss"

   # Watch phase activation
   tail -f logs/stage3_continuation_*.log | grep -E "Phase [345]"

   # Watch resume message
   tail -f logs/stage3_continuation_*.log | grep "RESUME TRAINING"

   # Monitor GPU
   watch -n 5 nvidia-smi

7. Phase Schedule Verification:
   --------------------------------------------------------
   Check which phases should be active:

   python3 << 'EOF'
   import torch
   ckpt = torch.load('YOUR_CHECKPOINT', map_location='cpu')
   step = ckpt.get('global_step', 0)
   print(f"Global step: {step}")
   print(f"Phase 3 (GHOP): {'Active' if step >= 0 else 'Inactive'}")
   print(f"Phase 4 (Contact): {'Active' if step >= 1000 else 'Inactive'}")
   print(f"Phase 5 (Temporal): {'Active' if step >= 2000 else 'Inactive'}")
   EOF

8. Expected Output Pattern:
   --------------------------------------------------------
   For a checkpoint at epoch 30, global_step 60000:

   RESUME TRAINING: Restoring full training state
   ======================================================================
     Checkpoint epoch: 30
     Global step: 60000
     Contains optimizer: True
     Contains LR scheduler: True

     ✓ Will resume from epoch 31 to 40

   Phase Schedule (based on global_step=60000):
     Phase 3 (GHOP SDS): Active (starts at step 0)
     Phase 4 (Contact): Active (starts at step 1000)
     Phase 5 (Temporal): Active (starts at step 2000)

   [Epoch 30] Avg loss: 0.1xxxx  # Should continue from previous loss

COMPARISON: STAGE 1 vs 2 vs 3 CONTINUATION
===========================================

| Feature          | Stage 1              | Stage 2              | Stage 3              |
|------------------|----------------------|----------------------|----------------------|
| Config           | stage1_rgb_only      | stage2_temporal_only | stage3_full_pipeline |
| Case             | hold_ABF12_ho3d      | hold_bottle1_itw     | hold_bottle1_itw     |
| Phases           | None                 | Phase 3 only         | Phase 3+4+5          |
| Typical epochs   | 80-120               | 20-30                | 30-50                |
| Complexity       | Low (RGB only)       | Medium (+GHOP SDS)   | High (All phases)    |
| Phase scheduling | N/A                  | N/A                  | Yes (step-based)     |
| Resume param     | --infer_ckpt         | --infer_ckpt         | --infer_ckpt         |

TROUBLESHOOTING
===============

Issue 1: "Phase schedule mismatch"
----------------------------------
Problem: Phases not activating as expected
Solution: Check global_step in checkpoint matches expected iteration count

Issue 2: "Training completes immediately"
-----------------------------------------
Problem: Target epochs <= checkpoint epoch
Solution: Increase target epochs beyond current checkpoint epoch

Issue 3: "Contact loss diverges"
--------------------------------
Problem: Phase 4 instability after resume
Solution: May need to reduce w_penetration or w_contact in config

Issue 4: "Temporal loss NaN"
-----------------------------
Problem: Phase 5 encounters invalid temporal pairs
Solution: Check dataset has temporal pairs, or disable Phase 5 temporarily
"""

print(usage_guide)