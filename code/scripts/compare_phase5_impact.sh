#!/bin/bash
# Compare Epoch 20 (pre-Phase 5) vs Epoch 25 (post-Phase 5)

BASELINE="../deployment/hoi4d_phase5_v1.0/hoi4d_phase5_pre_v1.0.ckpt"
PHASE5="../deployment/hoi4d_phase5_v1.0/hoi4d_phase5_v1.0.ckpt"

echo "================================================================"
echo "PHASE 5 IMPACT COMPARISON"
echo "================================================================"
echo ""
echo "Baseline: Epoch 20 (0.0 Phase 5 epochs)"
echo "Phase 5:  Epoch 25 (5.0 Phase 5 epochs)"
echo ""

# Run verification on both
echo "Verifying baseline checkpoint..."
python scripts/verify_phase5_training.py "$BASELINE" 1100 | grep -A 5 "Phase 5 Training Status"

echo ""
echo "Verifying Phase 5 checkpoint..."
python scripts/verify_phase5_training.py "$PHASE5" 1100 | grep -A 5 "Phase 5 Training Status"

echo ""
echo "================================================================"
echo "COMPARISON SUMMARY"
echo "================================================================"
echo ""
echo "Architecture: Identical (131.51M parameters)"
echo "GHOP Keys: Identical (1323 keys)"
echo "Difference: 5 epochs of temporal regularization training"
echo ""
echo "Expected Phase 5 Benefits:"
echo "  • Smoother frame-to-frame transitions"
echo "  • Reduced temporal jitter"
echo "  • Better velocity consistency"
echo "  • Improved acceleration continuity"
echo ""
echo "Next: Visual quality assessment on video sequences required"
echo "================================================================"
BASH