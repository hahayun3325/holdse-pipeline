#!/bin/bash
# File: scripts/diagnose_epoch17_oom.sh

cd ~/Projects/holdse/code

echo "========================================================================="
echo "DIAGNOSTIC: Find which component causes epoch 17 OOM"
echo "========================================================================="
echo ""

# Test 1: Disable Phase 4
echo "Test 1: Disable Phase 4 contact refinement"
CONFIG="confs/ghop_diagnostic_no_phase4.yaml"
cp confs/ghop_production_32dim_latest.yaml "$CONFIG"
sed -i 's/contact_start_iter: [0-9]\+/contact_start_iter: 999999/' "$CONFIG"

echo "  Running without Phase 4..."
timeout 5m python train.py \
    --config "$CONFIG" \
    --case ghop_bottle_1 \
    --use_ghop \
    --gpu_id 0 \
    --num_epoch 20 \
    --no-pin-memory \
    --no-comet \
    2>&1 | tee test_no_phase4.log

if grep -q "Epoch 17.*100%" test_no_phase4.log; then
    echo "  ✅ Passed epoch 17 without Phase 4!"
    echo "  ⚠️  Phase 4 is the culprit"
    exit 0
else
    echo "  ❌ Still OOMed at epoch 17"
fi

# Test 2: Disable Phase 5
echo ""
echo "Test 2: Disable Phase 5 temporal consistency"
CONFIG="confs/ghop_diagnostic_no_phase5.yaml"
cp confs/ghop_production_32dim_latest.yaml "$CONFIG"
sed -i 's/phase5_start_iter: [0-9]\+/phase5_start_iter: 999999/' "$CONFIG"

echo "  Running without Phase 5..."
timeout 5m python train.py \
    --config "$CONFIG" \
    --case ghop_bottle_1 \
    --use_ghop \
    --gpu_id 0 \
    --num_epoch 20 \
    --no-pin-memory \
    --no-comet \
    2>&1 | tee test_no_phase5.log

if grep -q "Epoch 17.*100%" test_no_phase5.log; then
    echo "  ✅ Passed epoch 17 without Phase 5!"
    echo "  ⚠️  Phase 5 is the culprit"
    exit 0
else
    echo "  ❌ Still OOMed at epoch 17"
fi

# Test 3: Disable meshing
echo ""
echo "Test 3: Disable meshing operations"
python train.py \
    --config confs/ghop_production_32dim_latest.yaml \
    --case ghop_bottle_1 \
    --use_ghop \
    --gpu_id 0 \
    --num_epoch 20 \
    --no-pin-memory \
    --no-comet \
    --no-meshing \
    2>&1 | tee test_no_meshing.log

if grep -q "Epoch 17.*100%" test_no_meshing.log; then
    echo "  ✅ Passed epoch 17 without meshing!"
    echo "  ⚠️  Meshing is the culprit"
else
    echo "  ❌ Still OOMed at epoch 17"
    echo "  ⚠️  Issue is in base training loop"
fi

echo ""
echo "========================================================================="
echo "Check test_*.log files for details"
echo "========================================================================="
