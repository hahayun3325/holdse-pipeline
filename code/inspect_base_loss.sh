#!/bin/bash

echo "=== Base HOLD Loss Module (Stage 1) ==="
echo ""

echo "1. Main loss class:"
grep -n "class HOLDLoss\|class Loss" src/hold/loss.py | head -5

echo ""
echo "2. Loss computation methods:"
grep -n "def.*loss\|def compute" src/hold/loss.py | head -20

echo ""
echo "3. RGB loss implementation:"
grep -B 5 -A 15 "def.*rgb.*loss\|loss.*rgb" src/hold/loss.py | head -25

echo ""
echo "4. Mask loss implementation:"
grep -B 5 -A 15 "def.*mask.*loss\|loss.*mask" src/hold/loss.py | head -25

echo ""
echo "5. Eikonal loss implementation:"
grep -B 5 -A 15 "def.*eikonal\|loss.*eikonal" src/hold/loss.py | head -25

echo ""
echo "6. Check if Phase 3/4/5 losses are conditionally applied:"
grep -n "phase3\|phase4\|phase5\|ghop\|sds\|contact\|temporal" src/hold/loss.py | head -20
