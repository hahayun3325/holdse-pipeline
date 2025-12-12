#!/bin/bash

echo "=== HOLD Model Initialization (Stage 1 - No Phases) ==="
echo ""

echo "1. Looking for implicit network initialization:"
grep -n "implicit_network\|ImplicitNetwork" src/hold/hold.py | head -10

echo ""
echo "2. Looking for rendering network initialization:"
grep -n "rendering_network\|RenderingNetwork" src/hold/hold.py | head -10

echo ""
echo "3. Looking for MANO parameter setup:"
grep -n "hand_pose\|global_orient\|betas.*param\|transl.*param" src/hold/hold.py | head -15

echo ""
echo "4. Looking for node structure (hand, object, background):"
grep -n "self.model.nodes\|Node(" src/hold/hold.py | head -20

echo ""
echo "5. Checking phase initialization guards:"
grep -n "if.*phase3.*enabled\|if.*phase4.*enabled\|if.*phase5.*enabled" src/hold/hold.py | head -15
