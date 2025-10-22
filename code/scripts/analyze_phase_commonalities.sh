#!/bin/bash
# File: scripts/analyze_phase_commonalities.sh
# Purpose: Find shared code/dependencies between Phase 4 and Phase 5
# FIXED: Proper regex escaping for bash

cd ~/Projects/holdse/code

echo "========================================================================="
echo "PHASE 4 & 5 COMMONALITY ANALYSIS"
echo "========================================================================="
echo ""

# ================================================================
# Step 1: Find Phase 4 and Phase 5 implementation files
# ================================================================
echo "Step 1: Locating implementation files..."
echo ""

# Phase 4 files
echo "Phase 4 (Contact Refinement) files:"
find ./src -name "*contact*" -o -name "*phase4*" -o -name "*mesh*" 2>/dev/null | grep -v __pycache__
echo ""

# Phase 5 files
echo "Phase 5 (Temporal Consistency) files:"
find ./src -name "*temporal*" -o -name "*phase5*" 2>/dev/null | grep -v __pycache__
echo ""

# ================================================================
# Step 2: Analyze imports in both phases
# ================================================================
echo "========================================================================="
echo "Step 2: Import Analysis"
echo "========================================================================="
echo ""

# Extract Phase 4 imports
echo "Phase 4 imports:"
if [ -f "src/hold/hold.py" ]; then
    # Find Phase 4 section
    awk '/# PHASE 4/,/# PHASE 5/ {print}' src/hold/hold.py | grep -E "^import |^from " | sort -u
fi
echo ""

# Extract Phase 5 imports
echo "Phase 5 imports:"
if [ -f "src/hold/hold.py" ]; then
    # Find Phase 5 section
    awk '/# PHASE 5/,/# END/ {print}' src/hold/hold.py | grep -E "^import |^from " | sort -u
fi
echo ""

# ================================================================
# Step 3: Find shared library dependencies
# ================================================================
echo "========================================================================="
echo "Step 3: Shared Dependencies"
echo "========================================================================="
echo ""

echo "Checking for PyTorch3D usage:"
grep -r "pytorch3d" src/ --include="*.py" 2>/dev/null | grep -v __pycache__ | head -10
echo ""

echo "Checking for mesh extraction:"
grep -r "marching_cubes" src/ --include="*.py" 2>/dev/null | grep -v __pycache__ | head -10
grep -r "mcubes" src/ --include="*.py" 2>/dev/null | grep -v __pycache__ | head -10
echo ""

echo "Checking for SDF operations:"
grep -r "sdf_grid\|sample_sdf" src/ --include="*.py" 2>/dev/null | grep -v __pycache__ | head -10
echo ""

# ================================================================
# Step 4: Analyze initialization patterns
# ================================================================
echo "========================================================================="
echo "Step 4: Initialization Patterns"
echo "========================================================================="
echo ""

echo "Phase 4 initialization:"
grep -A 20 "def.*phase.*4.*init" src/hold/hold.py 2>/dev/null | head -30
grep -A 20 "contact.*init" src/hold/hold.py 2>/dev/null | head -30
echo ""

echo "Phase 5 initialization:"
grep -A 20 "def.*phase.*5.*init" src/hold/hold.py 2>/dev/null | head -30
grep -A 20 "temporal.*init" src/hold/hold.py 2>/dev/null | head -30
echo ""

# ================================================================
# Step 5: Find large allocations (FIXED REGEX)
# ================================================================
echo "========================================================================="
echo "Step 5: Large Memory Allocations"
echo "========================================================================="
echo ""

echo "Looking for torch.zeros/ones/randn with large sizes:"
# Split into separate patterns to avoid complex regex
grep -rn "torch.zeros" src/ --include="*.py" 2>/dev/null | head -10
grep -rn "torch.ones" src/ --include="*.py" 2>/dev/null | head -10
grep -rn "torch.randn" src/ --include="*.py" 2>/dev/null | head -10
echo ""

echo "Looking for .cuda() or .to(device) calls:"
cuda_calls=$(grep -rn "\.cuda()\|\.to(device)\|\.to('cuda')" src/ --include="*.py" 2>/dev/null | grep -v __pycache__ | wc -l)
echo "Total .cuda() calls found: $cuda_calls (potential leak points)"
echo ""

# ================================================================
# Step 6: Check for caching/history structures (FIXED REGEX)
# ================================================================
echo "========================================================================="
echo "Step 6: Cache/History Structures"
echo "========================================================================="
echo ""

echo "Checking for history buffers:"
# Simplified regex patterns
grep -rn "self\.[a-z_]*.*=.*\[\]" src/ --include="*.py" 2>/dev/null | head -10
echo ""
grep -rn "self\.[a-z_]*.*=.*deque" src/ --include="*.py" 2>/dev/null | head -10
echo ""
grep -rn "self\.[a-z_]*.*=.*dict()" src/ --include="*.py" 2>/dev/null | head -10
echo ""

# ================================================================
# Step 7: Configuration comparison
# ================================================================
echo "========================================================================="
echo "Step 7: Configuration Analysis"
echo "========================================================================="
echo ""

echo "Phase 4 configuration:"
if [ -f "confs/ghop_quick_bottle_1.yaml" ]; then
    grep -A 30 "^phase4:" confs/ghop_quick_bottle_1.yaml
else
    echo "Config file not found"
fi
echo ""

echo "Phase 5 configuration:"
if [ -f "confs/ghop_quick_bottle_1.yaml" ]; then
    grep -A 30 "^phase5:" confs/ghop_quick_bottle_1.yaml
else
    echo "Config file not found"
fi
echo ""

# ================================================================
# Step 8: Additional analysis - Check for known memory-hungry operations
# ================================================================
echo "========================================================================="
echo "Step 8: Known Memory-Hungry Operations"
echo "========================================================================="
echo ""

echo "Checking for model.eval() without torch.no_grad():"
grep -rn "\.eval()" src/ --include="*.py" 2>/dev/null | head -10
echo ""

echo "Checking for model loading:"
grep -rn "load_state_dict\|torch.load" src/ --include="*.py" 2>/dev/null | head -10
echo ""

echo "Checking for DataLoader with pin_memory:"
grep -rn "pin_memory" src/ --include="*.py" 2>/dev/null
grep -rn "pin_memory" confs/*.yaml 2>/dev/null
echo ""

echo "========================================================================="
echo "Analysis complete!"
echo "========================================================================="
