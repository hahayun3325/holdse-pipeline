#!/bin/bash
# File: ~/Projects/holdse/code/find_checkpoint.sh

echo "======================================================================"
echo "Searching for HOLD checkpoint: 75268d864"
echo "======================================================================"

cd ~/Projects/holdse

# Search in multiple locations
echo "[1] Checking HOLD reference project..."
if [ -d ~/Projects/hold/code/saved_models ]; then
    find ~/Projects/hold/code/saved_models -name "*.ckpt" -type f 2>/dev/null | head -20
fi

echo ""
echo "[2] Checking HOLDSE checkpoints directory..."
if [ -d ./checkpoints ]; then
    find ./checkpoints -name "*.ckpt" -type f 2>/dev/null | head -20
fi

echo ""
echo "[3] Checking HOLD downloads..."
if [ -d ~/Projects/hold/downloads ]; then
    find ~/Projects/hold/downloads -name "*.ckpt" -type f 2>/dev/null | head -20
fi

echo ""
echo "[4] Searching for any checkpoint with '75268d864' in path..."
find ~ -path "*/75268d864/*" -name "*.ckpt" 2>/dev/null

echo ""
echo "[5] Searching for any 'last.ckpt' files..."
find ~/Projects/hold* -name "last.ckpt" -type f 2>/dev/null | head -10

echo ""
echo "======================================================================"