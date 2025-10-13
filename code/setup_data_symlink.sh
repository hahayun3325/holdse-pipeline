#!/bin/bash
# File: ~/Projects/holdse/code/setup_data_symlink.sh

echo "======================================================================"
echo "HOLDSE Data Setup: Symlinking HOLD datasets"
echo "======================================================================"

# Step 1: Check if HOLD data exists
HOLD_DATA_PATH=~/Projects/hold/code/data
if [ ! -d "$HOLD_DATA_PATH" ]; then
    echo "❌ ERROR: HOLD data not found at $HOLD_DATA_PATH"
    echo "Please run Option 2 to download datasets"
    exit 1
fi

# Step 2: Count available sequences
NUM_SEQUENCES=$(ls -d "$HOLD_DATA_PATH"/hold_* 2>/dev/null | wc -l)
echo "✓ Found $NUM_SEQUENCES HOLD sequences"

# Step 3: Navigate to HOLDSE code directory
cd ~/Projects/holdse/code/

# Step 4: Remove existing data symlink if present
if [ -L "./data" ]; then
    echo "Removing existing symlink..."
    rm ./data
fi

# Step 5: Create symlink
ln -s "$HOLD_DATA_PATH" ./data
echo "✓ Created symlink: ./data -> $HOLD_DATA_PATH"

# Step 6: Verify symlink
if [ -L "./data" ] && [ -d "./data" ]; then
    echo "✓ Symlink verification successful"
    ls -la ./data | head -10
else
    echo "❌ Symlink creation failed"
    exit 1
fi

# Step 7: Test data access
TEST_SEQUENCE="hold_MC1_ho3d"
TEST_FILE="./data/$TEST_SEQUENCE/build/data.npy"

if [ -f "$TEST_FILE" ]; then
    echo "✓ Test file accessible: $TEST_FILE"
    python -c "
import numpy as np
data = np.load('$TEST_FILE', allow_pickle=True).item()
print(f'✓ Data keys: {list(data.keys())}')
print(f'✓ Frames: {data.get(\"frames\", \"N/A\")}')
"
else
    echo "❌ Test file not found: $TEST_FILE"
    exit 1
fi

echo "======================================================================"
echo "✓ Setup complete! Available sequences:"
ls -1 ./data/ | grep "^hold_"
echo "======================================================================"
echo ""
echo "Run training with:"
echo "  python sanity_train.py --case $TEST_SEQUENCE --shape_init 75268d864 --gpu_id 0"