#!/bin/bash
# File: ~/Projects/holdse/code/setup_body_models_symlink.sh

echo "======================================================================"
echo "HOLDSE Body Models Setup: Symlinking from HOLD Reference"
echo "======================================================================"

cd ~/Projects/holdse/code

# Check if HOLD body_models exists
HOLD_BODY_MODELS=~/Projects/hold/code/body_models
if [ ! -d "$HOLD_BODY_MODELS" ]; then
    echo "❌ ERROR: HOLD body_models not found at $HOLD_BODY_MODELS"
    exit 1
fi

# Count files in body_models
NUM_FILES=$(ls -1 "$HOLD_BODY_MODELS" | wc -l)
echo "✓ Found $NUM_FILES files in HOLD body_models"

# Check critical files
REQUIRED_FILES=(
    "MANO_RIGHT.pkl"
    "MANO_LEFT.pkl"
    "contact_zones.pkl"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$HOLD_BODY_MODELS/$file" ]; then
        echo "❌ ERROR: Required file missing: $file"
        exit 1
    fi
    echo "  ✓ $file"
done

# Create symlink
if [ -L "./body_models" ]; then
    echo "⚠️  Warning: Symlink already exists, removing..."
    rm ./body_models
elif [ -d "./body_models" ]; then
    echo "⚠️  Warning: Directory exists, moving to backup..."
    mv ./body_models ./body_models.backup
fi

ln -s "$HOLD_BODY_MODELS" ./body_models
echo "✓ Created symlink: ./body_models -> $HOLD_BODY_MODELS"

# Verify symlink
if [ -L "./body_models" ] && [ -d "./body_models" ]; then
    echo "✓ Symlink verification successful"

    # Test critical file access
    if [ -f "./body_models/MANO_RIGHT.pkl" ]; then
        FILE_SIZE=$(stat -f%z "./body_models/MANO_RIGHT.pkl" 2>/dev/null || stat -c%s "./body_models/MANO_RIGHT.pkl")
        echo "✓ MANO_RIGHT.pkl accessible (size: $FILE_SIZE bytes)"
    else
        echo "❌ ERROR: Cannot access MANO_RIGHT.pkl through symlink"
        exit 1
    fi
else
    echo "❌ ERROR: Symlink creation failed"
    exit 1
fi

echo ""
echo "======================================================================"
echo "✓✓✓ Body Models Setup Complete ✓✓✓"
echo "======================================================================"
echo "Location: ./body_models -> $HOLD_BODY_MODELS"
echo "Files available: $NUM_FILES"
echo ""
echo "You can now run sanity training:"
echo "  python sanity_train.py --case hold_mug1_itw --shape_init 75268d864 --gpu_id 0"
echo "======================================================================"