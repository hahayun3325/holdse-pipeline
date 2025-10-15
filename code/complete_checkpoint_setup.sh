#!/bin/bash
# File: ~/Projects/holdse/code/complete_checkpoint_setup.sh

set -e

echo "======================================================================"
echo "HOLDSE Complete Checkpoint Setup"
echo "======================================================================"

cd ~/Projects/holdse

# Step 1: Create directory structure
echo "[1/5] Creating directory structure..."
mkdir -p code/saved_models
mkdir -p downloads/mandatory
echo "✓ Directories created"

# Step 2: Download checkpoint
echo ""
echo "[2/5] Downloading 75268d864.zip..."
DOWNLOAD_URL="https://download.is.tue.mpg.de/download.php?domain=hold&sfile=mandatory/75268d864.zip"

if command -v wget &> /dev/null; then
    wget -O downloads/mandatory/75268d864.zip "$DOWNLOAD_URL" --no-check-certificate
elif command -v curl &> /dev/null; then
    curl -L -o downloads/mandatory/75268d864.zip "$DOWNLOAD_URL" --insecure
else
    echo "❌ Neither wget nor curl found"
    echo "Please install wget or curl, or download manually"
    exit 1
fi

if [ ! -f downloads/mandatory/75268d864.zip ]; then
    echo "❌ Download failed"
    exit 1
fi

SIZE=$(stat -c%s downloads/mandatory/75268d864.zip 2>/dev/null || stat -f%z downloads/mandatory/75268d864.zip)
echo "✓ Downloaded: $SIZE bytes"

# Step 3: Verify checksum
echo ""
echo "[3/5] Verifying checksum..."
EXPECTED="489cb46012ae7afbe01253401b3099fe3b26b8b3bf7c9251d72990513fa721eb"
ACTUAL=$(sha256sum downloads/mandatory/75268d864.zip | awk '{print $1}')

if [ "$ACTUAL" = "$EXPECTED" ]; then
    echo "✓ Checksum verified"
else
    echo "⚠️  Checksum mismatch (continuing anyway)"
fi

# Step 4: Extract
echo ""
echo "[4/5] Extracting archive..."
unzip -q downloads/mandatory/75268d864.zip -d downloads/mandatory/

if [ ! -d downloads/mandatory/75268d864 ]; then
    echo "❌ Extraction failed"
    exit 1
fi
echo "✓ Extracted"

# Step 5: Move to final location
echo ""
echo "[5/5] Installing checkpoint..."
mv downloads/mandatory/75268d864 code/saved_models/

# Verify installation
if [ -f code/saved_models/75268d864/checkpoints/last.ckpt ]; then
    CKPT_SIZE=$(stat -c%s code/saved_models/75268d864/checkpoints/last.ckpt 2>/dev/null || stat -f%z code/saved_models/75268d864/checkpoints/last.ckpt)
    echo "✓ Checkpoint installed successfully"
    echo "  Location: code/saved_models/75268d864/checkpoints/last.ckpt"
    echo "  Size: $CKPT_SIZE bytes"
else
    echo "❌ Installation verification failed"
    exit 1
fi

# Cleanup
rm -f downloads/mandatory/75268d864.zip
echo "✓ Cleanup complete"

echo ""
echo "======================================================================"
echo "✓✓✓ Setup Complete ✓✓✓"
echo "======================================================================"
echo ""
echo "Verification:"
ls -lh code/saved_models/75268d864/checkpoints/
echo ""
echo "You can now run sanity training:"
echo "  cd ~/Projects/holdse/code"
echo "  python sanity_train.py --case hold_mug1_itw --shape_init 75268d864 --gpu_id 0"
echo "======================================================================"