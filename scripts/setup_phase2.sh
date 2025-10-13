#!/bin/bash
# Phase 2 setup script

echo "=========================================="
echo "Phase 2: GHOP Integration Setup"
echo "=========================================="

# 1. Create checkpoint directory
echo "Creating checkpoint directory..."
mkdir -p checkpoints/ghop
mkdir -p data/ghop

# 2. Download GHOP pretrained models
echo "Downloading GHOP pretrained models..."
echo "Please download the following files manually:"
echo "  - VQ-VAE: checkpoints/ghop/vqvae_last.ckpt"
echo "  - U-Net: checkpoints/ghop/unet_last.ckpt"
echo "  - Text templates: data/ghop/text_templates.json"
echo ""
echo "Download from: [GHOP release URL]"

# 3. Verify MANO model
echo "Verifying MANO model..."
if [ ! -d "assets/mano" ]; then
    echo "❌ MANO model not found!"
    echo "Please download from: https://mano.is.tue.mpg.de/"
    echo "Place in: assets/mano/"
else
    echo "✓ MANO model found"
fi

# 4. Install additional dependencies
echo "Installing dependencies..."
pip install einops
pip install omegaconf

# 5. Test installation
echo "Testing installation..."
python scripts/test_ghop.py

echo "=========================================="
echo "Phase 2 setup complete!"
echo "=========================================="
echo "Next steps:"
echo "1. Enable Phase 2 in confs/general.yaml (phase2.enabled: true)"
echo "2. Run: python train.py --conf confs/general.yaml"