#!/bin/bash

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║              GHOP Checkpoint Investigation                         ║"
echo "╚════════════════════════════════════════════════════════════════════╝"

echo -e "\n1. Checking GHOP repository structure..."
if [ -d "../GHOP" ] || [ -d "~/GHOP" ]; then
    echo "  ✓ GHOP repository found locally"
    find ../GHOP -name "*.ckpt" -o -name "*checkpoint*" 2>/dev/null | head -20
else
    echo "  ✗ GHOP repository not found locally"
fi

echo -e "\n2. Checking for pretrained models in current structure..."
find . -name "*.ckpt" -o -name "*checkpoint*" 2>/dev/null | grep -E "(ghop|vqvae|unet)" | head -20

echo -e "\n3. GHOP Checkpoint Requirements:"
echo "  - vqvae_last.ckpt: VQ-VAE encoder/decoder (compresses 64³ SDF to 16³ latent)"
echo "  - unet_last.ckpt: 3D U-Net diffusion model (18-channel input)"

echo -e "\n4. Options for obtaining checkpoints:"
echo "  Option A: Download from GHOP official release (if available)"
echo "  Option B: Train from scratch using GHOP training pipeline"
echo "  Option C: Use placeholder/dummy checkpoints for testing pipeline"

echo -e "\n5. Checking GHOP project configuration..."
if [ -f "code/src/model/ghop/ghop_prior.py" ]; then
    echo "  ✓ GHOP integration code exists"
    grep -n "checkpoint" code/src/model/ghop/ghop_prior.py | head -10
fi

echo -e "\n╔════════════════════════════════════════════════════════════════════╗"
echo "║                    Investigation Complete                          ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
