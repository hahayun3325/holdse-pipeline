#!/bin/bash

# ============================================================================
# Phase 3: GHOP Two-Stage Training Setup Script
# ============================================================================
# This script sets up the environment for GHOP Phase 3 integration
# Supersedes Phase 2 with enhanced two-stage training support
# ============================================================================

set -e  # Exit on error

echo "=========================================================================="
echo "Phase 3: GHOP Two-Stage Training Setup"
echo "=========================================================================="
echo ""

# ============================================================================
# 1. Create Directory Structure
# ============================================================================
echo "[1/7] Creating directory structure..."

# Checkpoint directories
mkdir -p checkpoints/ghop
mkdir -p checkpoints/ghop/backups

# Data directories
mkdir -p data/ghop
mkdir -p data/ghop/text_templates

# Log directories
mkdir -p logs/phase3
mkdir -p logs/phase3/tensorboard

# Assets directories
mkdir -p assets/mano

echo "✓ Directories created"
echo ""

# ============================================================================
# 2. Download GHOP Pretrained Models
# ============================================================================
echo "[2/7] Downloading GHOP pretrained models..."

# Check if models already exist
if [ -f "checkpoints/ghop/vqvae_last.ckpt" ] && [ -f "checkpoints/ghop/unet_last.ckpt" ]; then
    echo "✓ GHOP checkpoints already exist, skipping download"
else
    echo "Downloading VQ-VAE and U-Net checkpoints..."

    # Option A: Automatic download (if URLs are available)
    # Uncomment and replace with actual URLs when available
    # wget -O checkpoints/ghop/vqvae_last.ckpt \
    #     "https://huggingface.co/YOUR_ORG/ghop/resolve/main/vqvae_last.ckpt"
    # wget -O checkpoints/ghop/unet_last.ckpt \
    #     "https://huggingface.co/YOUR_ORG/ghop/resolve/main/unet_last.ckpt"

    # Option B: Manual download instructions
    echo ""
    echo "⚠️  Please download GHOP checkpoints manually:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  1. VQ-VAE Checkpoint:"
    echo "     Source: [GHOP Release Page or HuggingFace]"
    echo "     Target: checkpoints/ghop/vqvae_last.ckpt"
    echo ""
    echo "  2. 3D U-Net Checkpoint:"
    echo "     Source: [GHOP Release Page or HuggingFace]"
    echo "     Target: checkpoints/ghop/unet_last.ckpt"
    echo ""
    echo "  3. Text Template Library:"
    echo "     Source: [GHOP Release Page]"
    echo "     Target: data/ghop/text_templates.json"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Press Enter after downloading, or Ctrl+C to exit..."
    read -r
fi

echo ""

# ============================================================================
# 3. Verify MANO Model
# ============================================================================
echo "[3/7] Verifying MANO hand model..."

if [ ! -d "assets/mano" ]; then
    echo "❌ MANO model not found!"
    echo ""
    echo "Please download MANO model:"
    echo "  1. Register at: https://mano.is.tue.mpg.de/"
    echo "  2. Download MANO_RIGHT.pkl"
    echo "  3. Place in: assets/mano/"
    echo ""
    exit 1
else
    # Check for required MANO files
    if [ -f "assets/mano/MANO_RIGHT.pkl" ] || [ -f "assets/mano/mano_right.pkl" ]; then
        echo "✓ MANO model found"
    else
        echo "⚠️  MANO directory exists but missing model files"
        echo "   Expected: assets/mano/MANO_RIGHT.pkl"
    fi
fi

echo ""

# ============================================================================
# 4. Install Python Dependencies
# ============================================================================
echo "[4/7] Installing Python dependencies..."

# Core dependencies for Phase 3
pip install -q einops           # For tensor reshaping
pip install -q omegaconf        # For configuration management
pip install -q trimesh          # For mesh processing (contact loss)
pip install -q scikit-learn     # For KNN (contact loss)

# Optional: For better visualization
pip install -q matplotlib
pip install -q plotly

echo "✓ Dependencies installed"
echo ""

# ============================================================================
# 5. Verify Checkpoint Integrity
# ============================================================================
echo "[5/7] Verifying checkpoint integrity..."

# Function to check file size
check_checkpoint() {
    local file=$1
    local min_size_mb=$2

    if [ -f "$file" ]; then
        local size_mb=$(du -m "$file" | cut -f1)
        if [ "$size_mb" -ge "$min_size_mb" ]; then
            echo "✓ $file (${size_mb} MB)"
            return 0
        else
            echo "⚠️  $file exists but seems too small (${size_mb} MB, expected ≥${min_size_mb} MB)"
            return 1
        fi
    else
        echo "❌ $file not found"
        return 1
    fi
}

check_checkpoint "checkpoints/ghop/vqvae_last.ckpt" 50
check_checkpoint "checkpoints/ghop/unet_last.ckpt" 100

echo ""

# ============================================================================
# 6. Test GHOP Components
# ============================================================================
echo "[6/7] Testing GHOP component initialization..."

# Create a simple test script
cat > /tmp/test_phase3_ghop.py << 'EOF'
import torch
import sys

print("Testing GHOP Phase 3 components...")

try:
    # Test VQ-VAE import
    from src.model.ghop.autoencoder import GHOPVQVAEWrapper
    print("✓ VQ-VAE module imported successfully")

    # Test U-Net import
    from src.model.ghop.diffusion import GHOP3DUNetWrapper
    print("✓ U-Net module imported successfully")

    # Test Hand Field import
    from src.model.ghop.hand_field import HandFieldBuilder
    print("✓ Hand Field module imported successfully")

    # Test SDS Loss import
    from src.model.ghop.ghop_loss import SDSLoss
    print("✓ SDS Loss module imported successfully")

    # Test Two-Stage Manager import
    from src.model.ghop.ghop_prior import TwoStageTrainingManager
    print("✓ Two-Stage Manager imported successfully")

    print("\n✓ All Phase 3 components available!")
    sys.exit(0)

except ImportError as e:
    print(f"\n❌ Import error: {e}")
    print("Please check that all Phase 3 modules are in place.")
    sys.exit(1)
EOF

python /tmp/test_phase3_ghop.py

if [ $? -eq 0 ]; then
    echo "✓ Component test passed"
else
    echo "❌ Component test failed"
    echo "Please ensure all Phase 3 code modifications are applied."
    exit 1
fi

echo ""

# ============================================================================
# 7. Generate Sample Configuration
# ============================================================================
echo "[7/7] Generating sample configuration..."

# Create a sample Phase 3 config if it doesn't exist
if [ ! -f "confs/phase3_sample.yaml" ]; then
    cat > confs/phase3_sample.yaml << 'EOF'
# Sample Phase 3 Configuration
# Copy to confs/general.yaml or use directly with --config

phase3:
  enabled: true
  use_modular_init: true

  # Two-stage training schedule
  sds_iters: 500
  contact_iters: 100

  # Loss weights
  w_sds: 5000.0
  w_contact: 10.0

  # Grid settings
  grid_resolution: 64
  spatial_lim: 1.5

  # Hand field
  use_hand_field: true
  hand_field_resolution: 64
  hand_field_limit: 1.5

  # GHOP checkpoints
  ghop:
    vqvae_checkpoint: "checkpoints/ghop/vqvae_last.ckpt"
    unet_checkpoint: "checkpoints/ghop/unet_last.ckpt"
    text_lib: "data/ghop/text_templates.json"

  # SDS parameters
  sds:
    diffusion_steps: 1000
    guidance_scale: 4.0
    min_step_ratio: 0.02
    max_step_ratio: 0.98
EOF
    echo "✓ Sample config created: confs/phase3_sample.yaml"
else
    echo "✓ Sample config already exists"
fi

echo ""

# ============================================================================
# Setup Complete
# ============================================================================
echo "=========================================================================="
echo "✓ Phase 3 Setup Complete!"
echo "=========================================================================="
echo ""
echo "Configuration Summary:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Checkpoints:    checkpoints/ghop/"
echo "  Text Templates: data/ghop/text_templates.json"
echo "  Sample Config:  confs/phase3_sample.yaml"
echo "  Logs:           logs/phase3/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Next Steps:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  1. Update confs/general.yaml:"
echo "     • Set phase3.enabled: true"
echo "     • Set phase3.use_modular_init: true"
echo ""
echo "  2. Train with Phase 3:"
echo "     python train.py --case YOUR_CASE --config confs/general.yaml"
echo ""
echo "  3. Monitor training:"
echo "     tensorboard --logdir logs/phase3/tensorboard"
echo ""
echo "  4. Or use command-line args:"
echo "     python train.py --case YOUR_CASE --use_ghop \\"
echo "         --sds_iters 500 --contact_iters 100"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""