#!/bin/bash
set -e

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║     GHOP-HOLD Integrated Environment - Fixed Installation          ║"
echo "╚════════════════════════════════════════════════════════════════════╝"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_step() {
    echo -e "\n${BLUE}[$(date +%H:%M:%S)]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠ WARNING:${NC} $1"
}

# Check if environment exists
if conda env list | grep -q "ghop_hold_integrated"; then
    log_warning "Environment ghop_hold_integrated already exists. Activating..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate ghop_hold_integrated
else
    log_step "Creating conda environment..."
    conda create -n ghop_hold_integrated python=3.8 -y
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate ghop_hold_integrated
fi

# Step 1: System dependencies
log_step "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y build-essential ffmpeg libgl1-mesa-glx

# Step 2: PyTorch
log_step "Installing PyTorch 1.9.1 + CUDA 11.1..."
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 \
  -f https://download.pytorch.org/whl/torch_stable.html

# Verify PyTorch
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'✓ PyTorch {torch.__version__} with CUDA {torch.version.cuda}')"

# Step 3: Fix pip for PyTorch Lightning compatibility
log_step "Fixing pip version for PyTorch Lightning 1.5.7..."
pip install --force-reinstall pip==24.0
pip --version

# Step 4: Install requirements.txt (excluding pytorch-lightning)
log_step "Installing HOLD dependencies from requirements.txt..."
if [ -f requirements.txt ]; then
    # Install requirements except pytorch-lightning
    grep -v "pytorch-lightning" requirements.txt > /tmp/requirements_temp.txt
    pip install -r /tmp/requirements_temp.txt || true
    rm /tmp/requirements_temp.txt
else
    log_warning "requirements.txt not found, installing manually..."
    pip install chumpy comet-ml==3.40.0 cython==0.29.20 easydict ipdb kornia==0.6.12 \
                loguru matplotlib opencv-python opencv-contrib-python-headless \
                open3d pymeshlab==2022.2.post4 pyparsing
fi

# Step 5: Install PyTorch Lightning 1.5.7
log_step "Installing PyTorch Lightning 1.5.7..."
pip install pytorch-lightning==1.5.7

# Verify PyTorch Lightning
python -c "import pytorch_lightning; print(f'✓ PyTorch Lightning {pytorch_lightning.__version__}')"

# Step 6: Install conda dependencies
log_step "Installing fvcore, iopath, nvidiacub via conda..."
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y

# Step 7: Install visualization tools
log_step "Installing visualization tools..."
pip install aitviewer==1.13.0
pip install trimesh scikit-image matplotlib opencv-python tqdm

# Step 8: GHOP dependencies
log_step "Installing GHOP dependencies..."
pip install diffusers==0.14.0 transformers==4.25.0
pip install ftfy regex
pip install git+https://github.com/openai/CLIP.git
pip install einops==0.6.0 hydra-core==1.2.0 omegaconf==2.2.3 gdown

# Step 9: Build pytorch3d
log_step "Building pytorch3d 0.7.4 from source..."
mkdir -p submodules && cd submodules

if [ ! -d "pytorch3d" ]; then
    git clone https://github.com/facebookresearch/pytorch3d.git
fi

cd pytorch3d
git checkout 35badc08
pip install -e .
cd ../..

# Verify pytorch3d
python -c "import pytorch3d; print(f'✓ PyTorch3D {pytorch3d.__version__}')"

# Step 10: Build kaolin
log_step "Building kaolin v0.10.0 from source..."
cd submodules

if [ ! -d "kaolin" ]; then
    git clone --recursive https://github.com/NVIDIAGameWorks/kaolin
fi

cd kaolin
git checkout v0.10.0
python setup.py install
cd ../..

# Verify kaolin
python -c "import kaolin; print(f'✓ Kaolin {kaolin.__version__}')"

# Step 11: Install custom SMPLX
log_step "Installing custom SMPLX..."
cd submodules

if [ ! -d "smplx" ]; then
    git clone https://github.com/zc-alexfan/smplx.git
fi

cd smplx
git checkout 6675c3da8
python setup.py install
cd ../..

# Verify SMPLX
python -c "import smplx; print('✓ SMPLX installed')"

# Step 12: Build HOLD C++ extensions
log_step "Building HOLD C++ extensions..."
if [ -d "code" ]; then
    cd code
    python setup.py build_ext --inplace
    cd ..
    echo "✓ HOLD extensions built"
else
    log_warning "code/ directory not found. Skip building HOLD extensions."
fi

# Step 13: Fix package compatibility
log_step "Fixing package compatibility..."
pip install --force-reinstall setuptools==59.5.0 numpy==1.23.5 scikit-image==0.18.1

# Step 14: Create directory structure
log_step "Creating directory structure..."
mkdir -p code/ghop code/src/hold confs checkpoints/ghop checkpoints/hold/pretrained
mkdir -p assets/mano third_party/mano_v1_2/models data/hold data/ghop data/lib
mkdir -p output logs tests scripts

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              ✓ Installation Complete!                              ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Next steps:"
echo "  1. Download MANO models from https://mano.is.tue.mpg.de/"
echo "  2. Download GHOP checkpoints (vqvae_last.ckpt, unet_last.ckpt)"
echo "  3. Run verification: python scripts/verify_installation.py"
echo ""