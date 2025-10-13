#!/bin/bash
set -e

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║     GHOP-HOLD Integrated Environment - Automated Installation      ║"
echo "╚════════════════════════════════════════════════════════════════════╝"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_step() {
    echo -e "\n${BLUE}[$(date +%H:%M:%S)]${NC} $1"
}

# Step 1: Environment
log_step "Creating conda environment..."
conda create -n ghop_hold_integrated python=3.8 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ghop_hold_integrated

# Step 2: System dependencies
log_step "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y build-essential ffmpeg libgl1-mesa-glx

# Step 3: PyTorch
log_step "Installing PyTorch 1.9.1 + CUDA 11.1..."
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 \
  -f https://download.pytorch.org/whl/torch_stable.html

# Step 4: HOLD dependencies
log_step "Installing HOLD dependencies..."
pip install -r requirements.txt || true
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y
pip install pytorch-lightning==1.5.7 aitviewer==1.13.0
pip install trimesh scikit-image matplotlib opencv-python tqdm

# Step 5: GHOP dependencies
log_step "Installing GHOP dependencies..."
pip install diffusers==0.14.0 transformers==4.25.0
pip install ftfy regex
pip install git+https://github.com/openai/CLIP.git
pip install einops==0.6.0 hydra-core==1.2.0 omegaconf==2.2.3 gdown

# Step 6: Build pytorch3d
log_step "Building pytorch3d..."
mkdir -p submodules && cd submodules
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && git checkout 35badc08
pip install -e .
cd ../..

# Step 7: Build kaolin
log_step "Building kaolin..."
cd submodules
git clone --recursive https://github.com/NVIDIAGameWorks/kaolin
cd kaolin && git checkout v0.10.0
python setup.py install
cd ../..

# Step 8: Install SMPLX
log_step "Installing custom SMPLX..."
cd submodules
git clone https://github.com/zc-alexfan/smplx.git
cd smplx && git checkout 6675c3da8
python setup.py install
cd ../..

# Step 9: Build HOLD extensions
log_step "Building HOLD C++ extensions..."
cd code && python setup.py build_ext --inplace && cd ..

# Step 10: Fix compatibility
log_step "Fixing package compatibility..."
pip install --force-reinstall setuptools==59.5.0 numpy==1.23.5 scikit-image==0.18.1

# Step 11: Create structure
log_step "Creating directory structure..."
mkdir -p code/ghop code/src/hold confs checkpoints/ghop checkpoints/hold/pretrained
mkdir -p assets/mano third_party/mano_v1_2/models data/hold data/ghop data/lib
mkdir -p output logs tests scripts

echo -e "\n${GREEN}✓ Installation complete!${NC}"
echo "Next steps:"
echo "  1. Download MANO models from https://mano.is.tue.mpg.de/"
echo "  2. Download GHOP checkpoints"
echo "  3. Run: python scripts/verify_installation.py"

