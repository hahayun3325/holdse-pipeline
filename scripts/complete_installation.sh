#!/bin/bash
set -e

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║            Completing GHOP-HOLD Installation                       ║"
echo "╚════════════════════════════════════════════════════════════════════╝"

cd ~/Projects/holdse

# Step 1: Fix PyTorch version
echo -e "\n[1/7] Fixing PyTorch version..."
pip uninstall torch torchvision torchaudio -y || true
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 \
  -f https://download.pytorch.org/whl/torch_stable.html

# Step 2: Build pytorch3d
echo -e "\n[2/7] Building pytorch3d..."
mkdir -p submodules && cd submodules
[ ! -d "pytorch3d" ] && git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && git checkout 35badc08
pip install -e .
cd ../..

# Step 3: Build kaolin
echo -e "\n[3/7] Building kaolin..."
cd submodules
[ ! -d "kaolin" ] && git clone --recursive https://github.com/NVIDIAGameWorks/kaolin
cd kaolin && git checkout v0.10.0
python setup.py install
cd ../..

# Step 4: Install SMPLX
echo -e "\n[4/7] Installing SMPLX..."
cd submodules
[ ! -d "smplx" ] && git clone https://github.com/zc-alexfan/smplx.git
cd smplx && git checkout 6675c3da8
python setup.py install
cd ../..

# Step 5: Build HOLD extensions
echo -e "\n[5/7] Building HOLD C++ extensions..."
[ -d "code" ] && cd code && python setup.py build_ext --inplace && cd ..

# Step 6: Fix compatibility
echo -e "\n[6/7] Fixing package compatibility..."
pip install --force-reinstall setuptools==59.5.0 numpy==1.23.5 scikit-image==0.18.1

# Step 7: Verify
echo -e "\n[7/7] Running verification..."
python scripts/verify_complete.py

echo -e "\n✓ Installation complete!"
