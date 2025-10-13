#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║         GHOP-HOLD Hardware Requirements Verification              ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Exit codes
ALL_CHECKS_PASSED=0

# Function to print status
print_status() {
    local status=$1
    local message=$2
    if [ "$status" == "PASS" ]; then
        echo -e "${GREEN}✓ PASS${NC}: $message"
    elif [ "$status" == "FAIL" ]; then
        echo -e "${RED}✗ FAIL${NC}: $message"
        ALL_CHECKS_PASSED=1
    elif [ "$status" == "WARN" ]; then
        echo -e "${YELLOW}⚠ WARN${NC}: $message"
    else
        echo -e "${BLUE}ℹ INFO${NC}: $message"
    fi
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. Operating System"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "   OS Name: $NAME"
    echo "   Version: $VERSION"

    # Check Ubuntu 20.04
    if [[ "$NAME" == *"Ubuntu"* ]] && [[ "$VERSION_ID" == "20.04" ]]; then
        print_status "PASS" "Ubuntu 20.04.x detected"
    elif [[ "$NAME" == *"Ubuntu"* ]]; then
        print_status "WARN" "Ubuntu version is $VERSION_ID (tested on 20.04)"
    else
        print_status "WARN" "Not Ubuntu (tested on Ubuntu 20.04)"
    fi
else
    print_status "FAIL" "Cannot detect OS version"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. GPU"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if nvidia-smi exists
if command -v nvidia-smi &> /dev/null; then
    # Get GPU info
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | wc -l)

    echo "   GPU Model: $GPU_NAME"
    echo "   GPU Memory: ${GPU_MEMORY} MB ($(echo "scale=2; $GPU_MEMORY/1024" | bc) GB)"
    echo "   GPU Count: $GPU_COUNT"

    # Check VRAM >= 24GB (24576 MB)
    if [ "$GPU_MEMORY" -ge 24576 ]; then
        print_status "PASS" "GPU has ${GPU_MEMORY} MB (≥24GB required)"
    elif [ "$GPU_MEMORY" -ge 20480 ]; then
        print_status "WARN" "GPU has ${GPU_MEMORY} MB (24GB recommended, may work with reduced batch size)"
    else
        print_status "FAIL" "GPU has only ${GPU_MEMORY} MB (<24GB, insufficient)"
    fi

    # Check GPU model recommendations
    if [[ "$GPU_NAME" == *"RTX 3090"* ]] || [[ "$GPU_NAME" == *"A5000"* ]] || [[ "$GPU_NAME" == *"A6000"* ]] || [[ "$GPU_NAME" == *"A100"* ]] || [[ "$GPU_NAME" == *"RTX 4090"* ]]; then
        print_status "PASS" "GPU model is recommended for this project"
    else
        print_status "INFO" "GPU model not in recommended list (RTX 3090, A5000+)"
    fi
else
    print_status "FAIL" "nvidia-smi not found. Is NVIDIA driver installed?"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3. System Memory (RAM)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Get total RAM in GB
TOTAL_RAM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
TOTAL_RAM_GB=$(echo "scale=2; $TOTAL_RAM_KB/1024/1024" | bc)

echo "   Total RAM: ${TOTAL_RAM_GB} GB"

# Check RAM >= 32GB
RAM_CHECK=$(echo "$TOTAL_RAM_GB >= 32" | bc)
RAM_RECOMMENDED=$(echo "$TOTAL_RAM_GB >= 64" | bc)

if [ "$RAM_RECOMMENDED" -eq 1 ]; then
    print_status "PASS" "RAM is ${TOTAL_RAM_GB} GB (≥64GB recommended, excellent)"
elif [ "$RAM_CHECK" -eq 1 ]; then
    print_status "PASS" "RAM is ${TOTAL_RAM_GB} GB (≥32GB minimum met)"
else
    print_status "FAIL" "RAM is only ${TOTAL_RAM_GB} GB (<32GB, insufficient)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4. Storage"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check root partition storage
ROOT_AVAIL=$(df -BG / | tail -1 | awk '{print $4}' | sed 's/G//')
ROOT_TOTAL=$(df -BG / | tail -1 | awk '{print $2}' | sed 's/G//')

echo "   Root (/) Available: ${ROOT_AVAIL} GB"
echo "   Root (/) Total: ${ROOT_TOTAL} GB"

# Check if available >= 500GB
if [ "$ROOT_AVAIL" -ge 500 ]; then
    print_status "PASS" "Available storage is ${ROOT_AVAIL} GB (≥500GB required)"
elif [ "$ROOT_AVAIL" -ge 300 ]; then
    print_status "WARN" "Available storage is ${ROOT_AVAIL} GB (500GB recommended)"
else
    print_status "FAIL" "Available storage is only ${ROOT_AVAIL} GB (<500GB, insufficient)"
fi

# Check if SSD
DISK_TYPE=""
ROOT_DEVICE=$(df / | tail -1 | awk '{print $1}' | sed 's/[0-9]*$//')
ROOT_DEVICE_NAME=$(basename $ROOT_DEVICE)

if [ -f "/sys/block/${ROOT_DEVICE_NAME}/queue/rotational" ]; then
    ROTATIONAL=$(cat /sys/block/${ROOT_DEVICE_NAME}/queue/rotational)
    if [ "$ROTATIONAL" -eq 0 ]; then
        DISK_TYPE="SSD"
        print_status "PASS" "Root partition is on SSD (recommended)"
    else
        DISK_TYPE="HDD"
        print_status "WARN" "Root partition is on HDD (SSD strongly recommended)"
    fi
else
    print_status "INFO" "Cannot determine disk type"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "5. CUDA"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check nvcc
if command -v nvcc &> /dev/null; then
    NVCC_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9.]*\).*/\1/p')
    echo "   NVCC Version: $NVCC_VERSION"

    # Check if version is 11.1
    if [[ "$NVCC_VERSION" == "11.1" ]]; then
        print_status "PASS" "CUDA 11.1 detected (primary requirement)"
    elif [[ "$NVCC_VERSION" == "11.7" ]]; then
        print_status "PASS" "CUDA 11.7 detected (fallback for two-hand detection)"
    elif [[ "$NVCC_VERSION" == 11.* ]]; then
        print_status "WARN" "CUDA $NVCC_VERSION detected (11.1 or 11.7 recommended)"
    else
        print_status "WARN" "CUDA $NVCC_VERSION detected (11.1 required)"
    fi

    # Check CUDA_HOME
    if [ -n "$CUDA_HOME" ]; then
        echo "   CUDA_HOME: $CUDA_HOME"
        print_status "PASS" "CUDA_HOME environment variable is set"
    else
        print_status "WARN" "CUDA_HOME not set (may need to export)"
    fi
else
    print_status "FAIL" "nvcc not found. CUDA toolkit not installed or not in PATH"
fi

# Check CUDA runtime from nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    CUDA_DRIVER_VERSION=$(nvidia-smi | grep "CUDA Version" | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p')
    if [ -n "$CUDA_DRIVER_VERSION" ]; then
        echo "   CUDA Driver Version: $CUDA_DRIVER_VERSION"
        print_status "INFO" "CUDA driver supports CUDA $CUDA_DRIVER_VERSION"
    fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "6. Additional Checks"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check gcc
if command -v gcc &> /dev/null; then
    GCC_VERSION=$(gcc --version | head -n 1 | awk '{print $NF}')
    echo "   GCC Version: $GCC_VERSION"
    print_status "PASS" "GCC compiler found"
else
    print_status "WARN" "GCC not found (needed for building extensions)"
fi

# Check g++
if command -v g++ &> /dev/null; then
    GPP_VERSION=$(g++ --version | head -n 1 | awk '{print $NF}')
    echo "   G++ Version: $GPP_VERSION"
    print_status "PASS" "G++ compiler found"
else
    print_status "WARN" "G++ not found (needed for building extensions)"
fi

# Check conda
if command -v conda &> /dev/null; then
    CONDA_VERSION=$(conda --version | awk '{print $2}')
    echo "   Conda Version: $CONDA_VERSION"
    print_status "PASS" "Conda found"
else
    print_status "WARN" "Conda not found (recommended for environment management)"
fi

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    echo "   System Python Version: $PYTHON_VERSION"
    print_status "INFO" "System Python found (will use conda env Python 3.8)"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                         Summary                                    ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Print summary
if [ $ALL_CHECKS_PASSED -eq 0 ]; then
    echo -e "${GREEN}✓ All critical checks passed!${NC}"
    echo -e "${GREEN}  System meets requirements for GHOP-HOLD integration.${NC}"
    exit 0
else
    echo -e "${RED}✗ Some checks failed.${NC}"
    echo -e "${RED}  Please address the issues before proceeding.${NC}"
    exit 1
fi

