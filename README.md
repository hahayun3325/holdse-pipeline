# HOLDSE Pipeline

**HOISE (Hand-Object Interaction Semantic Embedding)** pipeline integrating HOLD with GHOP for monocular hand-object reconstruction.

## Overview

This project combines:
- **HOLD**: Hand-Object reconstruction by Leveraging interaction constraints
- **GHOP**: Generative Hand-Object Prior for diffusion-based guidance

## Features

- 10-phase volumetric rendering with adaptive sampling
- VQ-VAE encoder with 18-channel interaction grid
- Score Distillation Sampling (SDS) loss for generative priors
- Contact-aware refinement for physical plausibility
- Temporal consistency optimization
- Support for 26+ HOLD sequences

## Quick Start

### Installation

```
# Clone repository
git clone <your-repo-url>
cd holdse

# Setup environment
conda create -n ghop_hold_integrated python=3.8
conda activate ghop_hold_integrated

# Install dependencies
pip install -r requirements.txt
```

### Training

```
cd code

# Single sequence
python sanity_train.py --case hold_mug1_itw --shape_init 75268d864 --gpu_id 0

# Batch training
bash batch_train_all.sh
```

## Project Structure

```
holdse/
├── code/               # Main codebase
│   ├── src/           # Source modules
│   │   ├── hold/      # HOLD implementation
│   │   ├── model/     # Neural networks
│   │   │   └── ghop/  # GHOP integration
│   │   └── utils/     # Utilities
│   └── confs/         # Configuration files
├── common/            # Shared utilities
├── data/              # Datasets (symlinked)
├── checkpoints/       # Model checkpoints
└── logs/              # Training logs
```

## Requirements

- Ubuntu 22.04 LTS
- CUDA 11.1+
- PyTorch 1.9.1
- RTX 4090 (24GB VRAM recommended)

## Citation

If you use this code, please cite the original papers:

```
@article{hold2023,
  title={HOLD: Hand-Object reconstruction by Leveraging interaction constraints},
  author={...},
  year={2023}
}

@article{ghop2024,
  title={GHOP: Generative Hand-Object Prior},
  author={...},
  year={2024}
}
```

## License

See LICENSE file for details.
