# Create: scripts/debug_mask_gradients.py
import torch
import sys
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
import argparse
import traceback

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from src.hold.hold import HOLD
from src.datasets.ghop_hoi_dataset import GHOPHOIDataset

# Load checkpoint
model = HOLD.load_from_checkpoint(
    "logs/640c1f867_final/checkpoints/last.ckpt"
)
model.eval()

# Get one batch
dataset = GHOPHOIDataset(split='train')
batch = dataset[0]
batch = {k: v.unsqueeze(0).cuda() if torch.is_tensor(v) else v
         for k, v in batch.items()}

# Forward pass with gradient tracking
model.train()
outputs = model(batch)

# Check what outputs exist
print("Available outputs:", outputs.keys())
print("Batch keys:", batch.keys())

# Check if mask loss exists
if 'loss/mask' in outputs:
    mask_loss = outputs['loss/mask']
    print(f"Mask loss: {mask_loss.item()}")

    # Check if it has gradients to v3d_cano
    mask_loss.backward(retain_graph=True)
    if hasattr(model.obj_model, 'v3d_cano'):
        print(f"v3d_cano grad: {model.obj_model.v3d_cano.grad}")
        print(f"v3d_cano requires_grad: {model.obj_model.v3d_cano.requires_grad}")
else:
    print("❌ NO MASK LOSS FOUND - confirms no GT supervision")

# Check if GT masks exist in batch
if 'object_mask' in batch:
    print(f"✅ GT object mask exists: {batch['object_mask'].shape}")
else:
    print("❌ NO GT MASK IN BATCH - dataset doesn't provide masks")