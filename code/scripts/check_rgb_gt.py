import torch
import sys
from omegaconf import OmegaConf
from src.datasets.utils import create_dataset

sys.path.insert(0, '.')

config = OmegaConf.load('confs/ghop_production_chunked_20251031_140851.yaml')

class Args:
    case = 'ghop_bottle_1'
    n_images = 71

args = Args()

# Create training dataset
train_config = config.dataset.train
train_dataset = create_dataset(train_config, args)

# Check first batch
batch = train_dataset[0]

print("="*70)
print("TRAINING BATCH CONTENTS")
print("="*70)
print(f"Batch keys: {batch.keys()}")
print()

if 'rgb' in batch:
    rgb = batch['rgb']
    print(f"✅ RGB found in batch")
    print(f"   Shape: {rgb.shape}")
    print(f"   Type: {rgb.dtype}")
    print(f"   Min: {rgb.min():.4f}, Max: {rgb.max():.4f}")
    print(f"   Mean: {rgb.mean():.4f}")
else:
    print("❌ RGB NOT found in batch!")
    print("   This means RGB loss cannot be computed!")

print()
print("="*70)