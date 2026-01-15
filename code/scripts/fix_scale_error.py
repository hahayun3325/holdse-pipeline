import torch
import numpy as np
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')

# Load HOLDSE checkpoint
ckpt_path = "logs/176872f9f/checkpoints/last.ckpt"
ckpt = torch.load(ckpt_path, map_location='cpu')

# Apply 0.56× scale correction to object parameters
# (Based on observed 1.79× error: 1/1.79 ≈ 0.56)
scale_factor = 0.56

# Find object scale parameter
for key in ckpt['state_dict'].keys():
    if 'obj_scale' in key:
        original = ckpt['state_dict'][key].item()
        corrected = original * scale_factor
        ckpt['state_dict'][key] = torch.tensor(corrected)
        print(f"Corrected {key}: {original:.6f} → {corrected:.6f}")

# Save corrected checkpoint
output_path = "logs/176872f9f/checkpoints/last_scale_corrected.ckpt"
torch.save(ckpt, output_path)
print(f"\n✓ Saved scale-corrected checkpoint to: {output_path}")