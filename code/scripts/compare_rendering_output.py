import torch
import sys

sys.path.insert(0, '/home/fredcui/Projects/holdse/code')

from src.hold.hold import HOLD

# Load checkpoint and run a forward pass on one sample
ckpt_path = "logs/92ebd6cdc_000020000/checkpoints/last.ckpt"

print("=" * 60)
print("INSPECTING MODEL OUTPUT MASKS")
print("=" * 60)

# This requires loading the full model - simplified version
checkpoint = torch.load(ckpt_path, map_location='cpu')

if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']

    # Look for mask model parameters
    mask_keys = [k for k in state_dict.keys() if 'mask' in k.lower() and 'model' in k.lower()]
    print(f"\n1. Mask model parameters found: {len(mask_keys)}")
    for k in mask_keys[:10]:
        print(f"   {k}")

    # Check mask probability-related outputs (if stored in checkpoint)
    # These might be in the model's state or logged values
    print(f"\n2. Checkpoint keys: {list(checkpoint.keys())}")

    # Check if any mask-related logs are stored
    if 'callbacks' in checkpoint:
        print(f"\n3. Callbacks present: {list(checkpoint['callbacks'].keys())}")