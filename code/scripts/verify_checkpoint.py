import sys
import torch
from pathlib import Path

ckpt_path = sys.argv[1]
print(f"Verifying: {ckpt_path}")

try:
    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location='cpu')

    # Print summary
    print(f"✓ Loaded: {len(ckpt)} keys")
    print(f"  Keys: {list(ckpt.keys())}")

    if 'state_dict' in ckpt:
        print(f"✓ State dict: {len(ckpt['state_dict'])} parameters")

    if 'epoch' in ckpt:
        print(f"✓ Epoch: {ckpt['epoch']}")

    if 'global_step' in ckpt:
        print(f"✓ Global step: {ckpt['global_step']}")

    print("✓ Verification complete")

except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)