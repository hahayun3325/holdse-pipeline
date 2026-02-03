import torch
import os

# Paths
official_ckpt = "/home/fredcui/Projects/hold/code/logs/cb20a1702/checkpoints/last.ckpt"
holdse_ckpt = "logs/92ebd6cdc_000020000/checkpoints/last.ckpt"


def inspect_checkpoint(path, name):
    print(f"\n{'=' * 60}")
    print(f"Inspecting: {name}")
    print(f"Path: {path}")
    print(f"{'=' * 60}")

    if not os.path.exists(path):
        print(f"ERROR: Checkpoint not found at {path}")
        return

    ckpt = torch.load(path, map_location='cpu')

    # Check top-level keys
    print(f"\n1. Top-level keys: {list(ckpt.keys())}")

    # Check state_dict keys
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        print(f"\n2. State dict has {len(state_dict)} keys")

        # Look for object-related keys
        object_keys = [k for k in state_dict.keys() if 'object' in k.lower() or 'obj' in k.lower()]
        print(f"\n3. Object-related keys ({len(object_keys)} found):")
        for k in object_keys[:10]:  # Print first 10
            print(f"   {k}: shape={state_dict[k].shape if hasattr(state_dict[k], 'shape') else 'N/A'}")
            print(f"      mean={state_dict[k].mean().item():.4f}, std={state_dict[k].std().item():.4f}")

        # Look for mask/segmentation keys
        mask_keys = [k for k in state_dict.keys() if 'mask' in k.lower() or 'seg' in k.lower()]
        print(f"\n4. Mask/segmentation-related keys ({len(mask_keys)} found):")
        for k in mask_keys[:10]:
            print(f"   {k}")

        # Look for SDF-related keys (critical for object geometry)
        sdf_keys = [k for k in state_dict.keys() if 'sdf' in k.lower()]
        print(f"\n5. SDF-related keys ({len(sdf_keys)} found):")
        for k in sdf_keys[:10]:
            tensor = state_dict[k]
            print(f"   {k}: shape={tensor.shape}")
            print(
                f"      min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, mean={tensor.mean().item():.4f}")

        # Look for v3d_cano (object vertices)
        v3d_keys = [k for k in state_dict.keys() if 'v3d_cano' in k]
        print(f"\n6. v3d_cano keys ({len(v3d_keys)} found):")
        for k in v3d_keys:
            tensor = state_dict[k]
            print(f"   {k}: shape={tensor.shape}")
            print(f"      mean={tensor.mean().item():.4f}, std={tensor.std().item():.4f}")

    # Check hyperparameters
    if 'hyper_parameters' in ckpt:
        hparams = ckpt['hyper_parameters']
        print(f"\n7. Hyperparameters present: {list(hparams.keys())[:10]}...")

        # Check for mask-related config
        if 'w_mask' in hparams:
            print(f"   w_mask (mask loss weight): {hparams['w_mask']}")
        if 'w_mask_binary' in hparams:
            print(f"   w_mask_binary: {hparams['w_mask_binary']}")


inspect_checkpoint(official_ckpt, "OFFICIAL HOLD")
inspect_checkpoint(holdse_ckpt, "HOLDSE (20000 steps)")