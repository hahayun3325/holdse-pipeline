import torch
import numpy as np


def inspect_checkpoint(path, name):
    ckpt = torch.load(path, map_location='cpu')
    sd = ckpt['state_dict']

    print(f"\n{'=' * 60}")
    print(f"Inspecting: {name}")
    print(f"{'=' * 60}")

    # Find all object-related keys
    obj_keys = [k for k in sd.keys() if 'object' in k.lower()]
    print(f"\n1. Found {len(obj_keys)} object-related keys")

    # Check for specific geometry keys
    geometry_patterns = ['v3d_cano', 'f3d_cano', 'sdf', 'implicit', 'verts', 'mesh']
    for pattern in geometry_patterns:
        matching = [k for k in obj_keys if pattern in k.lower()]
        print(f"   - {pattern}: {len(matching)} keys")
        for k in matching[:3]:  # Show first 3
            tensor = sd[k]
            if tensor.dtype.is_floating_point:
                print(f"     {k}: shape={tensor.shape}, "
                      f"mean={tensor.mean():.4f}, std={tensor.std():.4f}, "
                      f"has_nan={torch.isnan(tensor).any().item()}")
            else:
                print(f"     {k}: shape={tensor.shape}, dtype={tensor.dtype} "
                      f"(integer tensor, skipping mean/std)")
    # Add to inspection script for HOLDSE
    sdf_keys = [k for k in sd.keys() if 'sdf' in k.lower() or 'implicit' in k.lower()]
    print(f"\nSDF/Implicit network keys: {len(sdf_keys)}")

    # Check if object model has valid parameters
    obj_param_keys = [k for k in sd.keys() if 'object' in k.lower() and 'model' in k.lower()]
    for k in obj_param_keys[:10]:
        tensor = sd[k]
        if tensor.dtype.is_floating_point:
            print(f"{k}: shape={tensor.shape}, mean={tensor.mean():.4f}, has_nan={torch.isnan(tensor).any()}")
        else:
            print(f"{k}: shape={tensor.shape}, dtype={tensor.dtype} (integer tensor)")

    # Check for NaN or extreme values
    print(f"\n2. Checking for invalid values...")
    for k in obj_keys:
        if torch.isnan(sd[k]).any():
            print(f"   WARNING: {k} contains NaN!")
        if torch.isinf(sd[k]).any():
            print(f"   WARNING: {k} contains Inf!")

    # Check SDF health
    sdf_grid = sd.get('model.nodes.object.server.object_model.sdf_grid', None)
    if sdf_grid is not None:
        print(f"\nSDF Grid Analysis:")
        print(f"  min: {sdf_grid.min():.4f}, max: {sdf_grid.max():.4f}")
        print(f"  zero-crossings: {((sdf_grid[:-1] * sdf_grid[1:]) < 0).sum().item()}")

    # Check v3d_cano variance
    v3d = sd.get('model.nodes.object.server.object_model.v3d_cano', None)
    if v3d is not None:
        print(f"\nv3d_cano Analysis:")
        print(f"  bbox: [{v3d.min():.4f}, {v3d.max():.4f}]")
        print(f"  std: {v3d.std():.4f} (very low = collapsed)")


# Run comparison
inspect_checkpoint('logs/7dacf8bc6_000036000/checkpoints/last.ckpt', 'HOLDSE step 20000')
inspect_checkpoint('/home/fredcui/Projects/hold/code/logs/cb20a1702/checkpoints/last.ckpt', 'OFFICIAL HOLD')