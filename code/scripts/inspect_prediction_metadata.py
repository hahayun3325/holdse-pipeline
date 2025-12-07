# scripts/inspect_prediction_metadata.py
import os
import sys
import torch
import argparse
import numpy as np

os.environ["COMET_MODE"] = "disabled"
sys.path.insert(0, ".")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("PREDICTION FILE METADATA")
    print("=" * 70)
    print(f"File: {args.predictions}\n")

    # Load with torch to handle persistent IDs
    data = torch.load(args.predictions, map_location="cpu")

    print("[1] Top-level keys:")
    for key in list(data.keys())[:15]:
        val = data[key]
        if hasattr(val, 'shape'):
            print(f"  {key}: shape {val.shape}")
        elif isinstance(val, (list, tuple)):
            print(f"  {key}: {type(val).__name__} len={len(val)}")
        else:
            print(f"  {key}: {type(val).__name__}")

    # Try to infer resolution from vertex/joint arrays
    print("\n[2] Inferred coordinate scales:")

    # Right hand joints
    if 'j3d_c.right' in data:
        j3d = data['j3d_c.right']
        print(f"  j3d_c.right shape: {j3d.shape}")
        print(f"  Mean magnitude: {np.linalg.norm(j3d, axis=-1).mean():.3f}")
        print(f"  Range: [{j3d.min():.3f}, {j3d.max():.3f}]")

    # Object vertices
    if 'v3d_c.object' in data:
        v3d = data['v3d_c.object']
        print(f"  v3d_c.object shape: {v3d.shape}")
        print(f"  Mean magnitude: {np.linalg.norm(v3d, axis=-1).mean():.3f}")
        print(f"  Range: [{v3d.min():.3f}, {v3d.max():.3f}]")

    # Check if metadata exists
    print("\n[3] Searching for metadata:")
    meta_keys = [k for k in data.keys() if 'meta' in k.lower() or 'info' in k.lower()]
    if meta_keys:
        for key in meta_keys:
            print(f"  Found: {key}")
            print(f"    {data[key]}")
    else:
        print("  No explicit metadata found")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()