# File: ~/Projects/holdse/code/inspect_data_detailed.py
"""
Detailed inspection of HOLD data.npy structure for training setup.
"""

import numpy as np
import sys
from pathlib import Path


def inspect_detailed(data_path):
    """Detailed inspection with entity breakdown."""
    print(f"{'=' * 70}")
    print(f"DETAILED DATA INSPECTION: {Path(data_path).parent.parent.name}")
    print(f"{'=' * 70}\n")

    if not Path(data_path).exists():
        print(f"❌ File not found: {data_path}")
        return

    data = np.load(data_path, allow_pickle=True).item()

    # ========================================
    # 1. Sequence Info
    # ========================================
    print("[1] SEQUENCE INFO")
    print(f"  Sequence name: {data.get('seq_name', 'N/A')}")
    print(f"  Bounding sphere: {data.get('scene_bounding_sphere', 'N/A')}")
    print(f"  Max radius ratio: {data.get('max_radius_ratio', 'N/A')}")
    print(f"  Normalize shift: {data.get('normalize_shift', 'N/A')}")

    # ========================================
    # 2. Camera Info
    # ========================================
    print(f"\n[2] CAMERA DATA")
    cameras = data.get('cameras', {})

    # Count frames from camera keys
    frame_count = len([k for k in cameras.keys() if k.startswith('world_mat_')])
    print(f"  Total frames: {frame_count}")

    if frame_count > 0:
        # Inspect first frame camera matrices
        world_mat_0 = cameras.get('world_mat_0')
        scale_mat_0 = cameras.get('scale_mat_0')

        print(f"  world_mat_0 shape: {world_mat_0.shape if world_mat_0 is not None else 'N/A'}")
        print(f"  scale_mat_0 shape: {scale_mat_0.shape if scale_mat_0 is not None else 'N/A'}")

        if world_mat_0 is not None:
            print(f"\n  Sample world_mat_0:\n{world_mat_0}")

    # ========================================
    # 3. Entities (Hand + Object)
    # ========================================
    print(f"\n[3] ENTITIES")
    entities = data.get('entities', {})
    print(f"  Entity types: {list(entities.keys())}")

    # Hand entity
    if 'right' in entities:
        print(f"\n  [3.1] RIGHT HAND")
        hand = entities['right']
        print(f"    Keys: {list(hand.keys())}")

        for key, value in hand.items():
            if isinstance(value, np.ndarray):
                print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, list):
                print(f"    {key}: list[{len(value)}] (first: {value[0] if value else 'empty'})")
            elif isinstance(value, dict):
                print(f"    {key}: dict with keys {list(value.keys())}")
            else:
                print(f"    {key}: {type(value).__name__} = {value}")

    # Object entity
    if 'object' in entities:
        print(f"\n  [3.2] OBJECT")
        obj = entities['object']
        print(f"    Keys: {list(obj.keys())}")

        for key, value in obj.items():
            if isinstance(value, np.ndarray):
                print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, list):
                print(f"    {key}: list[{len(value)}] (first: {value[0] if value else 'empty'})")
            elif isinstance(value, dict):
                print(f"    {key}: dict with keys {list(value.keys())}")
            else:
                print(f"    {key}: {type(value).__name__} = {value}")

    # ========================================
    # 4. Data Integrity Check
    # ========================================
    print(f"\n[4] DATA INTEGRITY")

    checks = {
        'Has cameras': 'cameras' in data and len(data['cameras']) > 0,
        'Has entities': 'entities' in data,
        'Has hand data': 'entities' in data and 'right' in data['entities'],
        'Has object data': 'entities' in data and 'object' in data['entities'],
        'Has bounding sphere': 'scene_bounding_sphere' in data,
        'Frame count > 0': frame_count > 0,
    }

    for check, passed in checks.items():
        status = "✓" if passed else "❌"
        print(f"  {status} {check}")

    print(f"\n{'=' * 70}")

    # Return summary for programmatic use
    return {
        'seq_name': data.get('seq_name'),
        'frame_count': frame_count,
        'has_hand': 'right' in entities,
        'has_object': 'object' in entities,
        'entities': list(entities.keys()),
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        # Default to mug sequence
        data_path = "./data/hold_mug1_itw/build/data.npy"

    summary = inspect_detailed(data_path)

    print("\nRECOMMENDED TRAINING COMMAND:")
    seq_name = Path(data_path).parent.parent.name
    print(f"  python sanity_train.py --case {seq_name} --shape_init 75268d864 --gpu_id 0")