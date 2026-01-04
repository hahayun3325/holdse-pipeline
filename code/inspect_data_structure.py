# File: ~/Projects/holdse/code/inspect_data_structure.py
"""
Inspect the structure of data.npy files to understand preprocessing format.
"""

import numpy as np
import sys
from pathlib import Path


def inspect_data_npy(data_path):
    """
    Inspect contents of data.npy file.

    Args:
        data_path: Path to data.npy file
    """
    print(f"{'=' * 70}")
    print(f"Inspecting: {data_path}")
    print(f"{'=' * 70}")

    if not Path(data_path).exists():
        print(f"❌ File not found: {data_path}")
        return

    # Load data
    data = np.load(data_path, allow_pickle=True).item()

    # Print keys
    print(f"\n[Keys] ({len(data.keys())} total)")
    for key in sorted(data.keys()):
        print(f"  - {key}")

    # Print detailed info for each key
    print(f"\n[Detailed Contents]")
    for key, value in sorted(data.items()):
        print(f"\n{key}:")

        if isinstance(value, np.ndarray):
            print(f"  Type: numpy.ndarray")
            print(f"  Shape: {value.shape}")
            print(f"  Dtype: {value.dtype}")
            if value.size < 10:
                print(f"  Value: {value}")
        elif isinstance(value, list):
            print(f"  Type: list")
            print(f"  Length: {len(value)}")
            if len(value) > 0:
                print(f"  First item type: {type(value[0])}")
                if len(value) <= 3:
                    for i, item in enumerate(value):
                        print(f"    [{i}]: {item}")
        elif isinstance(value, dict):
            print(f"  Type: dict")
            print(f"  Keys: {list(value.keys())}")
        else:
            print(f"  Type: {type(value)}")
            print(f"  Value: {value}")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    # Default: inspect MC1_ho3d sequence
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = "./data/hold_MC1_ho3d/build/data.npy"

    inspect_data_npy(data_path)

    # Usage examples
    print("\nUsage:")
    print("  python inspect_data_structure.py ./data/hold_MC1_ho3d/build/data.npy")
    print("  python inspect_data_structure.py ./data/hold_mug1_itw/build/data.npy")

# import numpy as np
#
# data = np.load('data/hold_MC1_ho3d/build/data.npy', allow_pickle=True).item()
#
# print("="*60)
# print("ENTITIES STRUCTURE")
# print("="*60)
#
# # Inspect right (hand)
# print("\nentities['right']:")
# right = data['entities']['right']
# if isinstance(right, dict):
#     print(f"  Type: dict")
#     print(f"  Keys: {list(right.keys())}")
#     for k, v in right.items():
#         if isinstance(v, np.ndarray):
#             print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
#         else:
#             print(f"    {k}: {type(v)}")
# else:
#     print(f"  Type: {type(right)}")
#     if hasattr(right, 'shape'):
#         print(f"  Shape: {right.shape}")
#
# # Inspect object
# print("\nentities['object']:")
# obj = data['entities']['object']
# if isinstance(obj, dict):
#     print(f"  Type: dict")
#     print(f"  Keys: {list(obj.keys())}")
#     for k, v in obj.items():
#         if isinstance(v, np.ndarray):
#             print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
#         else:
#             print(f"    {k}: {type(v)}")
# else:
#     print(f"  Type: {type(obj)}")
#     if hasattr(obj, 'shape'):
#         print(f"  Shape: {obj.shape}")