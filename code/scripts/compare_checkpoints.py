# Save as: scripts/compare_checkpoints.py
import torch
import sys


def compare_checkpoints(official_path, stage1_path):
    """Compare two checkpoints dimension by dimension."""

    official = torch.load(official_path, map_location='cpu')
    stage1 = torch.load(stage1_path, map_location='cpu')

    print("=" * 100)
    print(f"{'Layer Name':<60} {'Official':<20} {'Stage 1':<20}")
    print("=" * 100)

    # Get all keys from both
    all_keys = set(official['state_dict'].keys()) | set(stage1['state_dict'].keys())

    mismatches = []
    matches = []

    for key in sorted(all_keys):
        official_shape = official['state_dict'].get(key, torch.tensor([])).shape
        stage1_shape = stage1['state_dict'].get(key, torch.tensor([])).shape

        if official_shape != stage1_shape:
            status = "❌ MISMATCH"
            mismatches.append((key, official_shape, stage1_shape))
            print(f"{key:<60} {str(official_shape):<20} {str(stage1_shape):<20} {status}")
        else:
            matches.append(key)

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Total layers: {len(all_keys)}")
    print(f"Matching: {len(matches)} ✅")
    print(f"Mismatching: {len(mismatches)} ❌")

    if mismatches:
        print("\n" + "=" * 100)
        print("CRITICAL MISMATCHES (Rendering Networks)")
        print("=" * 100)
        for key, off_shape, s1_shape in mismatches:
            if 'rendering_network' in key and 'lin0' in key:
                diff = off_shape[1] - s1_shape[1] if len(off_shape) > 1 and len(s1_shape) > 1 else 0
                print(f"\n{key}")
                print(f"  Official:  {off_shape} (d_in = {off_shape[1] if len(off_shape) > 1 else 'N/A'})")
                print(f"  Stage 1:   {s1_shape} (d_in = {s1_shape[1] if len(s1_shape) > 1 else 'N/A'})")
                print(f"  Diff:      {diff} dimensions")


if __name__ == '__main__':
    compare_checkpoints(
        '/home/fredcui/Projects/hold/code/logs/cb20a1702/checkpoints/last.ckpt',
        'logs/140dc5c18/checkpoints/last.ckpt'
    )