# File: code/diagnose_rendering.py
"""Diagnose why rendering produced only 3 samples."""

import torch
from pathlib import Path


def check_rendering_output(test_dir):
    """Check rendering output structure."""
    test_dir = Path(test_dir)
    visuals = test_dir / 'visuals'

    print("=" * 70)
    print("RENDERING OUTPUT DIAGNOSIS")
    print("=" * 70)

    # Count files in each directory
    for subdir in visuals.iterdir():
        if subdir.is_dir():
            files = list(subdir.glob('*.png')) + list(subdir.glob('*.jpg'))
            print(f"{subdir.name:25s} {len(files):3d} files")

            # Check naming pattern
            if files:
                first_file = files[0].name
                print(f"  └─ Example: {first_file}")

                # Check if it's sample-based or frame-based
                if 'id_' in first_file:
                    print(f"     ⚠️  Sample-based naming (validation style)")
                elif any(first_file.startswith(f'{i:05d}') for i in range(100)):
                    print(f"     ✅ Frame-based naming (render style)")

    # Check for meshes
    mesh_dir = test_dir / 'meshes'
    if mesh_dir.exists():
        meshes = list(mesh_dir.glob('*.obj'))
        print(f"\n{'meshes':25s} {len(meshes):3d} files")
        if meshes:
            print(f"  ✅ Meshes exported")
        else:
            print(f"  ❌ No meshes (directory empty)")
    else:
        print(f"\n{'meshes':25s}   0 files")
        print(f"  ❌ Directory doesn't exist")

    # Summary
    total_images = sum(len(list(d.glob('*.png'))) + len(list(d.glob('*.jpg')))
                       for d in visuals.iterdir() if d.is_dir())

    print("\n" + "=" * 70)
    print(f"SUMMARY")
    print("=" * 70)
    print(f"Total images: {total_images}")
    print(f"Expected: ~923 (71 frames × 13 types)")

    if total_images < 100:
        print(f"\n❌ PROBLEM: Only {total_images} images")
        print(f"   This suggests render.py processed only 1 batch")
        print(f"   with 3 pixel samples per batch.")
        print(f"\n💡 FIX: Accumulate ALL batch outputs before calling")
        print(f"   validation_epoch_end() with the full list.")


if __name__ == '__main__':
    check_rendering_output('logs/b2e4b039a/test')