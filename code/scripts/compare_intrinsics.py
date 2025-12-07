# scripts/compare_intrinsics.py
import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, ".")

# HO3D official intrinsics (from HO3D paper/dataset)
HO3D_OFFICIAL_K = np.array([
    [617.343, 0, 312.42],
    [0, 617.343, 241.42],
    [0, 0, 1]
])

HO3D_OFFICIAL_RES = (640, 480)  # W, H


def main():
    print("\n" + "=" * 70)
    print("INTRINSICS COMPARISON")
    print("=" * 70)

    print("\n[1] HO3D Official (from dataset documentation):")
    print(f"  Resolution: {HO3D_OFFICIAL_RES[0]}×{HO3D_OFFICIAL_RES[1]}")
    print(f"  K matrix:\n{HO3D_OFFICIAL_K}")
    print(f"  fx={HO3D_OFFICIAL_K[0, 0]:.2f}, fy={HO3D_OFFICIAL_K[1, 1]:.2f}")
    print(f"  cx={HO3D_OFFICIAL_K[0, 2]:.2f}, cy={HO3D_OFFICIAL_K[1, 2]:.2f}")

    # Load from your data
    data_dir = Path("data/hold_MC1_ho3d/build")
    if (data_dir / "data.npy").exists():
        print("\n[2] Your data/hold_MC1_ho3d/build/data.npy:")
        data = np.load(data_dir / "data.npy", allow_pickle=True).item()

        if 'K' in data:
            K_yours = np.array(data['K'])
            print(f"  K matrix:\n{K_yours}")
            print(f"  fx={K_yours[0, 0]:.2f}, fy={K_yours[1, 1]:.2f}")
            print(f"  cx={K_yours[0, 2]:.2f}, cy={K_yours[1, 2]:.2f}")

            # Compare
            print("\n[3] Difference:")
            diff = K_yours - HO3D_OFFICIAL_K
            print(f"  ΔK:\n{diff}")
            print(f"  Δfx = {diff[0, 0]:.2f}")
            print(f"  Δfy = {diff[1, 1]:.2f}")
            print(f"  Δcx = {diff[0, 2]:.2f}")
            print(f"  Δcy = {diff[1, 2]:.2f}")

            # Check if scaled
            if 'img_hw' in data:
                img_hw = data['img_hw']
                print(f"\n[4] Resolution in data.npy:")
                print(f"  img_hw: {img_hw}")

                scale_h = img_hw[0] / HO3D_OFFICIAL_RES[1]
                scale_w = img_hw[1] / HO3D_OFFICIAL_RES[0]
                print(f"  Scale factor: H={scale_h:.3f}, W={scale_w:.3f}")

                if abs(scale_h - 0.5) < 0.01:
                    print(f"  ⚠️  HALF RESOLUTION DETECTED!")
                    print(f"      Training at {img_hw[1]}×{img_hw[0]} instead of 640×480")
        else:
            print("  No 'K' key found")
    else:
        print("\n[2] data/hold_MC1_ho3d/build/data.npy not found")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()