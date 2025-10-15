#!/usr/bin/env python
# File: ~/Projects/holdse/code/verify_body_models.py (CORRECTED)

import os
import os.path as osp
import sys

def verify_body_models():
    print("=" * 70)
    print("Body Models Verification")
    print("=" * 70)

    body_models_path = "./body_models"

    # Check 1: Directory exists
    if not osp.exists(body_models_path):
        print("❌ FAILED: body_models directory does not exist")
        return False

    if osp.islink(body_models_path):
        target = os.readlink(body_models_path)
        print(f"✓ body_models is symlink -> {target}")
    else:
        print(f"✓ body_models is directory")

    # Check 2: Critical files
    required_files = {
        "MANO_RIGHT.pkl": "Right hand MANO model",
        "MANO_LEFT.pkl": "Left hand MANO model",
        "contact_zones.pkl": "Contact zone definitions"
    }

    all_ok = True
    for filename, description in required_files.items():
        filepath = osp.join(body_models_path, filename)
        if osp.exists(filepath):
            size = osp.getsize(filepath)
            print(f"✓ {filename}: {size:,} bytes - {description}")
        else:
            print(f"❌ {filename}: MISSING - {description}")
            all_ok = False

    # Check 3: Test import with CORRECT parameter name
    print("\n" + "=" * 70)
    print("Testing MANO import...")
    print("=" * 70)

    try:
        # Add src to path
        import sys
        sys.path.insert(0, './src')

        from utils.external.body_models import MANO

        # ✓ CORRECTED: Use 'model_path' instead of 'mano_root'
        mano_layer = MANO(
            model_path=osp.join(body_models_path, "MANO_RIGHT.pkl"),  # ✓ Full path to file
            is_rhand=True,
            use_pca=False,
            flat_hand_mean=False
        )
        print("✓ MANO import successful")
        print(f"✓ MANO layer initialized with {mano_layer.v_template.shape[0]} vertices")
        all_ok = all_ok and True

    except Exception as e:
        print(f"❌ MANO import failed: {e}")
        import traceback
        traceback.print_exc()
        all_ok = False

    print("\n" + "=" * 70)
    if all_ok:
        print("✓✓✓ ALL CHECKS PASSED ✓✓✓")
        print("You can now run training")
    else:
        print("❌ VERIFICATION FAILED")
        print("Check error details above")
    print("=" * 70)

    return all_ok

if __name__ == "__main__":
    success = verify_body_models()
    sys.exit(0 if success else 1)