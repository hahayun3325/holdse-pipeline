#!/usr/bin/env python3
"""Complete environment verification for GHOP-HOLD."""

import sys
import os
import torch

def header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def check(condition, message):
    status = "✓" if condition else "✗"
    color = "\033[92m" if condition else "\033[91m"
    print(f"{color}{status}\033[0m {message}")
    return condition

def verify_pytorch():
    """Verify PyTorch and CUDA."""
    header("1. PyTorch & CUDA")
    
    checks = []
    print(f"   PyTorch: {torch.__version__}")
    checks.append(check(
        torch.__version__.startswith("1.9"),
        "PyTorch 1.9.x (required)"
    ))
    
    cuda_available = torch.cuda.is_available()
    print(f"   CUDA Available: {cuda_available}")
    checks.append(check(cuda_available, "CUDA available"))
    
    if cuda_available:
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        checks.append(check(
            torch.version.cuda == "11.1",
            "CUDA 11.1"
        ))
    
    return all(checks)

def verify_packages():
    """Verify critical packages."""
    header("2. Critical Packages")
    
    packages = [
        ("pytorch3d", "pytorch3d"),
        ("kaolin", "kaolin"),
        ("smplx", "smplx"),
        ("pytorch_lightning", "pytorch_lightning"),
        ("diffusers", "diffusers"),
        ("transformers", "transformers"),
        ("clip", "clip"),
        ("einops", "einops"),
        ("trimesh", "trimesh"),
        ("scipy", "scipy"),
        ("numpy", "numpy"),
        ("omegaconf", "omegaconf"),
        ("hydra", "hydra"),
    ]
    
    checks = []
    for name, mod in packages:
        try:
            module = __import__(mod)
            version = getattr(module, "__version__", "installed")
            checks.append(check(True, f"{name}: {version}"))
        except ImportError:
            checks.append(check(False, f"{name}: NOT FOUND"))
    
    return all(checks)

def verify_files():
    """Verify file structure."""
    header("3. File Structure")
    
    files = [
        ("code/confs/general.yaml", True),
        ("data/lib/text_templates.json", True),
        ("assets/mano/MANO_RIGHT.pkl", True),
        ("assets/mano/MANO_LEFT.pkl", True),
        ("third_party/mano_v1_2/models/MANO_RIGHT.pkl", True),
        ("checkpoints/ghop/vqvae_last.ckpt", False),  # Optional
        ("checkpoints/ghop/unet_last.ckpt", False),   # Optional
        ("code/src/model/ghop/__init__.py", True),
        ("code/src/model/ghop/ghop_prior.py", True),
        ("code/src/model/ghop/ghop_loss.py", True),
        ("code/src/model/ghop/hand_field.py", True),
    ]
    
    checks = []
    for filepath, required in files:
        exists = os.path.exists(filepath)
        if required:
            checks.append(check(exists, filepath))
        else:
            if exists:
                check(True, f"{filepath} (optional)")
            else:
                print(f"  ⚠ {filepath} (optional, not found)")
    
    return all(checks)

def verify_ghop_module():
    """Test GHOP module import."""
    header("4. GHOP Module")
    
    try:
        # Add code directory to path
        sys.path.insert(0, 'code')
        sys.path.insert(0, 'code/src')
        sys.path.insert(0, 'code/src/model')
        
        from model.ghop import GHOPPriorModule
        print("   Importing GHOPPriorModule from code/src/model/ghop...")
        
        return check(True, "GHOP module imports successfully")
    except Exception as e:
        return check(False, f"GHOP module failed: {e}")

def main():
    """Run all verifications."""
    print("\n╔" + "="*68 + "╗")
    print("║" + " "*8 + "GHOP-HOLD Complete Environment Verification" + " "*17 + "║")
    print("╚" + "="*68 + "╝")
    
    results = {
        "PyTorch & CUDA": verify_pytorch(),
        "Packages": verify_packages(),
        "Files": verify_files(),
        "GHOP Module": verify_ghop_module(),
    }
    
    header("Summary")
    
    all_passed = True
    for component, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        color = "\033[92m" if passed else "\033[91m"
        print(f"   {component}: {color}{status}\033[0m")
        all_passed = all_passed and passed
    
    print("="*70)
    
    if all_passed:
        print("\n\033[92m✓ Environment ready for GHOP-HOLD training!\033[0m")
        print("\nNext steps:")
        print("  1. Obtain GHOP checkpoints (optional for placeholder testing)")
        print("  2. Prepare training data")
        print("  3. Start training: cd code && python train.py --case <sequence_name>")
        print()
        return 0
    else:
        print("\n\033[91m✗ Some components need attention.\033[0m\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
