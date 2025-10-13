#!/usr/bin/env python3
"""Flexible verification that handles optional GHOP components."""

import sys
import os
import torch

def header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def check(condition, message, optional=False):
    status = "✓" if condition else ("⚠" if optional else "✗")
    color = "\033[92m" if condition else ("\033[93m" if optional else "\033[91m")
    print(f"{color}{status}\033[0m {message}")
    return condition if not optional else True

def verify_pytorch():
    header("1. PyTorch & CUDA")
    checks = []
    
    try:
        import torch
        print(f"   PyTorch: {torch.__version__}")
        checks.append(check(torch.__version__.startswith("1.9"), "PyTorch 1.9.x"))

        cuda_available = torch.cuda.is_available()
        checks.append(check(cuda_available, "CUDA available"))

        if cuda_available:
            print(f"   CUDA: {torch.version.cuda}")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            checks.append(check(torch.version.cuda == "11.1", "CUDA 11.1"))
    except Exception as e:
        checks.append(check(False, f"PyTorch import failed: {e}"))
        return False
    
    return all(checks)

def verify_packages():
    header("2. Critical Packages")
    
    # Required packages with correct import names
    required = [
        ("pytorch3d", "pytorch3d"),
        ("kaolin", "kaolin"),
        ("smplx", "smplx"),
        ("pytorch_lightning", "pytorch_lightning"),
        ("transformers", "transformers"),
        ("einops", "einops"),
        ("trimesh", "trimesh"),
        ("numpy", "numpy"),
    ]
    
    # Optional GHOP packages with correct import names
    optional = [
        ("diffusers", "diffusers"),
        ("clip", "clip"),
        ("hydra-core", "hydra"),  # Package name vs import name
        ("omegaconf", "omegaconf"),
    ]
    
    checks = []
    
    for name, mod in required:
        try:
            module = __import__(mod)
            version = getattr(module, "__version__", "installed")
            checks.append(check(True, f"{name}: {version}"))
        except ImportError as e:
            checks.append(check(False, f"{name}: NOT FOUND"))
        except Exception as e:
            checks.append(check(False, f"{name}: Error - {str(e)[:50]}"))

    for name, mod in optional:
        try:
            module = __import__(mod)
            version = getattr(module, "__version__", "installed")
            check(True, f"{name}: {version} (optional)", optional=True)
        except ImportError:
            check(False, f"{name}: NOT FOUND (optional)", optional=True)
        except Exception as e:
            check(False, f"{name}: Error (optional)", optional=True)

    return all(checks)

def verify_files():
    header("3. File Structure")
    
    required_files = [
        "code/confs/general.yaml",
        "assets/mano/MANO_RIGHT.pkl",
        "code/src/model/ghop/__init__.py",
        "data/lib/text_templates.json",
    ]
    
    optional_files = [
        "checkpoints/ghop/vqvae_last.ckpt",
        "checkpoints/ghop/unet_last.ckpt",
        "checkpoints/ghop/last.ckpt",  # Unified checkpoint
    ]
    
    checks = []
    
    for path in required_files:
        exists = os.path.exists(path)
        if exists:
            size = os.path.getsize(path)
            if size > 1024*1024:  # > 1MB
                size_str = f"{size/(1024*1024):.1f}MB"
            else:
                size_str = f"{size/1024:.1f}KB"
            checks.append(check(True, f"{path} ({size_str})"))
        else:
            checks.append(check(False, f"{path}"))
    
    for path in optional_files:
        exists = os.path.exists(path)
        if exists:
            size = os.path.getsize(path)
            size_str = f"{size/(1024**3):.2f}GB" if size > 1024**3 else f"{size/(1024**2):.1f}MB"
            check(True, f"{path} ({size_str}) (optional)", optional=True)
        else:
            check(False, f"{path} (optional)", optional=True)
    
    return all(checks)

def verify_ghop_mode():
    header("4. GHOP Integration Mode")
    
    # Check if GHOP checkpoints exist (either separate or unified)
    has_vqvae = os.path.exists("checkpoints/ghop/vqvae_last.ckpt")
    has_unet = os.path.exists("checkpoints/ghop/unet_last.ckpt")
    has_unified = os.path.exists("checkpoints/ghop/last.ckpt")

    if (has_vqvae and has_unet) or has_unified:
        print("   Mode: \033[92mFull GHOP Integration\033[0m")
        print("   → GHOP prior will be active during training")
        if has_unified:
            print("   → Using unified checkpoint (last.ckpt)")
        else:
            print("   → Using separate checkpoints (vqvae + unet)")
    else:
        print("   Mode: \033[93mHOLD Baseline\033[0m")
        print("   → Training without GHOP prior (standard HOLD)")
        print("   → GHOP checkpoints can be added later for enhancement")

    # Try to import GHOP module
    try:
        sys.path.insert(0, 'code')
        sys.path.insert(0, 'code/src')
        # Try different import paths
        try:
            from model.ghop import GHOPPriorModule
            check(True, "GHOP module (GHOPPriorModule) available", optional=True)
        except ImportError:
            # Try alternative import
            import model.ghop
            check(True, "GHOP module package available", optional=True)
    except Exception as e:
        check(False, f"GHOP module: {str(e)[:60]}", optional=True)
    
    return True  # Always pass - GHOP is optional

def verify_jutils():
    header("5. Stub Modules")

    try:
        sys.path.insert(0, 'code')
        import jutils
        version = getattr(jutils, "__version__", "unknown")
        check(True, f"jutils stub: {version}")
        return True
    except ImportError:
        check(False, "jutils stub: NOT FOUND")
        return False
    except Exception as e:
        check(False, f"jutils stub: Error - {e}")
        return False

def main():
    print("\n╔" + "="*68 + "╗")
    print("║" + " "*8 + "GHOP-HOLD Flexible Environment Verification" + " "*16 + "║")
    print("╚" + "="*68 + "╝")
    
    results = {
        "PyTorch & CUDA": verify_pytorch(),
        "Packages": verify_packages(),
        "Files": verify_files(),
        "Stub Modules": verify_jutils(),
        "GHOP Mode": verify_ghop_mode(),
    }
    
    header("Summary")
    
    all_critical = all(results.values())
    
    for component, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        color = "\033[92m" if passed else "\033[91m"
        print(f"   {component}: {color}{status}\033[0m")
    
    print("="*70)
    
    if all_critical:
        print("\n\033[92m✓ Environment ready for training!\033[0m")
        print("\nYou can now:")
        print("  • Train with HOLD baseline (no GHOP prior)")
        print("  • Add GHOP checkpoints later for full integration")
        print("\nStart training:")
        print("  cd code && python train.py --case <sequence_name>")
        print()
        return 0
    else:
        print("\n\033[91m✗ Critical components missing.\033[0m\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
