import sys

print("Verifying critical packages...")

checks = []

# PyTorch
try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    checks.append(True)
except Exception as e:
    print(f"✗ PyTorch: {e}")
    checks.append(False)

# PyTorch Lightning
try:
    import pytorch_lightning as pl
    print(f"✓ PyTorch Lightning: {pl.__version__}")
    checks.append(True)
except Exception as e:
    print(f"✗ PyTorch Lightning: {e}")
    checks.append(False)

# pytorch3d
try:
    import pytorch3d
    print(f"✓ PyTorch3D: {pytorch3d.__version__}")
    checks.append(True)
except Exception as e:
    print(f"✗ PyTorch3D: {e}")
    checks.append(False)

# Other packages
packages = [
    ("diffusers", "diffusers"),
    ("transformers", "transformers"),
    ("clip", "clip"),
    ("einops", "einops"),
]

for name, module in packages:
    try:
        mod = __import__(module)
        version = getattr(mod, "__version__", "installed")
        print(f"✓ {name}: {version}")
        checks.append(True)
    except Exception as e:
        print(f"✗ {name}: {e}")
        checks.append(False)

print("\n" + "="*50)
if all(checks):
    print("✓ All checks passed!")
    sys.exit(0)
else:
    print("✗ Some checks failed")
    sys.exit(1)
