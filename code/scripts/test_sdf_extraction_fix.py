#!/usr/bin/env python
"""
Test script to verify SDF extraction works with frame index fix.
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '.')

from src.hold.hold import HOLD
from pytorch_lightning import Trainer

print("="*70)
print("TEST: SDF Extraction with Frame Index Fix")
print("="*70)

# ================================================================
# Step 1: Load checkpoint and recreate model
# ================================================================

print("\n[1] Loading Stage 2 checkpoint...")

ckpt_path = 'logs/stage2_final.ckpt'
ckpt = torch.load(ckpt_path, map_location='cpu')

print(f"✓ Checkpoint loaded: epoch {ckpt['epoch']}, step {ckpt['global_step']}")

# Extract hyperparameters
from omegaconf import OmegaConf
if 'hyper_parameters' in ckpt:
    hparams = ckpt['hyper_parameters']
elif 'hparams' in ckpt:
    hparams = ckpt['hparams']
else:
    print("⚠️  No hyperparameters found, using default config")
    hparams = OmegaConf.load('confs/ghop_stage2_hold_MC1_ho3d.yaml')

# Create model
print("\n[2] Instantiating HOLD model...")
try:
    model = HOLD(**hparams)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.eval()
    model = model.cuda() if torch.cuda.is_available() else model
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    sys.exit(1)

# ================================================================
# Step 2: Create test batch with frame index
# ================================================================

print("\n[3] Creating test batch...")

device = next(model.parameters()).device

batch = {
    'idx': torch.tensor([[0], [1]], device=device),  # Frame 0 and 1
    'uv': torch.randn(2, 128, 2, device=device),
    'intrinsics': torch.eye(4, device=device).unsqueeze(0).expand(2, -1, -1),
    'extrinsics': torch.eye(4, device=device).unsqueeze(0).expand(2, -1, -1),
}

print(f"✓ Test batch created: {len(batch['idx'])} samples")
print(f"  Frame indices: {batch['idx'].squeeze().cpu().numpy()}")

# ================================================================
# Step 3: Test SDF grid extraction
# ================================================================

print("\n[4] Testing SDF grid extraction...")

try:
    with torch.no_grad():
        sdf_grid = model._extract_sdf_grid_from_nodes(batch, resolution=32)
    
    print(f"✓ SDF extraction succeeded!")
    print(f"  SDF grid shape: {sdf_grid.shape}")
    print(f"  SDF value range: [{sdf_grid.min().item():.4f}, {sdf_grid.max().item():.4f}]")
    print(f"  SDF mean: {sdf_grid.mean().item():.4f}")
    print(f"  SDF std: {sdf_grid.std().item():.4f}")
    
    # Check for zero crossings (sign changes)
    positive = (sdf_grid > 0).sum().item()
    negative = (sdf_grid < 0).sum().item()
    zero_crossings = min(positive, negative)
    
    print(f"\n  Sign distribution:")
    print(f"    Positive values: {positive:,} ({positive/sdf_grid.numel()*100:.1f}%)")
    print(f"    Negative values: {negative:,} ({negative/sdf_grid.numel()*100:.1f}%)")
    
    if zero_crossings > 0:
        print(f"\n  ✅ ZERO CROSSINGS DETECTED! ({zero_crossings:,} potential surface voxels)")
        print(f"     Marching Cubes should succeed!")
        test_passed = True
    else:
        print(f"\n  ❌ NO ZERO CROSSINGS - all values have same sign")
        print(f"     Marching Cubes will still fail")
        test_passed = False
        
except Exception as e:
    print(f"❌ SDF extraction failed: {e}")
    import traceback
    traceback.print_exc()
    test_passed = False
    sys.exit(1)

# ================================================================
# Step 4: Test Marching Cubes extraction
# ================================================================

print("\n[5] Testing Marching Cubes mesh extraction...")

try:
    with torch.no_grad():
        obj_verts_list, obj_faces_list = model._extract_object_mesh_from_sdf(batch)
    
    success_count = sum(1 for verts in obj_verts_list if verts.shape[0] > 0)
    
    print(f"✓ Marching Cubes completed!")
    print(f"  Successful extractions: {success_count}/{len(obj_verts_list)}")
    
    for i, (verts, faces) in enumerate(zip(obj_verts_list, obj_faces_list)):
        if verts.shape[0] > 0:
            print(f"  Frame {i}: {verts.shape[0]} vertices, {faces.shape[0]} faces ✅")
        else:
            print(f"  Frame {i}: Empty mesh ❌")
    
    if success_count > 0:
        print(f"\n  ✅ MESH EXTRACTION SUCCEEDED for {success_count} frames!")
        print(f"     The fix is WORKING!")
        test_passed = True
    else:
        print(f"\n  ❌ All meshes are empty - fix incomplete")
        test_passed = False
        
except Exception as e:
    print(f"❌ Marching Cubes failed: {e}")
    import traceback
    traceback.print_exc()
    test_passed = False

# ================================================================
# Summary
# ================================================================

print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)

if test_passed:
    print("✅ ALL TESTS PASSED")
    print("   The frame index fix is working correctly!")
    print("   Phase 4 contact loss should now activate during training.")
else:
    print("❌ TESTS FAILED")
    print("   Additional debugging needed.")
    print("\nPossible issues:")
    print("  1. Fix not fully applied (check idx_expanded usage)")
    print("  2. forward_sdf signature mismatch")
    print("  3. Object SDF still not properly conditioned on frame")

print("="*70)

sys.exit(0 if test_passed else 1)
