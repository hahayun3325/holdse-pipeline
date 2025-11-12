#!/usr/bin/env python
"""Check if implicit networks produce meaningful SDF"""

import torch
import sys

sys.path.insert(0, '.')
sys.path.insert(0, '../common')

import os

os.environ['COMET_MODE'] = 'disabled'

from src.hold.hold import HOLD
from omegaconf import OmegaConf

config_path = 'confs/ghop_production_chunked_20251027_131408.yaml'
checkpoint_path = 'logs/6aaaf5002/checkpoints/last.ckpt'

opt = OmegaConf.load(config_path)
if not hasattr(opt.model, 'scene_bounding_sphere'):
    opt.model.scene_bounding_sphere = 3.0


class Args:
    case = 'ghop_bottle_1'
    n_images = 71
    infer_ckpt = checkpoint_path
    ckpt_p = checkpoint_path
    barf_s = 0
    barf_e = 0
    no_barf = True
    shape_init = ""


args = Args()

print("Loading model...")
model = HOLD(opt, args)
model.phase3_enabled = False
model.phase4_enabled = False
model.phase5_enabled = False

ckpt = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(ckpt['state_dict'], strict=False)
model.cuda()
model.eval()

print("\n" + "=" * 70)
print("TESTING IMPLICIT NETWORK OUTPUTS")
print("=" * 70)

# Test with random points
test_points = torch.randn(1, 100, 3).cuda() * 0.5  # Points near origin

print("\nTesting hand implicit network...")
if 'right' in model.model.nodes:
    hand_node = model.model.nodes['right']
    with torch.no_grad():
        # Create dummy conditioning
        cond = {'pose': torch.zeros(1, 45).cuda()}
        deform_info = {'cond': cond}

        try:
            sdf_output = hand_node.implicit_network(test_points, deform_info)
            print(f"  Hand SDF output shape: {sdf_output.shape}")
            print(f"  Min: {sdf_output.min().item():.4f}, Max: {sdf_output.max().item():.4f}")
            print(f"  Mean: {sdf_output.mean().item():.4f}, Std: {sdf_output.std().item():.4f}")

            if sdf_output.std().item() < 0.01:
                print(f"  ⚠️  WARNING: SDF output is nearly constant!")
        except Exception as e:
            print(f"  ❌ Error: {e}")

print("\nTesting object implicit network...")
if 'object' in model.model.nodes:
    obj_node = model.model.nodes['object']
    with torch.no_grad():
        try:
            sdf_output = obj_node.implicit_network(test_points, None)
            print(f"  Object SDF output shape: {sdf_output.shape}")
            print(f"  Min: {sdf_output.min().item():.4f}, Max: {sdf_output.max().item():.4f}")
            print(f"  Mean: {sdf_output.mean().item():.4f}, Std: {sdf_output.std().item():.4f}")

            if sdf_output.std().item() < 0.01:
                print(f"  ⚠️  WARNING: SDF output is nearly constant!")
        except Exception as e:
            print(f"  ❌ Error: {e}")

print("\n" + "=" * 70)