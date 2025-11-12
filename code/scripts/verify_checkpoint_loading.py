#!/usr/bin/env python
"""Verify if checkpoint loads correctly into model"""

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
    shape_init = ''
    freeze_pose = False
    experiment = 'test'  # ← FIX: Add missing attribute
    log_every = 10
    log_dir = 'logs/test'
    no_vis = False
    render_downsample = 2
    num_sample = 2048
    exp_key = 'test'
    debug = False


args = Args()

print('=' * 70)
print('CHECKPOINT LOADING VERIFICATION')
print('=' * 70)

print('\n1. Creating model...')
model = HOLD(opt, args)
model.phase3_enabled = False
model.phase4_enabled = False
model.phase5_enabled = False

print('\n2. Model node names:')
for node_name in model.model.nodes.keys():
    print(f'   {node_name}')

print('\n3. Loading checkpoint...')
ckpt = torch.load(checkpoint_path, map_location='cpu')

print(f'\n4. Checkpoint contains {len(ckpt["state_dict"])} keys')
print('   Sample checkpoint keys:')
for k in list(ckpt['state_dict'].keys())[:5]:
    print(f'     {k}')

print('\n5. Attempting to load...')
missing_keys, unexpected_keys = model.load_state_dict(ckpt['state_dict'], strict=False)

print(f'\n6. Load results:')
print(f'   Missing keys: {len(missing_keys)}')
if len(missing_keys) > 0:
    print('   Sample missing keys:')
    for k in missing_keys[:10]:
        print(f'     {k}')

print(f'\n   Unexpected keys: {len(unexpected_keys)}')
if len(unexpected_keys) > 0:
    print('   Sample unexpected keys:')
    for k in unexpected_keys[:10]:
        print(f'     {k}')

# Check if rendering network was actually loaded
print('\n7. Checking rendering network weights:')
right_node = model.model.nodes['right']

# Get first rendering network parameter from model
model_params = list(right_node.rendering_network.parameters())
if len(model_params) > 0:
    first_model_param = model_params[0]
    print(f'   Model rendering network first param:')
    print(f'     shape: {first_model_param.shape}')
    print(f'     mean: {first_model_param.mean().item():.6f}')
    print(f'     std: {first_model_param.std().item():.6f}')

# Get corresponding parameter from checkpoint
ckpt_render_keys = [k for k in ckpt['state_dict'].keys()
                    if 'rendering_network' in k and 'right' in k]
if ckpt_render_keys:
    first_ckpt_key = ckpt_render_keys[0]
    first_ckpt_param = ckpt['state_dict'][first_ckpt_key]

    print(f'\n   Checkpoint rendering network first param ({first_ckpt_key}):')
    print(f'     shape: {first_ckpt_param.shape}')
    print(f'     mean: {first_ckpt_param.mean().item():.6f}')
    print(f'     std: {first_ckpt_param.std().item():.6f}')

    # Compare
    mean_diff = abs(first_model_param.mean().item() - first_ckpt_param.mean().item())
    print(f'\n8. Comparison:')
    print(f'   Mean difference: {mean_diff:.6f}')

    if mean_diff < 1e-5:
        print('   ✅ MATCH! Rendering network loaded correctly!')
    else:
        print('   ❌ MISMATCH! Rendering network NOT loaded!')
        print('      This explains the gray images.')
else:
    print('   ❌ No rendering network keys found in checkpoint!')

print('\n' + '=' * 70)