#!/usr/bin/env python
"""Diagnose Background module NaN production - FIXED VERSION"""

import torch
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '../common')

import os
os.environ['COMET_MODE'] = 'disabled'

from src.hold.hold import HOLD
from src.datasets.utils import create_dataset
from thing import thing2dev
from omegaconf import OmegaConf

config_path = 'confs/ghop_production_chunked_20251027_131408.yaml'
checkpoint_path = 'logs/ad1f0073b/checkpoints/last.ckpt'

opt = OmegaConf.load(config_path)
if not hasattr(opt.model, 'scene_bounding_sphere'):
    opt.model.scene_bounding_sphere = 3.0

class Args:
    case = 'ghop_bottle_1'
    n_images = 71
    num_sample = 2048
    infer_ckpt = checkpoint_path
    ckpt_p = checkpoint_path
    no_vis = False
    render_downsample = 2
    freeze_pose = False
    experiment = 'test'
    log_every = 10
    log_dir = 'logs/test'
    barf_s = 0
    barf_e = 0
    no_barf = True
    shape_init = ""
    exp_key = 'test'
    debug = False

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

print("\n" + "="*70)
print("BACKGROUND MODULE DIAGNOSIS - FIXED VERSION")
print("="*70)

# Inspect background module structure
print("\n1. Background Module Attributes:")
print(f"   Type: {type(model.model.background)}")

# Check if background has expected networks
if hasattr(model.model.background, 'bg_implicit_network'):
    print(f"   Has bg_implicit_network: ✅")
else:
    print(f"   Has bg_implicit_network: ❌")

if hasattr(model.model.background, 'bg_rendering_network'):
    print(f"   Has bg_rendering_network: ✅")
else:
    print(f"   Has bg_rendering_network: ❌")

if hasattr(model.model.background, 'frame_latent_encoder'):
    print(f"   Has frame_latent_encoder: ✅")
    print(f"       Num embeddings: {model.model.background.frame_latent_encoder.num_embeddings}")
    print(f"       Embedding dim: {model.model.background.frame_latent_encoder.embedding_dim}")
else:
    print(f"   Has frame_latent_encoder: ❌")

# Check background weights
print("\n2. Background Weights Status:")
bg_state_dict = model.model.background.state_dict()
nan_params = []
inf_params = []
ok_params = []

for key, param in bg_state_dict.items():
    has_nan = torch.isnan(param).any().item()
    has_inf = torch.isinf(param).any().item()

    if has_nan:
        nan_params.append(key)
    elif has_inf:
        inf_params.append(key)
    else:
        ok_params.append(key)

print(f"   Total parameters: {len(bg_state_dict)}")
print(f"   ✅ OK:  {len(ok_params)}")
print(f"   ⚠️ Inf: {len(inf_params)}")
print(f"   ❌ NaN: {len(nan_params)}")

if nan_params:
    print(f"\n   Parameters with NaN:")
    for key in nan_params[:5]:
        print(f"     - {key}")

# Sample first few parameters
print(f"\n   Sample parameter values:")
for key, param in list(bg_state_dict.items())[:5]:
    has_nan = torch.isnan(param).any().item()
    has_inf = torch.isinf(param).any().item()
    status = "❌ NaN" if has_nan else ("⚠️ Inf" if has_inf else "✅ OK")
    print(f"     {key[:60]:60s}: {status}")
    if not has_nan and not has_inf:
        print(f"       shape={param.shape}, min={param.min().item():.6f}, max={param.max().item():.6f}")

# Load dataset and test rendering using validation_step (correct method)
print("\n3. Testing Background Rendering via validation_step:")
val_config = opt.dataset.val if hasattr(opt.dataset, 'val') else opt.dataset.valid
val_dataset = create_dataset(val_config, args)

for batch in val_dataset:
    batch_cuda = thing2dev(batch, 'cuda')

    print(f"\n   Batch info:")
    print(f"     idx: {batch_cuda.get('idx')}")
    print(f"     Has 'index': {('index' in batch_cuda)}")

    with torch.no_grad():
        # Use validation_step which properly expands parameters
        print("\n   Calling validation_step (includes parameter expansion)...")
        try:
            output = model.validation_step(batch_cuda)
            print("   ✅ Validation step completed")

            # Now manually call background to see inputs
            print("\n   Extracting foreground dict for background input inspection...")

            # Get the actual batch that was used (with expanded params)
            fg_dict = model.model.forward_fg(batch_cuda)

            print("\n   Background input parameters:")
            bg_input_keys = ['bg_weights', 'ray_dirs', 'cam_loc', 'bg_z_vals', 'index']
            for key in bg_input_keys:
                if key in fg_dict:
                    tensor = fg_dict[key]
                    has_nan = torch.isnan(tensor).any().item()
                    has_inf = torch.isinf(tensor).any().item()
                    status = "❌ NaN" if has_nan else ("⚠️ Inf" if has_inf else "✅ OK")
                    print(f"     {key:15s}: {status:10s} shape={tensor.shape}, dtype={tensor.dtype}")
                    if not has_nan and not has_inf:
                        print(f"                    min={tensor.min().item():.6f}, max={tensor.max().item():.6f}, mean={tensor.mean().item():.6f}")
                else:
                    print(f"     {key:15s}: ❌ MISSING")

            # Call background directly
            print("\n   Calling background module directly...")
            bg_dict = model.model.background(
                fg_dict['bg_weights'],
                fg_dict['ray_dirs'],
                fg_dict['cam_loc'],
                fg_dict['bg_z_vals'],
                fg_dict.get('index'),  # Use .get() to handle missing index
            )

            print("   ✅ Background forward pass completed")

            print("\n   Background output analysis:")
            for key in ['bg_rgb', 'bg_rgb_only', 'bg_semantics']:
                if key in bg_dict:
                    tensor = bg_dict[key]
                    has_nan = torch.isnan(tensor).any().item()
                    has_inf = torch.isinf(tensor).any().item()
                    status = "❌ NaN" if has_nan else ("⚠️ Inf" if has_inf else "✅ OK")
                    print(f"     {key:15s}: {status:10s} shape={tensor.shape}")

                    if has_nan:
                        nan_count = torch.isnan(tensor).sum().item()
                        total = tensor.numel()
                        print(f"                    NaN pixels: {nan_count}/{total} ({100*nan_count/total:.1f}%)")

                        # Check if entire tensor or partial
                        if nan_count == total:
                            print(f"                    ⚠️ ENTIRE TENSOR IS NaN!")
                    elif not has_inf:
                        print(f"                    min={tensor.min().item():.6f}, max={tensor.max().item():.6f}")

            # Check if idx/index is being passed correctly
            print("\n   Index parameter check:")
            print(f"     'index' in fg_dict: {('index' in fg_dict)}")
            if 'index' in fg_dict:
                print(f"     fg_dict['index']: {fg_dict['index']}")
            print(f"     'idx' in batch_cuda: {('idx' in batch_cuda)}")
            if 'idx' in batch_cuda:
                print(f"     batch_cuda['idx']: {batch_cuda['idx']}")

        except Exception as e:
            print(f"   ❌ Error during execution: {e}")
            import traceback
            traceback.print_exc()

    break

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)

print("\n4. Summary:")
print("   - Check if 'index' parameter is None or missing")
print("   - If index is None, background returns torch.ones which may cause shape mismatch")
print("   - Check background output shapes match expected pixel count")
