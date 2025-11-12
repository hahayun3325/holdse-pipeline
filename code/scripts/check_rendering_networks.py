#!/usr/bin/env python
"""Check if rendering networks produce varied colors"""

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
checkpoint_path = 'logs/6aaaf5002/checkpoints/last.ckpt'

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
    freeze_pose = False  # ← FIX: Add missing attribute
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

print("\n" + "=" * 70)
print("TESTING RENDERING NETWORKS WITH ACTUAL DATA")
print("=" * 70)

val_config = opt.dataset.val if hasattr(opt.dataset, 'val') else opt.dataset.valid
val_dataset = create_dataset(val_config, args)

for batch in val_dataset:
    batch_cuda = thing2dev(batch, 'cuda')

    with torch.no_grad():
        # Get foreground dict (contains all intermediate rendering outputs)
        print("\nRunning forward_fg to get intermediate outputs...")
        try:
            fg_dict = model.model.forward_fg(batch_cuda)
            print("✅ forward_fg succeeded\n")

            # Check SDF outputs
            print("SDF Values:")
            for node_name, node in model.model.nodes.items():
                if f'{node_name}.sdf' in fg_dict:
                    sdf = fg_dict[f'{node_name}.sdf']
                    print(f"  {node_name}.sdf:")
                    print(f"    shape: {sdf.shape}")
                    print(f"    min: {sdf.min().item():.4f}, max: {sdf.max().item():.4f}")
                    print(f"    mean: {sdf.mean().item():.4f}, std: {sdf.std().item():.4f}")

                    if sdf.std().item() < 0.01:
                        print(f"    ⚠️ WARNING: SDF is nearly constant!")

            # Check feature vectors
            print("\nFeature Vectors:")
            for key in fg_dict.keys():
                if 'feature' in key.lower():
                    features = fg_dict[key]
                    print(f"  {key}:")
                    print(f"    shape: {features.shape}")
                    print(f"    std: {features.std().item():.4f}")

            # Check RGB outputs BEFORE compositing
            print("\nRGB Outputs (before composite):")
            for key in ['right.rgb', 'object.rgb']:
                if key in fg_dict:
                    rgb = fg_dict[key]
                    print(f"  {key}:")
                    print(f"    shape: {rgb.shape}")
                    print(f"    min: {rgb.min().item():.4f}, max: {rgb.max().item():.4f}")
                    print(f"    mean: {rgb.mean().item():.4f}, std: {rgb.std().item():.4f}")

                    if rgb.std().item() < 0.01:
                        print(f"    ⚠️ WARNING: RGB output is nearly constant!")
                        print(f"    This means the rendering network is broken!")

            # Check opacity/alpha values
            print("\nOpacity/Alpha:")
            for key in fg_dict.keys():
                if 'alpha' in key.lower() or 'opacity' in key.lower():
                    alpha = fg_dict[key]
                    print(f"  {key}:")
                    print(f"    min: {alpha.min().item():.4f}, max: {alpha.max().item():.4f}")
                    print(f"    mean: {alpha.mean().item():.4f}")

        except KeyError as e:
            print(f"❌ Error: {e}")
            print("This is the parameter expansion issue - trying validation_step instead...")

            output = model.validation_step(batch_cuda)
            print("\n✅ validation_step succeeded")
            print("\nChecking validation_step outputs:")

            for key in ['right.sdf', 'object.sdf']:
                if key in output:
                    sdf = output[key]
                    print(f"\n{key}:")
                    print(f"  min: {sdf.min().item():.4f}, max: {sdf.max().item():.4f}")
                    print(f"  std: {sdf.std().item():.4f}")

    break

print("\n" + "=" * 70)