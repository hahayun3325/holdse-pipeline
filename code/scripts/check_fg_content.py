#!/usr/bin/env python
"""Check foreground rendering output"""

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

print("\n" + "=" * 70)
print("CHECKING FOREGROUND RENDERING CONTENT")
print("=" * 70)

val_config = opt.dataset.val if hasattr(opt.dataset, 'val') else opt.dataset.valid
val_dataset = create_dataset(val_config, args)

for batch in val_dataset:
    batch_cuda = thing2dev(batch, 'cuda')

    with torch.no_grad():
        output = model.validation_step(batch_cuda)

        print("\nComponent analysis:")

        components = {
            'rgb': output.get('rgb'),
            'fg_rgb.vis': output.get('fg_rgb.vis'),
            'right.fg_rgb.vis': output.get('right.fg_rgb.vis'),
            'object.fg_rgb.vis': output.get('object.fg_rgb.vis'),
            'bg_rgb_only': output.get('bg_rgb_only'),
            'mask_prob': output.get('mask_prob'),
            'right.mask_prob': output.get('right.mask_prob'),
            'object.mask_prob': output.get('object.mask_prob'),
        }

        for name, tensor in components.items():
            if tensor is not None:
                print(f"\n{name}:")
                print(f"  shape: {tensor.shape}")
                print(
                    f"  min: {tensor.min().item():.4f}, max: {tensor.max().item():.4f}, mean: {tensor.mean().item():.4f}")
                print(f"  has_nan: {torch.isnan(tensor).any().item()}")

                # Check if tensor has variation
                std = tensor.std().item()
                print(f"  std: {std:.4f}")

                if std < 0.01:
                    print(f"  ⚠️  WARNING: Very low variation - nearly constant!")

                # Count non-zero pixels (for masks)
                if 'mask' in name:
                    non_zero = (tensor > 0.5).sum().item()
                    total = tensor.numel()
                    print(f"  pixels > 0.5: {non_zero}/{total} ({100 * non_zero / total:.1f}%)")

    break

print("\n" + "=" * 70)