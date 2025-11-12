#!/usr/bin/env python
"""Working RGB pipeline trace with correct import"""

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

# Store original forward method
original_rendering_forward = None

def trace_rendering_forward(self, points, normals, view_dirs, body_pose, feature_vectors, frame_latent_code=None):
    """Intercepted rendering network forward"""

    print(f"\n[RenderingNet.forward] Called (mode={self.mode}):")

    # Check feature vectors
    if feature_vectors is not None:
        print(f"  feature_vectors: shape={feature_vectors.shape}")
        print(f"                   min={feature_vectors.min():.4f}, max={feature_vectors.max():.4f}")
        print(f"                   mean={feature_vectors.mean():.4f}, std={feature_vectors.std():.4f}")
        if feature_vectors.std() < 0.01:
            print(f"                   ⚠️  Features nearly constant!")

    # Call original
    output = original_rendering_forward(self, points, normals, view_dirs, body_pose, feature_vectors, frame_latent_code)

    # Check output
    print(f"  output RGB:      shape={output.shape}")
    print(f"                   min={output.min():.4f}, max={output.max():.4f}")
    print(f"                   mean={output.mean():.4f}, std={output.std():.4f}")
    if output.std() < 0.01:
        print(f"                   ❌ Output nearly constant! Rendering network broken!")

    return output

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
    num_sample = 2048

args = Args()

print("="*70)
print("RGB PIPELINE TRACER")
print("="*70)

print("\nLoading model...")
model = HOLD(opt, args)
model.phase3_enabled = False
model.phase4_enabled = False
model.phase5_enabled = False

ckpt = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(ckpt['state_dict'], strict=False)
model.cuda()
model.eval()

# Patch RenderingNet
from src.networks.texture_net import RenderingNet
original_rendering_forward = RenderingNet.forward
RenderingNet.forward = trace_rendering_forward

print("\nLoading dataset...")
val_config = opt.dataset.val if hasattr(opt.dataset, 'val') else opt.dataset.valid
val_dataset = create_dataset(val_config, args)

print("\n" + "="*70)
print("RENDERING WITH TRACING")
print("="*70)

for batch in val_dataset:
    batch_cuda = thing2dev(batch, 'cuda')

    with torch.no_grad():
        output = model.validation_step(batch_cuda)

        print("\n" + "="*70)
        print("FINAL COMPOSITED RGB:")
        print(f"  min={output['rgb'].min():.4f}, max={output['rgb'].max():.4f}")
        print(f"  mean={output['rgb'].mean():.4f}, std={output['rgb'].std():.4f}")
        print("="*70)

    break

print("\n" + "="*70)
print("TRACE COMPLETE")
print("="*70)