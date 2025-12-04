#!/usr/bin/env python3
"""Debug version: Print predicted parameters."""
import sys
sys.path.insert(0, '.')
import torch
from pathlib import Path
from src.hold.hold import HOLD
from src.datasets.utils import create_dataset
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

def thing2dev(data, device='cuda'):
    if isinstance(data, dict):
        return {k: thing2dev(v, device) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data

# Load checkpoint
ckpt_path = 'logs/e1c95c0d0/checkpoints/last.ckpt'
config_path = 'confs/stage1_hold_MC1_ho3d_8layer_implicit.yaml'

print(f"Loading: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location='cpu')

opt = OmegaConf.load(config_path)
if not hasattr(opt.model, 'scene_bounding_sphere'):
    opt.model.scene_bounding_sphere = 3.0

class Args:
    case = 'hold_MC1_ho3d'
    n_images = 144
    num_sample = 2048
    no_vis = False
    render_downsample = 1
    freeze_pose = False
    experiment = 'debug'
    log_every = 100
    log_dir = 'logs/debug'
    barf_s = 0
    barf_e = 0
    no_barf = True
    shape_init = ""
    exp_key = 'debug'
    debug = False
    agent_id = -1
    offset = 1
    num_workers = 0

args = Args()

print("Initializing model...")
model = HOLD(opt, args)
model.load_state_dict(ckpt['state_dict'], strict=False)
model.cuda()
model.eval()

# Create dataset
test_config = OmegaConf.create({
    'type': 'test', 'batch_size': 1, 'drop_last': False,
    'shuffle': False, 'num_workers': 0, 'pixel_per_batch': 512,
})
full_dataset = create_dataset(test_config, args)
base_dataset = full_dataset.dataset if hasattr(full_dataset, 'dataset') else full_dataset
dataloader = DataLoader(base_dataset, batch_size=1, shuffle=False)

print(f"\nExtracting first 5 frames to check parameter diversity...\n")

param_history = []

with torch.no_grad():
    for idx, batch in enumerate(dataloader):
        if idx >= 5:
            break

        batch_cuda = thing2dev(batch, 'cuda')
        output = model.validation_step(batch_cuda)

        # Detect keys
        beta_key = 'right.betas' if 'right.betas' in output else 'right_betas'
        pose_key = 'right.pose' if 'right.pose' in output else 'right_pose'

        betas = output[beta_key].detach().cpu()
        pose = output[pose_key].detach().cpu()

        param_history.append({
            'frame': idx,
            'betas_mean': betas.mean().item(),
            'betas_std': betas.std().item(),
            'pose_mean': pose.mean().item(),
            'pose_std': pose.std().item(),
        })

        print(f"Frame {idx}:")
        print(f"  betas: mean={betas.mean().item():+.6f}, std={betas.std().item():.6f}")
        print(f"  pose:  mean={pose.mean().item():+.6f}, std={pose.std().item():.6f}")

# Check if all frames have identical parameters
print("\n" + "=" * 70)
print("PARAMETER DIVERSITY CHECK")
print("=" * 70)

all_betas_mean = [p['betas_mean'] for p in param_history]
all_pose_mean = [p['pose_mean'] for p in param_history]

import numpy as np
betas_variance = np.var(all_betas_mean)
pose_variance = np.var(all_pose_mean)

print(f"\nVariance across frames:")
print(f"  betas_mean variance: {betas_variance:.10f}")
print(f"  pose_mean variance:  {pose_variance:.10f}")

if betas_variance < 1e-8 and pose_variance < 1e-8:
    print("\n❌ PROBLEM: All frames predict IDENTICAL parameters!")
    print("   The network is returning fixed values regardless of input.")
else:
    print("\n✅ Parameters vary across frames (as expected)")