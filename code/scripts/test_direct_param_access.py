#!/usr/bin/env python3
import torch
from pathlib import Path
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '.')
sys.path.insert(0, '..')

checkpoint_path = 'logs/176872f9f/checkpoints/last.ckpt'
ckpt = torch.load(checkpoint_path, map_location='cpu')

# Method 1: Load through HOLD model
from src.hold.hold import HOLD
from omegaconf import OmegaConf

opt = OmegaConf.load('confs/stage1_hold_MC1_ho3d.yaml')

class Args:
    case = 'hold_MC1_ho3d'
    n_images = 144
    infer_ckpt = checkpoint_path
    loading_from_checkpoint = True
    freeze_pose = False
    # ... other required args

args = Args()
model = HOLD(opt, args)
model.load_state_dict(ckpt['state_dict'], strict=False)
model.cuda()
model.eval()

# Test frame 0
frame_idx = 0

print("\n" + "="*70)
print("PARAMETER SOURCE COMPARISON")
print("="*70)

# From checkpoint directly
ckpt_pose = ckpt['state_dict']['model.nodes.right.params.pose.weight'][frame_idx]
print(f"\n1. Checkpoint (frame {frame_idx}):")
print(f"   Shape: {ckpt_pose.shape}")
print(f"   Mean: {ckpt_pose.mean():.6f}")
print(f"   First 5: {ckpt_pose[:5]}")

# From loaded model
model_pose = model.model.nodes['right'].params.pose.weight[frame_idx]
print(f"\n2. Loaded model (frame {frame_idx}):")
print(f"   Shape: {model_pose.shape}")
print(f"   Mean: {model_pose.mean():.6f}")
print(f"   First 5: {model_pose[:5]}")

# From node.params()
batch = {'idx': torch.tensor([[frame_idx]], device='cuda')}
params_dict = model.model.nodes['right'].params(batch['idx'])
print(f"\n3. node.params() returned keys: {list(params_dict.keys())}")
for key, val in params_dict.items():
    if isinstance(val, torch.Tensor):
        print(f"   {key}: shape={val.shape}, mean={val.mean():.6f}")

# From validation_step()
batch_full = {
    'idx': torch.tensor([[frame_idx]], device='cuda'),
    'uv': torch.rand(1, 128, 2, device='cuda'),
    'c2w': torch.eye(4, device='cuda').unsqueeze(0),
    'intrinsics': torch.eye(3, device='cuda').unsqueeze(0),
}
output = model.validation_step(batch_full)

print(f"\n4. validation_step() returned keys: {list(output.keys())[:20]}...")

# Find pose-related keys
pose_keys = [k for k in output.keys() if 'pose' in k.lower() and 'right' in k.lower()]
print(f"   Pose-related keys: {pose_keys}")

for key in pose_keys:
    val = output[key]
    if isinstance(val, torch.Tensor) and val.numel() > 0:
        print(f"   {key}: shape={val.shape}, mean={val.mean():.6f}")

print("="*70)