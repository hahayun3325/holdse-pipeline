import sys
sys.path.insert(0, '.')
import torch
from omegaconf import OmegaConf
from src.hold.hold import HOLD
import numpy as np

def check_ckpt(ckpt_path, config_path):
    print(f"\nChecking checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    # Check if parameters are in checkpoint
    params = ['model.nodes.right.server.mano_params.betas', 
              'model.nodes.right.server.mano_params.global_orient',
              'model.nodes.right.server.mano_params.hand_pose']
              
    for p in params:
        if p in ckpt['state_dict']:
            val = ckpt['state_dict'][p]
            print(f"  {p.split('.')[-1]}: mean={val.mean().item():.6f}, std={val.std().item():.6f}")
        else:
            print(f"  {p.split('.')[-1]}: NOT FOUND in checkpoint")

print("="*60)
print("CHECKPOINT PARAMETER ANALYSIS")
print("="*60)

check_ckpt('logs/140dc5c18/checkpoints/last.ckpt', 'confs/stage1_hold_MC1_ho3d.yaml')
check_ckpt('logs/e1c95c0d0/checkpoints/last.ckpt', 'confs/stage1_hold_MC1_ho3d_8layer_implicit.yaml')
