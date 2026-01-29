import torch
import numpy as np

# Load checkpoint at different stages
ckpts = {
    'step_8K': 'logs/640c1f867/checkpoints/steps/stepstep=07999.ckpt',
    'step_15K': 'logs/640c1f867/checkpoints/steps/stepstep=14999.ckpt', 
    'step_30K': 'logs/640c1f867/checkpoints/steps/stepstep=29999.ckpt',
    'step_60K': 'logs/640c1f867/checkpoints/steps/stepstep=59999.ckpt',
}

for name, path in ckpts.items():
    try:
        ckpt = torch.load(path, map_location='cpu')
        
        # Find v3d_cano (object vertices)
        v3d_keys = [k for k in ckpt['state_dict'].keys() if 'v3d' in k and 'cano' in k]
        
        if v3d_keys:
            v3d = ckpt['state_dict'][v3d_keys[0]]
            print(f'{name}: v3d_cano shape={v3d.shape}, mean={v3d.mean():.4f}, std={v3d.std():.4f}, range=[{v3d.min():.4f}, {v3d.max():.4f}]')
        
        # Check optimizer state for v3d_cano
        if 'optimizer_states' in ckpt and len(ckpt['optimizer_states']) > 0:
            opt_state = ckpt['optimizer_states'][0]
            if 'state' in opt_state:
                # Look for v3d momentum
                for param_id, state in opt_state['state'].items():
                    if 'exp_avg' in state:
                        exp_avg = state['exp_avg']
                        if exp_avg.shape == v3d.shape:
                            momentum_norm = exp_avg.norm().item()
                            print(f'  → Optimizer momentum norm: {momentum_norm:.6f}')
                            break
    except Exception as e:
        print(f'{name}: Error - {e}')

print('\n=== Analysis ===')
print('If std decreases over time → geometry collapsing')
print('If momentum norm is tiny → gradients not flowing')
