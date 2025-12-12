#!/usr/bin/env python3
import yaml

print("="*70)
print("LOSS COMPARISON: STAGE 1 vs STAGE 2 vs STAGE 3")
print("="*70)

stages = {
    'stage1': 'confs/stage1_hold_MC1_ho3d_sds_from_official.yaml',
    'stage2': 'confs/stage2_hold_MC1_ho3d_sds_from_official.yaml',
    'stage3': 'confs/stage3_hold_MC1_ho3d_sds_from_official.yaml'
}

all_losses = {}

for stage_name, config_path in stages.items():
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        losses = {}
        
        # Base losses
        if 'loss' in config:
            for key, val in config['loss'].items():
                if key.startswith('w_'):
                    losses[key] = val
        
        # Phase-specific losses
        for phase in ['phase3', 'phase4', 'phase5']:
            if phase in config and config[phase].get('enabled', False):
                if phase == 'phase3':
                    losses['w_sds'] = config[phase].get('w_sds', 'N/A')
                elif phase == 'phase4':
                    losses['w_contact'] = config[phase].get('w_contact', 'N/A')
                elif phase == 'phase5':
                    losses['w_temporal'] = config[phase].get('w_temporal', 'N/A')
        
        all_losses[stage_name] = losses
    except FileNotFoundError:
        all_losses[stage_name] = {'error': 'Config not found'}

# Print comparison table
print("\n{:<20s} {:<15s} {:<15s} {:<15s}".format("Loss Type", "Stage 1", "Stage 2", "Stage 3"))
print("-" * 70)

all_loss_names = set()
for losses in all_losses.values():
    all_loss_names.update(losses.keys())

for loss_name in sorted(all_loss_names):
    if loss_name == 'error':
        continue
    
    stage1_val = all_losses.get('stage1', {}).get(loss_name, '-')
    stage2_val = all_losses.get('stage2', {}).get(loss_name, '-')
    stage3_val = all_losses.get('stage3', {}).get(loss_name, '-')
    
    print("{:<20s} {:<15s} {:<15s} {:<15s}".format(
        loss_name, str(stage1_val), str(stage2_val), str(stage3_val)
    ))

print("\n" + "="*70)
