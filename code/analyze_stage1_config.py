#!/usr/bin/env python3
import yaml

config_path = "confs/stage1_hold_MC1_ho3d_sds_from_official.yaml"

print("="*70)
print("STAGE 1 CONFIGURATION ANALYSIS")
print("="*70)

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

print("\n1. PHASE STATUS (All should be disabled for Stage 1):")
print("-" * 50)
for phase in ['phase2', 'phase3', 'phase4', 'phase5']:
    if phase in config:
        enabled = config[phase].get('enabled', False)
        status = "✓ ENABLED" if enabled else "✗ DISABLED"
        print(f"   {phase}: {status}")
    else:
        print(f"   {phase}: Not configured")

print("\n2. TRAINING CONFIGURATION:")
print("-" * 50)
if 'training' in config:
    print(f"   Epochs: {config['training'].get('num_epochs', 'N/A')}")
    print(f"   Max steps: {config['training'].get('max_steps', 'N/A')}")
    print(f"   Eval frequency: {config['training'].get('eval_every_epoch', 'N/A')}")
    print(f"   Log frequency: {config['training'].get('log_every', 'N/A')}")
    print(f"   Gradient clip: {config['training'].get('gradient_clip', 'N/A')}")

print("\n3. BASE LOSS WEIGHTS (Stage 1 uses only these):")
print("-" * 50)
if 'loss' in config:
    for key, val in sorted(config['loss'].items()):
        if key.startswith('w_'):
            loss_name = key[2:].upper()
            print(f"   {loss_name:15s}: {val}")

print("\n4. NETWORK ARCHITECTURES:")
print("-" * 50)

# Hand Implicit Network
if 'model' in config and 'implicit_network' in config['model']:
    net = config['model']['implicit_network']
    print(f"\n   A. Hand Implicit Network:")
    print(f"      - Input dims: {net.get('d_in', 'N/A')}")
    print(f"      - Output dims: {net.get('d_out', 'N/A')} (SDF)")
    print(f"      - Hidden layers: {net.get('dims', 'N/A')}")
    print(f"      - Skip connections: {net.get('skip_in', 'N/A')}")
    print(f"      - Multires encoding: {net.get('multires', 'N/A')}")
    print(f"      - Conditioning: {net.get('cond', 'N/A')}")
    print(f"      - Weight norm: {net.get('weight_norm', 'N/A')}")

# Hand Rendering Network
if 'model' in config and 'rendering_network' in config['model']:
    net = config['model']['rendering_network']
    print(f"\n   B. Hand Rendering Network:")
    print(f"      - Input dims: {net.get('d_in', 'N/A')}")
    print(f"      - Output dims: {net.get('d_out', 'N/A')} (RGB)")
    print(f"      - Hidden layers: {net.get('dims', 'N/A')}")
    print(f"      - Feature size: {net.get('feature_vector_size', 'N/A')}")
    print(f"      - Mode: {net.get('mode', 'N/A')}")
    print(f"      - View encoding: {net.get('multires_view', 'N/A')}")

# Background Networks (if present)
if 'model' in config and 'bg_implicit_network' in config['model']:
    net = config['model']['bg_implicit_network']
    print(f"\n   C. Background Implicit Network:")
    print(f"      - Input dims: {net.get('d_in', 'N/A')}")
    print(f"      - Output dims: {net.get('d_out', 'N/A')}")
    print(f"      - Hidden layers: {net.get('dims', 'N/A')}")
    print(f"      - Conditioning: {net.get('cond', 'N/A')}")
    print(f"      - Frame encoding: {net.get('dim_frame_encoding', 'N/A')}")

print("\n5. DATASET CONFIGURATION:")
print("-" * 50)
if 'dataset' in config:
    print(f"   Path: {config['dataset'].get('dataset_path', 'N/A')}")
    print(f"   Sequence: {config['dataset'].get('seq_name', 'N/A')}")
    if 'train' in config['dataset']:
        print(f"   Batch size: {config['dataset']['train'].get('batch_size', 'N/A')}")
        print(f"   Num workers: {config['dataset']['train'].get('num_workers', 'N/A')}")
        print(f"   Shuffle: {config['dataset']['train'].get('shuffle', 'N/A')}")

print("\n6. OPTIMIZER:")
print("-" * 50)
if 'optimizer' in config:
    print(f"   Type: {config['optimizer'].get('type', 'N/A')}")
    print(f"   Learning rate: {config['optimizer'].get('lr', 'N/A')}")
    print(f"   Weight decay: {config['optimizer'].get('weight_decay', 'N/A')}")
    print(f"   Betas: {config['optimizer'].get('betas', 'N/A')}")

print("\n" + "="*70)
print("STAGE 1 INTERPRETATION:")
print("="*70)
print("""
Stage 1 trains the BASE HOLD model without any GHOP enhancements:
  
  ✓ Trains: Hand implicit network (SDF)
  ✓ Trains: Hand rendering network (RGB)
  ✓ Trains: Object implicit network (SDF)
  ✓ Trains: Background implicit network (optional)
  
  ✓ Optimizes: MANO hand parameters (pose, shape, translation)
  ✓ Optimizes: Object pose parameters
  
  ✓ Uses losses: RGB, Mask, Eikonal, Smoothness
  ✗ No GHOP: SDS, Contact, Temporal losses are disabled
  
  → Output: Pretrained HOLD model ready for Stage 2 GHOP integration
""")
print("="*70)
