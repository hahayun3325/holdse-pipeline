# File: code/analyze_training_config.py
"""Analyze what was actually trained."""

import yaml
from pathlib import Path
import torch


def analyze_config():
    """Check training configuration."""
    config_path = Path('confs/ghop_production_bottle_1.yaml')

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 70)
    print("TRAINING CONFIGURATION ANALYSIS")
    print("=" * 70)

    # Check training settings
    training = config.get('training', {})
    print(f"\n📊 Training Settings:")
    print(f"  Epochs: {training.get('num_epochs', 'NOT SET')}")
    print(f"  Max steps: {training.get('max_steps', 'NOT SET')}")
    print(f"  Eval frequency: {training.get('eval_every_epoch', 'NOT SET')}")

    # Check loss weights
    print(f"\n🎯 Loss Configuration:")
    if 'loss' in config or 'losses' in config:
        losses = config.get('loss', config.get('losses', {}))
        for key, val in losses.items():
            print(f"  {key}: {val}")
    else:
        print("  ⚠️  No explicit loss configuration found!")
        print("  → Using default loss weights")

    # Check optimizer
    optimizer = config.get('optimizer', {})
    print(f"\n⚙️  Optimizer:")
    print(f"  Type: {optimizer.get('type', 'NOT SET')}")
    print(f"  LR: {optimizer.get('lr', 'NOT SET')}")

    # Check dataset
    dataset = config.get('dataset', {})
    train_dataset = dataset.get('train', {})
    print(f"\n📁 Dataset:")
    print(f"  Batch size: {train_dataset.get('batch_size', 'NOT SET')}")
    print(f"  Num workers: {train_dataset.get('num_workers', 'NOT SET')}")

    # Check model settings
    model = config.get('model', {})
    print(f"\n🏗️  Model Architecture:")
    if 'rendering_network' in model:
        rn = model['rendering_network']
        print(f"  Rendering network dims: {rn.get('dims', 'NOT SET')}")
        print(f"  Rendering network d_out: {rn.get('d_out', 'NOT SET')}")
    else:
        print("  ⚠️  Rendering network not configured!")

    # Analyze checkpoint
    print(f"\n💾 Checkpoint Analysis:")
    ckpt_path = Path('logs/b2e4b039a/checkpoints/last.ckpt')
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location='cpu')

        print(f"  Epoch: {ckpt.get('epoch', 'N/A')}")
        print(f"  Global step: {ckpt.get('global_step', 'N/A')}")

        # Check which networks have parameters
        state_dict = ckpt.get('state_dict', {})

        has_rendering = any('rendering' in k for k in state_dict.keys())
        has_implicit = any('implicit' in k for k in state_dict.keys())
        has_bg = any('bg_' in k or 'background' in k for k in state_dict.keys())

        print(f"\n  Networks in checkpoint:")
        print(f"    Implicit/Geometry: {has_implicit}")
        print(f"    Rendering: {has_rendering}")
        print(f"    Background: {has_bg}")

        # Count rendering network parameters
        rendering_params = [k for k in state_dict.keys() if 'rendering' in k]
        print(f"\n  Rendering network parameters: {len(rendering_params)}")
        if len(rendering_params) > 0:
            print(f"    First 5: {rendering_params[:5]}")
        else:
            print(f"    ⚠️  NO RENDERING PARAMETERS!")

    # Diagnosis
    print(f"\n" + "=" * 70)
    print(f"DIAGNOSIS")
    print(f"=" * 70)

    # Calculate expected steps
    batch_size = train_dataset.get('batch_size', 2)
    num_epochs = training.get('num_epochs', 20)
    num_images = 71  # Known from dataset

    steps_per_epoch = num_images // batch_size
    expected_steps = steps_per_epoch * num_epochs

    print(f"\nExpected training:")
    print(f"  Images: {num_images}")
    print(f"  Batch size: {batch_size}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total epochs: {num_epochs}")
    print(f"  Expected total steps: {expected_steps}")

    actual_steps = ckpt.get('global_step', 0)
    print(f"\nActual training:")
    print(f"  Actual steps: {actual_steps}")

    if actual_steps < expected_steps * 0.5:
        print(f"\n❌ PROBLEM: Training incomplete!")
        print(f"   Only {actual_steps}/{expected_steps} steps ({actual_steps / expected_steps * 100:.1f}%)")
        print(f"   → Training may have crashed or been interrupted")
    elif actual_steps < expected_steps:
        print(f"\n⚠️  Training shorter than expected")
        print(f"   {actual_steps}/{expected_steps} steps ({actual_steps / expected_steps * 100:.1f}%)")
    else:
        print(f"\n✅ Training completed expected number of steps")


if __name__ == '__main__':
    analyze_config()