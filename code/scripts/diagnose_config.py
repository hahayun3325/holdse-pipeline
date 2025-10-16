# File: scripts/diagnose_config.py
"""Check if config has all required fields."""
import yaml
from pathlib import Path

config_path = Path("confs/test_checkpoint_loading.yaml")

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

print("=" * 70)
print("Config Diagnosis")
print("=" * 70)

# Check dataset structure
print("\nDataset configuration:")
if 'dataset' in config:
    print("  ✓ dataset key exists")

    if 'train' in config['dataset']:
        print("  ✓ dataset.train exists")
        train_config = config['dataset']['train']

        # Check required fields
        required_fields = ['type', 'batch_size']
        for field in required_fields:
            if field in train_config:
                print(f"    ✓ dataset.train.{field} = {train_config[field]}")
            else:
                print(f"    ❌ dataset.train.{field} MISSING")
    else:
        print("  ❌ dataset.train MISSING")
else:
    print("  ❌ dataset key MISSING")

# Check Phase 3
print("\nPhase 3 configuration:")
if 'phase3' in config:
    print("  ✓ phase3 key exists")
    print(f"    enabled: {config['phase3'].get('enabled', 'not set')}")

    if 'ghop' in config['phase3']:
        ghop = config['phase3']['ghop']
        print(f"    unified_checkpoint: {ghop.get('unified_checkpoint', 'not set')}")
        print(f"    vqvae_use_pretrained: {ghop.get('vqvae_use_pretrained', 'not set')}")
        print(f"    unet_use_pretrained: {ghop.get('unet_use_pretrained', 'not set')}")
else:
    print("  ❌ phase3 key MISSING")

print("\n" + "=" * 70)