import sys
sys.path.insert(0, '.')

# Quick test: Load actual training config
from omegaconf import OmegaConf
import argparse

# Load actual training config
config = OmegaConf.load('confs/stage3_hold_MC1_ho3d_sds_test_1epoch.yaml')

# Create args from config
args = argparse.Namespace()
args.data_dir = 'data/hold_MC1_ho3d/build'
args.case = 'hold_MC1_ho3d'
args.seq_name = 'hold_MC1_ho3d'
args.offset = 1
args.tempo_len = 100

# Add common attributes to avoid errors
for attr in ['debug', 'verbose', 'split', 'img_size', 'setup_with_object', 'setup_with_hand']:
    if not hasattr(args, attr):
        setattr(args, attr, False if attr in ['debug', 'verbose'] else 'train' if attr == 'split' else 512 if attr == 'img_size' else True)

print("Testing modified TempoDataset...")

from src.datasets.tempo_dataset import TempoDataset

try:
    dataset = TempoDataset(args)
    sample = dataset[0]
    
    # Quick check for temporal fields
    has_temporal = 'hA_n' in sample and 'c2w_n' in sample
    
    print(f"✅ Dataset created: {len(dataset)} samples")
    print(f"✅ Sample loaded: {len(sample.keys())} keys")
    print(f"{'✅' if has_temporal else '❌'} Temporal fields: hA_n={'hA_n' in sample}, c2w_n={'c2w_n' in sample}")
    
    if has_temporal:
        print(f"   hA_n shape: {sample['hA_n'].shape}")
        print(f"   c2w_n shape: {sample['c2w_n'].shape}")
        print(f"\n✅ Phase 5 will ACTIVATE")
    else:
        print(f"\n❌ Phase 5 will SKIP")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
