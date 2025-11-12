#!/usr/bin/env python3
"""Inspect Stage 1 data and rendering"""

import sys
import os
import torch
sys.path.insert(0, 'src')

def inspect_data():
    """Check dataset output dimensions and ranges"""
    
    print("=" * 80)
    print("STAGE 1 DATA INSPECTION")
    print("=" * 80)
    print()
    
    try:
        from datasets.tempo_dataset import TempoDataset
        print("✓ TempoDataset found")
        
        # Try to create mock args
        class Args:
            case = 'ghop_bottle_1'
            data_root = os.path.expanduser('~/Projects/holdse/data')
            offset = 1
        
        # Try to load dataset
        try:
            dataset = TempoDataset(Args())
            print(f"✓ Dataset loaded: {len(dataset)} samples")
            
            # Inspect a sample
            sample = dataset[0]
            print()
            print("Sample structure:")
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}, "
                          f"min={value.min():.4f}, max={value.max():.4f}")
                else:
                    print(f"  {key}: type={type(value)}")
            
            print()
            print("Checking RGB values:")
            if 'rgb' in sample:
                rgb = sample['rgb']
                print(f"  RGB shape: {rgb.shape}")
                print(f"  RGB range: [{rgb.min():.4f}, {rgb.max():.4f}]")
                print(f"  RGB dtype: {rgb.dtype}")
                
                if rgb.max() > 1.5:
                    print(f"  ⚠️  RGB values appear to be in [0, 255] range")
                elif rgb.max() <= 1.0:
                    print(f"  ✓ RGB values appear to be in [0, 1] range")
            
            if 'gt.rgb' in sample:
                gt_rgb = sample['gt.rgb']
                print(f"  Ground truth RGB shape: {gt_rgb.shape}")
                print(f"  Ground truth RGB range: [{gt_rgb.min():.4f}, {gt_rgb.max():.4f}]")
                
        except Exception as e:
            print(f"✗ Failed to load dataset: {e}")
            import traceback
            traceback.print_exc()
            
    except ImportError as e:
        print(f"✗ Cannot import TempoDataset: {e}")

if __name__ == "__main__":
    inspect_data()

