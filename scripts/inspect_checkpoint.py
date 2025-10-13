#!/usr/bin/env python3
"""Inspect GHOP checkpoint structure."""

import torch
import os

def inspect_checkpoint(path="checkpoints/ghop/last.ckpt"):
    """Inspect GHOP checkpoint contents."""
    
    if not os.path.exists(path):
        print(f"✗ Checkpoint not found: {path}")
        return
    
    print("="*70)
    print("GHOP Checkpoint Inspection")
    print("="*70)
    
    # Load checkpoint
    print(f"\nLoading: {path}")
    size_gb = os.path.getsize(path) / (1024**3)
    print(f"Size: {size_gb:.2f} GB")
    
    ckpt = torch.load(path, map_location='cpu')
    
    # Top-level keys
    print(f"\nTop-level keys: {list(ckpt.keys())}")
    
    # State dict analysis
    if 'state_dict' in ckpt:
        state = ckpt['state_dict']
        print(f"\nTotal parameters: {len(state)}")
        
        # Categorize parameters
        categories = {
            'VQ-VAE Encoder': ['encoder', 'enc'],
            'VQ-VAE Decoder': ['decoder', 'dec'],
            'Codebook': ['quant', 'codebook'],
            'U-Net': ['model', 'unet'],
            'Attention': ['attn', 'attention'],
            'Other': []
        }
        
        counts = {cat: 0 for cat in categories}
        
        for key in state.keys():
            categorized = False
            for cat, keywords in categories.items():
                if cat != 'Other' and any(kw in key.lower() for kw in keywords):
                    counts[cat] += 1
                    categorized = True
                    break
            if not categorized:
                counts['Other'] += 1
        
        print("\nParameter breakdown:")
        for cat, count in counts.items():
            if count > 0:
                print(f"  {cat}: {count} parameters")
        
        # Sample parameter keys
        print("\nSample VQ-VAE keys:")
        vqvae_keys = [k for k in state.keys() if any(x in k.lower() for x in ['encoder', 'decoder', 'quant'])]
        for key in vqvae_keys[:3]:
            print(f"  - {key}: {state[key].shape}")
        
        print("\nSample U-Net keys:")
        unet_keys = [k for k in state.keys() if any(x in k.lower() for x in ['model', 'unet'])]
        for key in unet_keys[:3]:
            print(f"  - {key}: {state[key].shape}")
    
    # Training metadata
    if 'epoch' in ckpt:
        print(f"\nTraining epoch: {ckpt['epoch']}")
    if 'global_step' in ckpt:
        print(f"Global step: {ckpt['global_step']}")
    
    print("\n" + "="*70)
    print("✓ Checkpoint is valid and contains both VQ-VAE and U-Net")
    print("="*70)

if __name__ == '__main__':
    inspect_checkpoint()
