#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')

import torch
from omegaconf import OmegaConf

# Load checkpoint
ckpt_path = '/home/fredcui/Projects/ghop/output/joint_3dprior/mix_data/checkpoints/last.ckpt'
ckpt = torch.load(ckpt_path, map_location='cpu')
state_dict = ckpt['state_dict']

# Load your config
cfg = OmegaConf.load('checkpoints/ghop/config.yaml')

print("=" * 70)
print("ARCHITECTURE COMPARISON")
print("=" * 70)

# ============================================================
# Find actual keys (flexible search)
# ============================================================
# VQ-VAE encoder input
vqvae_input_key = None
for key in state_dict.keys():
    if 'encoder' in key.lower() and ('conv_in.weight' in key or 'input' in key):
        vqvae_input_key = key
        break

if vqvae_input_key:
    vqvae_input_shape = state_dict[vqvae_input_key].shape
    print(f"\n✅ Found VQ-VAE input layer: {vqvae_input_key}")
    print(f"   Shape: {vqvae_input_shape}")
    print(f"   Checkpoint expects: {vqvae_input_shape[1]} input channels")
else:
    print("\n❌ Could not find VQ-VAE encoder input layer")
    vqvae_input_shape = None

# U-Net output
unet_output_key = None
for key in state_dict.keys():
    if 'unet' in key.lower() and 'out' in key and 'weight' in key and 'conv' not in key.lower():
        if state_dict[key].ndim == 5:  # 3D conv: [out_ch, in_ch, d, h, w]
            unet_output_key = key
            break

if unet_output_key:
    unet_output_shape = state_dict[unet_output_key].shape
    print(f"\n✅ Found U-Net output layer: {unet_output_key}")
    print(f"   Shape: {unet_output_shape}")
    print(f"   Checkpoint outputs: {unet_output_shape[0]} channels")
else:
    print("\n❌ Could not find U-Net output layer")
    unet_output_shape = None

# ============================================================
# Compare with config
# ============================================================
print("\n" + "=" * 70)
print("CONFIG COMPARISON")
print("=" * 70)

# VQ-VAE input channels
if vqvae_input_shape:
    checkpoint_vqvae_in = vqvae_input_shape[1]

    # Try different config paths
    config_vqvae_in = None
    if hasattr(cfg.model, 'vqvae') and hasattr(cfg.model.vqvae, 'in_channels'):
        config_vqvae_in = cfg.model.vqvae.in_channels
    elif hasattr(cfg.model, 'first_stage') and hasattr(cfg.model.first_stage, 'in_channels'):
        config_vqvae_in = cfg.model.first_stage.in_channels

    print(f"\nVQ-VAE Input Channels:")
    print(f"  Checkpoint: {checkpoint_vqvae_in}")
    print(f"  Config:     {config_vqvae_in if config_vqvae_in else 'NOT FOUND IN CONFIG'}")

    if config_vqvae_in == checkpoint_vqvae_in:
        print("  ✅ MATCH")
    else:
        print(f"  ❌ MISMATCH - Need to set in_channels={checkpoint_vqvae_in}")

# U-Net output channels
if unet_output_shape:
    checkpoint_unet_out = unet_output_shape[0]

    # Try different config paths
    config_unet_out = None
    if hasattr(cfg.model, 'unet') and hasattr(cfg.model.unet, 'params'):
        config_unet_out = cfg.model.unet.params.get('out_channels', None)
    elif hasattr(cfg.model, 'unet') and hasattr(cfg.model.unet, 'out_channels'):
        config_unet_out = cfg.model.unet.out_channels

    print(f"\nU-Net Output Channels:")
    print(f"  Checkpoint: {checkpoint_unet_out}")
    print(f"  Config:     {config_unet_out if config_unet_out else 'NOT FOUND IN CONFIG'}")

    if config_unet_out == checkpoint_unet_out:
        print("  ✅ MATCH")
    else:
        print(f"  ❌ MISMATCH - Need to set out_channels={checkpoint_unet_out}")

print("=" * 70)

# ============================================================
# Provide fix instructions
# ============================================================
if vqvae_input_shape and checkpoint_vqvae_in != config_vqvae_in:
    print("\n⚠️  FIX REQUIRED: Update checkpoints/ghop/config.yaml")
    print(f"\n  Current config has U-Net in_channels: {cfg.model.unet.params.in_channels}")
    print(f"  But VQ-VAE expects: {checkpoint_vqvae_in}")
    print(f"\n  The wrapper will receive {cfg.model.unet.params.in_channels}-channel input")
    print(f"  and needs to convert it to {checkpoint_vqvae_in} channels for VQ-VAE")

if unet_output_shape and checkpoint_unet_out != config_unet_out:
    print("\n⚠️  FIX REQUIRED: Update checkpoints/ghop/config.yaml")
    print(f"\n  Add under model.unet.params:")
    print(f"    out_channels: {checkpoint_unet_out}")