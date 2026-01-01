# import torch
#
# ckpt_path = '/home/fredcui/Projects/ghop/output/joint_3dprior/mix_data/checkpoints/last.ckpt'
# ckpt = torch.load(ckpt_path, map_location='cpu')
# state_dict = ckpt['state_dict']
#
# print(f"Total keys: {len(state_dict)}")
# print(f"\n{'=' * 70}")
# print("KEY PREFIX ANALYSIS")
# print('=' * 70)
#
# # Group by first prefix
# prefixes = {}
# for key in state_dict.keys():
#     parts = key.split('.')
#     prefix = parts[0] if len(parts) > 0 else 'root'
#     prefixes[prefix] = prefixes.get(prefix, 0) + 1
#
# for prefix, count in sorted(prefixes.items(), key=lambda x: -x[1]):
#     print(f"{prefix:30s}: {count:4d} keys")
#
# print(f"\n{'=' * 70}")
# print("VQ-VAE KEY SEARCH")
# print('=' * 70)
#
# # Search for encoder/decoder patterns
# vqvae_patterns = ['encoder', 'decoder', 'quantiz', 'quant_conv', 'post_quant']
# vqvae_keys = {}
# for key in state_dict.keys():
#     for pattern in vqvae_patterns:
#         if pattern in key.lower():
#             prefix = '.'.join(key.split('.')[:2])  # First two levels
#             vqvae_keys[prefix] = vqvae_keys.get(prefix, 0) + 1
#             break
#
# if vqvae_keys:
#     print("Found VQ-VAE related keys:")
#     for prefix, count in sorted(vqvae_keys.items()):
#         print(f"  {prefix}: {count} keys")
#
#     # Show samples
#     print("\nSample VQ-VAE keys:")
#     sample_keys = [k for k in list(state_dict.keys())[:50] if any(p in k.lower() for p in vqvae_patterns)]
#     for key in sample_keys[:5]:
#         print(f"  {key}")
# else:
#     print("❌ NO VQ-VAE keys found!")
#
# print(f"\n{'=' * 70}")
# print("U-NET KEY SEARCH")
# print('=' * 70)
#
# unet_patterns = ['unet', 'diffusion', 'time_embed', 'in_layers', 'out_layers']
# unet_keys = {}
# for key in state_dict.keys():
#     for pattern in unet_patterns:
#         if pattern in key.lower():
#             prefix = '.'.join(key.split('.')[:2])
#             unet_keys[prefix] = unet_keys.get(prefix, 0) + 1
#             break
#
# if unet_keys:
#     print("Found U-Net related keys:")
#     for prefix, count in sorted(unet_keys.items()):
#         print(f"  {prefix}: {count} keys")
#
#     print("\nSample U-Net keys:")
#     sample_keys = [k for k in list(state_dict.keys())[:50] if any(p in k.lower() for p in unet_patterns)]
#     for key in sample_keys[:5]:
#         print(f"  {key}")
# else:
#     print("❌ NO U-Net keys found!")
#
# print(f"\n{'=' * 70}")
# print("ACTUAL KEY SAMPLES (first 20)")
# print('=' * 70)
# for key in list(state_dict.keys())[:20]:
#     print(f"  {key}")

import torch

ckpt = torch.load('/home/fredcui/Projects/ghop/output/joint_3dprior/mix_data/checkpoints/last.ckpt', map_location='cpu')

# Check for architecture config
if 'hyper_parameters' in ckpt:
    print("Hyperparameters:", ckpt['hyper_parameters'])

if 'config' in ckpt:
    print("Config:", ckpt['config'])

# Check encoder weight shapes to infer architecture
state_dict = ckpt['state_dict']

# Find GroupNorm weights
for key, value in state_dict.items():
    if 'norm' in key and 'weight' in key and 'ae.model' in key:
        print(f"{key}: {value.shape}")