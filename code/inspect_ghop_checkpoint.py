import torch
import sys

ckpt_path = "/home/fredcui/Projects/ghop/output/joint_3dprior/mix_data/checkpoints/last.ckpt"

print(f"Loading checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location='cpu')

print(f"\n{'='*70}")
print("TOP-LEVEL CHECKPOINT STRUCTURE")
print(f"{'='*70}")
print(f"Top-level keys: {list(ckpt.keys())}")

# Check for nested U-Net
if 'unet' in ckpt:
    print(f"\n✓ Found nested 'unet' key")
    if isinstance(ckpt['unet'], dict):
        print(f"  Contains {len(ckpt['unet'])} parameters")
        print(f"  Sample keys: {list(ckpt['unet'].keys())[:5]}")

# Check state_dict
if 'state_dict' in ckpt:
    state_dict = ckpt['state_dict']
    print(f"\n✓ Found 'state_dict' with {len(state_dict)} parameters")
    
    # Count different prefixes
    prefixes = {}
    for key in state_dict.keys():
        prefix = key.split('.')[0]
        prefixes[prefix] = prefixes.get(prefix, 0) + 1
    
    print(f"\nParameter count by prefix:")
    for prefix, count in sorted(prefixes.items(), key=lambda x: -x[1])[:10]:
        print(f"  {prefix:30s}: {count:5d} params")
    
    # Search for U-Net with different prefixes
    print(f"\n{'='*70}")
    print("U-NET PARAMETER SEARCH")
    print(f"{'='*70}")
    
    unet_direct = {k: v for k, v in state_dict.items() if k.startswith('unet.')}
    print(f"Parameters with 'unet.' prefix: {len(unet_direct)}")
    
    glide_params = {k: v for k, v in state_dict.items() if k.startswith('glide_model.')}
    print(f"Parameters with 'glide_model.' prefix: {len(glide_params)}")
    if len(glide_params) > 0:
        print(f"  Sample keys: {list(glide_params.keys())[:5]}")
    
    ae_unet = {k: v for k, v in state_dict.items() if k.startswith('ae.model.unet.')}
    print(f"Parameters with 'ae.model.unet.' prefix: {len(ae_unet)}")
    
    attention_params = {k: v for k, v in state_dict.items() if 'attention' in k.lower() or 'attn' in k.lower()}
    print(f"Parameters with 'attention' in name: {len(attention_params)}")
    if len(attention_params) > 0:
        print(f"  Sample keys: {list(attention_params.keys())[:3]}")
    
    # Check for critical U-Net components
    print(f"\n{'='*70}")
    print("CRITICAL U-NET COMPONENT CHECK")
    print(f"{'='*70}")
    
    critical_patterns = [
        'input_blocks',
        'middle_block',
        'output_blocks',
        'time_embed',
        'out.'
    ]
    
    for pattern in critical_patterns:
        matching = [k for k in state_dict.keys() if pattern in k]
        print(f"  Keys containing '{pattern}': {len(matching)}")
        if len(matching) > 0:
            # Show actual prefix
            first_key = matching[0]
            prefix = first_key.split(pattern)[0]
            print(f"    Prefix: '{prefix}'")
            print(f"    Example: {first_key}")

print(f"\n{'='*70}")
print("INSPECTION COMPLETE")
print(f"{'='*70}")
