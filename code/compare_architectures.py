import torch
import sys

ckpt_path = "/home/fredcui/Projects/ghop/output/joint_3dprior/mix_data/checkpoints/last.ckpt"

print("="*70)
print("CHECKPOINT ARCHITECTURE ANALYSIS")
print("="*70)

ckpt = torch.load(ckpt_path, map_location='cpu')
state_dict = ckpt['state_dict']

# Find input block channels
input_blocks = {}
for key in state_dict.keys():
    if key.startswith('glide_model.input_blocks.') and key.endswith('.weight'):
        if state_dict[key].ndim == 5:
            parts = key.split('.')
            block_id = int(parts[2])
            out_ch = state_dict[key].shape[0]
            
            if block_id not in input_blocks:
                input_blocks[block_id] = out_ch

# Sort and display
sorted_blocks = sorted(input_blocks.items())
print(f"\nInput blocks found: {len(sorted_blocks)}")
print("\nBlock ID → Output Channels:")
for block_id, channels in sorted_blocks[:15]:
    print(f"  Block {block_id:2d}: {channels:4d} channels")

# Extract channel progression
channel_seq = [ch for _, ch in sorted_blocks]
print(f"\nChannel sequence: {channel_seq[:10]}")

# Detect architecture
if channel_seq:
    model_channels = channel_seq[0]
    unique_channels = []
    seen = set()
    for ch in channel_seq:
        if ch not in seen:
            unique_channels.append(ch)
            seen.add(ch)
    
    channel_mult = [ch // model_channels for ch in unique_channels]
    
    print(f"\n{'='*70}")
    print("DETECTED ARCHITECTURE")
    print(f"{'='*70}")
    print(f"model_channels: {model_channels}")
    print(f"channel_mult: {channel_mult}")
    print(f"unique_channels: {unique_channels}")

# Check specific key shapes
print(f"\n{'='*70}")
print("SAMPLE LAYER SHAPES (for verification)")
print(f"{'='*70}")

critical_keys = [
    'glide_model.input_blocks.0.0.weight',
    'glide_model.input_blocks.1.0.weight',
    'glide_model.middle_block.0.in_layers.0.weight',
    'glide_model.output_blocks.0.0.in_layers.0.weight',
]

for key in critical_keys:
    if key in state_dict:
        shape = state_dict[key].shape
        print(f"{key}")
        print(f"  Shape: {shape}")
        print(f"  Out channels: {shape[0]}, In channels: {shape[1]}")
    else:
        print(f"{key}: NOT FOUND")

print(f"\n{'='*70}")
