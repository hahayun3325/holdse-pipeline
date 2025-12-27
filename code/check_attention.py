import torch

ckpt_path = "/home/fredcui/Projects/ghop/output/joint_3dprior/mix_data/checkpoints/last.ckpt"
ckpt = torch.load(ckpt_path, map_location='cpu')
state_dict = ckpt['state_dict']

# Find all attention-related keys in input/middle/output blocks
attention_keys = []
for key in state_dict.keys():
    if 'glide_model' in key and any(x in key for x in ['attn', 'attention', 'qkv', 'norm']):
        if any(block in key for block in ['input_blocks', 'middle_block', 'output_blocks']):
            attention_keys.append(key)

print("="*70)
print(f"ATTENTION LAYERS IN CHECKPOINT: {len(attention_keys)} parameters")
print("="*70)

# Group by block
from collections import defaultdict
by_block = defaultdict(list)
for key in sorted(attention_keys):
    # Extract block identifier
    if 'input_blocks' in key:
        block_id = key.split('.')[2]
        by_block[f"input_blocks.{block_id}"].append(key)
    elif 'middle_block' in key:
        by_block["middle_block"].append(key)
    elif 'output_blocks' in key:
        block_id = key.split('.')[2]
        by_block[f"output_blocks.{block_id}"].append(key)

for block_name in sorted(by_block.keys()):
    keys = by_block[block_name]
    print(f"\n{block_name}: {len(keys)} attention params")
    # Show sample
    for key in keys[:3]:
        shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else '???'
        print(f"  {key.split('glide_model.')[1]:60s} {shape}")
    if len(keys) > 3:
        print(f"  ... and {len(keys)-3} more")

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"Total attention parameters: {len(attention_keys)}")
print(f"Blocks with attention: {len(by_block)}")

# Determine attention_resolutions
attention_blocks = set()
for key in attention_keys:
    if 'input_blocks' in key:
        block_id = int(key.split('.')[2])
        attention_blocks.add(block_id)

if attention_blocks:
    print(f"\nInput blocks with attention: {sorted(attention_blocks)}")
    print("This indicates attention_resolutions setting needed")
