import torch

ckpt_path = "/home/fredcui/Projects/ghop/output/joint_3dprior/mix_data/checkpoints/last.ckpt"
ckpt = torch.load(ckpt_path, map_location='cpu')
state_dict = ckpt['state_dict']

# Count layers in first input block to determine res_blocks
block0_keys = [k for k in state_dict.keys() if k.startswith('glide_model.input_blocks.0.')]

print("="*70)
print("FIRST INPUT BLOCK STRUCTURE")
print("="*70)
print(f"Total keys in input_blocks.0: {len(block0_keys)}")
print("\nAll keys:")
for key in sorted(block0_keys):
    if state_dict[key].ndim > 0:
        print(f"  {key:60s} shape: {tuple(state_dict[key].shape)}")

# Count ResBlock instances (each has ~6-8 parameters)
# Pattern: input_blocks.0.{res_block_id}.layer.weight
res_block_ids = set()
for key in block0_keys:
    parts = key.split('.')
    if len(parts) >= 4 and parts[3].isdigit():
        res_block_ids.add(int(parts[3]))

print(f"\n{'='*70}")
print(f"Detected res_block IDs in input_blocks.0: {sorted(res_block_ids)}")
print(f"Estimated num_res_blocks: {len(res_block_ids)}")

# Check middle block structure (definitive for num_res_blocks)
middle_keys = [k for k in state_dict.keys() if k.startswith('glide_model.middle_block.')]
middle_block_ids = set()
for key in middle_keys:
    parts = key.split('.')
    if len(parts) >= 3 and parts[2].isdigit():
        middle_block_ids.add(int(parts[2]))

print(f"\nMiddle block structure:")
print(f"  ResBlock IDs: {sorted(middle_block_ids)}")
print(f"  Total middle_block keys: {len(middle_keys)}")

# Check for attention layers
attn_keys = [k for k in state_dict.keys() if 'attention' in k.lower() or 'attn' in k.lower()]
print(f"\n{'='*70}")
print(f"Attention layers found: {len(attn_keys)}")
if len(attn_keys) > 0:
    print("Sample attention keys:")
    for key in sorted(attn_keys)[:5]:
        print(f"  {key}")
