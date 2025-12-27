import torch

ckpt = torch.load('/home/fredcui/Projects/ghop/output/joint_3dprior/mix_data/checkpoints/last.ckpt', map_location='cpu')
state_dict = ckpt['state_dict']

# Analyze one transformer block
block_keys = [k for k in state_dict.keys() if 'input_blocks.10.1.transformer_blocks.0' in k]

print("="*70)
print("CHECKPOINT TRANSFORMER BLOCK STRUCTURE (input_blocks.10.1):")
print("="*70)
for key in sorted(block_keys):
    shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else '???'
    param_name = key.split('transformer_blocks.0.')[1]
    print(f"  {param_name:50s} {shape}")

print(f"\nTotal parameters in one transformer block: {len(block_keys)}")

# Get channel dimensions
attn1_k = state_dict['glide_model.input_blocks.10.1.transformer_blocks.0.attn1.to_k.weight']
print(f"\nDimensions:")
print(f"  attn1.to_k shape: {attn1_k.shape}")
print(f"  Suggests inner_dim={attn1_k.shape[0]}, channels={attn1_k.shape[1]}")

# Check number of heads
norm1 = state_dict['glide_model.input_blocks.10.1.transformer_blocks.0.norm1.weight']
print(f"  norm1 shape: {norm1.shape[0]} → this is the channel dimension")
