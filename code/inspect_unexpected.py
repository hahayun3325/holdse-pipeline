import torch
from src.model.ghop.diffusion import GHOP3DUNetWrapper

# Load checkpoint
ckpt = torch.load('/home/fredcui/Projects/ghop/output/joint_3dprior/mix_data/checkpoints/last.ckpt', map_location='cpu')
state_dict = ckpt['state_dict']

# Extract checkpoint U-Net keys
ckpt_keys = set()
for key in state_dict.keys():
    if key.startswith('glide_model.'):
        new_key = key.replace('glide_model.', 'unet.', 1)
        ckpt_keys.add(new_key)

# Create model and get its keys
wrapper = GHOP3DUNetWrapper(
    unet_ckpt_path='/home/fredcui/Projects/ghop/output/joint_3dprior/mix_data/checkpoints/last.ckpt',
    device='cpu'
)
model_keys = set(wrapper.unet.state_dict().keys())

# Find unexpected keys (in checkpoint but not in model)
unexpected = ckpt_keys - model_keys

print("="*70)
print(f"UNEXPECTED PARAMETERS IN CHECKPOINT: {len(unexpected)}")
print("="*70)

# Group by pattern
transformer_keys = [k for k in unexpected if 'transformer_blocks' in k]
other_keys = [k for k in unexpected if 'transformer_blocks' not in k]

print(f"\nTransformer-related: {len(transformer_keys)}")
if transformer_keys:
    print("\nSample transformer unexpected keys:")
    for key in sorted(transformer_keys)[:10]:
        print(f"  {key}")

print(f"\nOther unexpected: {len(other_keys)}")
if other_keys:
    print("\nSample other unexpected keys:")
    for key in sorted(other_keys)[:10]:
        print(f"  {key}")

# Also check what the MODEL has for transformers
print("\n" + "="*70)
print("MODEL'S TRANSFORMER PARAMETER NAMES:")
print("="*70)
model_transformer_keys = [k for k in model_keys if 'transformer_blocks' in k or 'attn' in k or 'cross_attn' in k]
print(f"\nTotal transformer-related in model: {len(model_transformer_keys)}")
print("\nSample model transformer keys:")
for key in sorted(model_transformer_keys)[:10]:
    print(f"  {key}")
