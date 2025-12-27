import torch
import sys
sys.path.insert(0, 'src')

# Load checkpoint
ckpt = torch.load('/home/fredcui/Projects/ghop/output/joint_3dprior/mix_data/checkpoints/last.ckpt', map_location='cpu')
ckpt_keys = set(k for k in ckpt['state_dict'].keys() if 'glide_model.' in k)

print("="*70)
print("CHECKPOINT ATTENTION KEYS (sample):")
print("="*70)
attention_keys = [k for k in sorted(ckpt_keys) if 'transformer_blocks' in k or 'attn' in k]
for key in attention_keys[:20]:
    print(f"  {key}")
if len(attention_keys) > 20:
    print(f"  ... and {len(attention_keys)-20} more")
print(f"\nTotal attention-related keys: {len(attention_keys)}")

# Now try to create the model
from model.ghop.diffusion import GHOP3DUNet

model = GHOP3DUNet(
    in_channels=3,
    out_channels=23,
    model_channels=64,
    channel_mult=[1, 2, 3],
    num_res_blocks=3,
    attention_resolutions=[2, 4],  # ← This should create attention!
    dropout=0.0,
    context_dim=768
)

model_keys = set(model.state_dict().keys())

print("\n" + "="*70)
print("MODEL ATTENTION KEYS (sample):")
print("="*70)
model_attn_keys = [k for k in sorted(model_keys) if 'transformer_blocks' in k or 'attn' in k]
for key in model_attn_keys[:20]:
    print(f"  {key}")
if len(model_attn_keys) > 20:
    print(f"  ... and {len(model_attn_keys)-20} more")
print(f"\nTotal attention-related keys: {len(model_attn_keys)}")

print("\n" + "="*70)
print("COMPARISON:")
print("="*70)
print(f"Checkpoint has attention keys: {len(attention_keys)}")
print(f"Model has attention keys: {len(model_attn_keys)}")

if len(model_attn_keys) == 0:
    print("\n❌ MODEL HAS NO ATTENTION LAYERS!")
    print("   GHOP3DUNet is ignoring attention_resolutions parameter!")
elif len(model_attn_keys) != len(attention_keys):
    print(f"\n⚠️  MISMATCH: Model has {len(model_attn_keys)}, checkpoint has {len(attention_keys)}")
else:
    print("\n✅ Model has same number of attention keys as checkpoint")
