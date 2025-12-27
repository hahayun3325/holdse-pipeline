import os, torch
from collections import defaultdict
from src.model.ghop.diffusion import GHOP3DUNetWrapper

CKPT = "/home/fredcui/Projects/ghop/output/joint_3dprior/mix_data/checkpoints/last.ckpt"

ckpt = torch.load(CKPT, map_location="cpu")
state = ckpt.get("state_dict", ckpt)

# Only treat these as U-Net weights (ignore text_cond_model, ae.model, etc.)
UNET_PREFIXES = (
    "glide_model.input_blocks.",
    "glide_model.middle_block.",
    "glide_model.output_blocks.",
    "glide_model.time_embed.",
    "glide_model.out.",
)

# Remap checkpoint -> unet keyspace used by GHOP3DUNet (NO 'unet.' prefix)
ckpt_unet = {}
other_groups = defaultdict(int)

for k, v in state.items():
    if k.startswith("ae.model."):
        other_groups["ae.model"] += 1
        continue
    if k.startswith("glide_model.text_cond_model."):
        other_groups["text_cond_model"] += 1
        continue
    if k.startswith(UNET_PREFIXES):
        ckpt_unet[k.replace("glide_model.", "", 1)] = v
    elif k.startswith("glide_model."):
        other_groups["glide_model_other"] += 1
    else:
        other_groups["other"] += 1

print(f"Checkpoint total tensors: {len(state)}")
print(f"Checkpoint U-Net tensors: {len(ckpt_unet)}")
print("Other checkpoint groups:", dict(other_groups))

# Build wrapper (it will load ckpt in __init__)
wrapper = GHOP3DUNetWrapper(unet_ckpt_path=CKPT, device="cpu")
model_unet = wrapper.unet.state_dict()

model_keys = set(model_unet.keys())
ckpt_keys = set(ckpt_unet.keys())

missing_in_ckpt = sorted(model_keys - ckpt_keys)
extra_in_ckpt = sorted(ckpt_keys - model_keys)

print("\n=== KEY SET CHECK ===")
print("Model U-Net tensors:", len(model_keys))
print("Missing in checkpoint:", len(missing_in_ckpt))
print("Extra in checkpoint:", len(extra_in_ckpt))

if missing_in_ckpt[:10]:
    print("Sample missing:", missing_in_ckpt[:10])
if extra_in_ckpt[:10]:
    print("Sample extra:", extra_in_ckpt[:10])

# Shape check
shape_mismatch = []
for k in sorted(model_keys & ckpt_keys):
    if tuple(model_unet[k].shape) != tuple(ckpt_unet[k].shape):
        shape_mismatch.append((k, tuple(model_unet[k].shape), tuple(ckpt_unet[k].shape)))

print("\n=== SHAPE CHECK ===")
print("Shape mismatches:", len(shape_mismatch))
if shape_mismatch[:10]:
    for item in shape_mismatch[:10]:
        print(" ", item)

# Hard fail if anything unexpected shows up
assert len(missing_in_ckpt) == 0, "Model has keys not present in checkpoint U-Net subset."
assert len(extra_in_ckpt) == 0, "Checkpoint has U-Net keys not present in model."
assert len(shape_mismatch) == 0, "At least one key matches but tensor shape differs."

print("\n✅ Key+shape compatibility is perfect for U-Net.")
