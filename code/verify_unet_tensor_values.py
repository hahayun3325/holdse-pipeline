import random, torch
from src.model.ghop.diffusion import GHOP3DUNetWrapper

CKPT = "/home/fredcui/Projects/ghop/output/joint_3dprior/mix_data/checkpoints/last.ckpt"

ckpt = torch.load(CKPT, map_location="cpu")
state = ckpt.get("state_dict", ckpt)

UNET_PREFIXES = (
    "glide_model.input_blocks.",
    "glide_model.middle_block.",
    "glide_model.output_blocks.",
    "glide_model.time_embed.",
    "glide_model.out.",
)

ckpt_unet = {k.replace("glide_model.", "", 1): v
            for k, v in state.items()
            if k.startswith(UNET_PREFIXES)}

wrapper = GHOP3DUNetWrapper(unet_ckpt_path=CKPT, device="cpu")
model_unet = wrapper.unet.state_dict()

common = sorted(set(model_unet.keys()) & set(ckpt_unet.keys()))
print("Common keys:", len(common))

# Spot-check (fast)
random.seed(0)
sample = random.sample(common, k=min(30, len(common)))
bad = []
for k in sample:
    a = model_unet[k].detach().cpu()
    b = ckpt_unet[k].detach().cpu()
    if a.dtype.is_floating_point:
        ok = torch.allclose(a, b, atol=0.0, rtol=0.0)
    else:
        ok = torch.equal(a, b)
    if not ok:
        bad.append(k)

print("Spot-check mismatches:", len(bad))
if bad[:10]:
    print("Sample mismatched keys:", bad[:10])

# Optional: full verification (slower but definitive)
FULL = True
if FULL:
    bad_full = []
    for k in common:
        a = model_unet[k].detach().cpu()
        b = ckpt_unet[k].detach().cpu()
        if a.dtype.is_floating_point:
            ok = torch.allclose(a, b, atol=0.0, rtol=0.0)
        else:
            ok = torch.equal(a, b)
        if not ok:
            bad_full.append(k)
    print("Full mismatches:", len(bad_full))
    if bad_full[:10]:
        print("Sample mismatched keys:", bad_full[:10])
    assert len(bad_full) == 0, "Some tensors differ: checkpoint not fully applied."

print("✅ U-Net tensor values match checkpoint.")
