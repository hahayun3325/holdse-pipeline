import argparse
import torch
import numpy as np

def count_zero_crossings(sdf_grid: torch.Tensor):
    # sdf_grid: [1, D, H, W] or [D, H, W]
    if sdf_grid.dim() == 4:
        sdf = sdf_grid[0]
    else:
        sdf = sdf_grid

    signs = torch.sign(sdf)
    # treat zeros as +1 to avoid fake crossings
    signs[signs == 0] = 1.0

    cx = (signs[:, :, 1:] * signs[:, :, :-1] < 0).sum().item()
    cy = (signs[:, 1:, :] * signs[:, :-1, :] < 0).sum().item()
    cz = (signs[1:, :, :] * signs[:-1, :, :] < 0).sum().item()
    return cx + cy + cz


def summarize_tensor(name, t):
    t = t.float()
    print(f"{name}: shape={tuple(t.shape)}, "
          f"min={t.min().item():.4f}, max={t.max().item():.4f}, "
          f"mean={t.mean().item():.4f}, std={t.std().item():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default="logs/cb20a1702/checkpoints/last.ckpt")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    print("\n============================================================")
    print(f"Inspecting checkpoint: {args.ckpt}")
    print("============================================================\n")

    # ---- 1. Detect FiLM vs baseline ----
    object_prefix = "model.nodes.object."
    implicit_prefix = object_prefix + "implicit_network."

    has_film = any(
        k.startswith(implicit_prefix + "film_gamma") or
        k.startswith(implicit_prefix + "film_beta")
        for k in state_dict.keys()
    )

    lin0_wv_key = implicit_prefix + "lin0.weight_v"
    lin0_wv = state_dict.get(lin0_wv_key, None)

    print("Architecture detection:")
    print(f"  has_film: {has_film}")
    if lin0_wv is not None:
        print(f"  lin0.weight_v shape: {tuple(lin0_wv.shape)}")
        if lin0_wv.shape[1] == 39:
            print("  -> Interpreted as FiLM+PE (39‑dim positional encoding).")
        elif lin0_wv.shape[1] == 167:
            print("  -> Interpreted as concatenation (39 + 128).")
        else:
            print("  -> Unexpected lin0 input dim, manual check needed.")
    else:
        print("  lin0.weight_v: MISSING (will be randomly initialized at load).")

    # ---- 2. Report FiLM weights if present ----
    if has_film:
        for i in range(4):
            gk = f"{implicit_prefix}film_gamma.{i}.weight"
            bk = f"{implicit_prefix}film_gamma.{i}.bias"
            if gk in state_dict:
                summarize_tensor(f"film_gamma.{i}.weight", state_dict[gk])
            if bk in state_dict:
                summarize_tensor(f"film_gamma.{i}.bias", state_dict[bk])

    # ---- 3. Inspect SDF grid (Option B) ----
    sdf_key = object_prefix + "server.object_model.sdf_grid"
    if sdf_key in state_dict:
        sdf = state_dict[sdf_key]
        print("\nSDF grid:")
        summarize_tensor("  sdf_grid", sdf)
        try:
            zc = count_zero_crossings(sdf)
            print(f"  zero-crossings: {zc}")
        except Exception as e:
            print(f"  zero-crossings: ERROR ({e})")
    else:
        print("\nSDF grid: not found (baseline-style object).")

    # ---- 4. Inspect canonical vertices ----
    v3d_key = object_prefix + "server.object_model.v3d_cano"
    if v3d_key in state_dict:
        v = state_dict[v3d_key].float()
        mins = v.min(dim=0).values
        maxs = v.max(dim=0).values
        print("\nv3d_cano:")
        print(f"  bbox: [{mins.min().item():.4f}, {maxs.max().item():.4f}]")
        print(f"  std: {v.std().item():.4f}")
    else:
        print("\nv3d_cano: not found.")

    print("\nDone.\n")


if __name__ == "__main__":
    main()

# # Run comparison
# inspect_checkpoint('logs/816cf7741_000000500/checkpoints/last.ckpt', 'HOLDSE step 500')
# # inspect_checkpoint('logs/bd68a88d0_000004000/checkpoints/last.ckpt', 'HOLDSE step 4000 with Distillation')
# inspect_checkpoint('/home/fredcui/Projects/hold/code/logs/cb20a1702/checkpoints/last.ckpt', 'OFFICIAL HOLD')

'''
python scripts/inspect_checkpoint.py --ckpt logs/cb20a1702_test/checkpoints/last.ckpt

'''