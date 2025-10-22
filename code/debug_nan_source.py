# File: code/debug_nan_source.py
"""Find where NaN originates in rendering."""

import torch
import sys
import numpy as np

sys.path = ['..'] + sys.path
from src.utils.parser import parser_args
from src.datasets.utils import create_dataset
import common.thing as thing
from src.hold.hold import HOLD
from common.torch_utils import reset_all_seeds


def main():
    device = "cuda:0"
    args, opt = parser_args()

    # Load model
    model = HOLD(opt, args)
    reset_all_seeds(1)
    sd = torch.load(args.load_ckpt)["state_dict"]
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()

    # Disable barf
    for node in model.model.nodes.values():
        node.implicit_network.embedder_obj.eval()
    model.model.background.bg_implicit_network.embedder_obj.eval()
    model.model.background.bg_rendering_network.embedder_obj.eval()

    # Load dataset
    testset = create_dataset(opt.dataset.test, args)
    batch = next(iter(testset))
    batch = thing.thing2dev(batch, device)

    print("=" * 70)
    print("DEBUGGING NaN SOURCE")
    print("=" * 70)

    # Check input batch
    print("\n1. Checking input batch:")
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            has_nan = torch.isnan(val).any()
            has_inf = torch.isinf(val).any()
            print(f"  {key:25s} shape={str(val.shape):20s} NaN={has_nan} Inf={has_inf}")
            if has_nan or has_inf:
                print(f"    ❌ PROBLEM IN INPUT!")

    # Run inference with intermediate checks
    print("\n2. Running inference with intermediate checks:")

    with torch.no_grad():
        # Monkey-patch rendering function to add checks
        original_forward = model.model.forward

        def checked_forward(*args, **kwargs):
            out = original_forward(*args, **kwargs)

            # Check outputs
            for key, val in out.items():
                if isinstance(val, torch.Tensor) and val.dtype == torch.float32:
                    if torch.isnan(val).any():
                        print(f"    ❌ NaN in output['{key}']")
                    if torch.isinf(val).any():
                        print(f"    ⚠️  Inf in output['{key}']")

            return out

        model.model.forward = checked_forward

        out = model.inference_step(batch)

        model.model.forward = original_forward

    # Final analysis
    print("\n3. Final output analysis:")
    rgb = out.get('rgb')
    if rgb is not None:
        print(f"  RGB shape: {rgb.shape}")
        print(f"  RGB dtype: {rgb.dtype}")
        print(f"  RGB device: {rgb.device}")

        # Statistics
        if not torch.isnan(rgb).all():
            valid_mask = ~torch.isnan(rgb)
            if valid_mask.any():
                valid_vals = rgb[valid_mask]
                print(
                    f"  Valid values: min={valid_vals.min():.4f} max={valid_vals.max():.4f} mean={valid_vals.mean():.4f}")

        nan_count = torch.isnan(rgb).sum().item()
        inf_count = torch.isinf(rgb).sum().item()
        total = rgb.numel()

        print(f"  NaN count: {nan_count}/{total} ({nan_count / total * 100:.1f}%)")
        print(f"  Inf count: {inf_count}/{total} ({inf_count / total * 100:.1f}%)")

        if nan_count == total:
            print(f"\n  ❌ ALL VALUES ARE NaN!")
            print(f"  Likely causes:")
            print(f"    1. Rendering network not properly initialized")
            print(f"    2. Division by zero in rendering")
            print(f"    3. Invalid operations (log of negative, etc.)")
        elif nan_count > 0:
            print(f"\n  ⚠️  PARTIAL NaN!")
            print(f"  Some pixels rendered successfully")


if __name__ == '__main__':
    main()