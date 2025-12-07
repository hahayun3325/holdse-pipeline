# scripts/debug_dims_stage1.py
import os
import sys
import argparse
import torch

os.environ["COMET_MODE"] = "disabled"
sys.path.insert(0, ".")
sys.path.insert(0, "../common")

from src.hold.hold import HOLD
from src.utils.parser import parser_args


def main():
    # 1) First parse ONLY --config and --checkpoint locally, strip them from sys.argv
    local_parser = argparse.ArgumentParser(add_help=False)
    local_parser.add_argument("--config", required=True, help="Path to YAML config")
    local_parser.add_argument("--checkpoint", required=True, help="Path to .ckpt")
    # parse_known_args so we don't eat args meant for parser_args()
    local_args, remaining_argv = local_parser.parse_known_args()

    # Rebuild sys.argv for parser_args(): keep script name + remaining args
    sys.argv = [sys.argv[0]] + remaining_argv + ["--config", local_args.config]

    # 2) Now call parser_args() exactly as train.py does
    args, opt = parser_args()

    # 3) Instantiate model the same way as in train.py
    model = HOLD(opt, args)

    # 4) Load checkpoint
    ckpt = torch.load(local_args.checkpoint, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("\n[DEBUG] load_state_dict:")
    print("  missing keys:", len(missing))
    print("  unexpected keys:", len(unexpected))

    print("\n" + "="*70)
    print("[INSPECTION] Model structure via PyTorch APIs")
    print("="*70)

    # 1. Show all parameter keys to understand storage structure
    print("\n[1] Checkpoint parameter keys (first 30):")
    state_keys = list(ckpt.get("state_dict", ckpt).keys())
    for i, key in enumerate(state_keys[:30]):
        print(f"  {key}")
    if len(state_keys) > 30:
        print(f"  ... ({len(state_keys) - 30} more keys)")

    # 2. Show named_modules to see full hierarchy
    print("\n[2] Model module hierarchy (modules with 'render' or 'network'):")
    for name, module in model.model.named_modules():
        if 'render' in name.lower() or 'network' in name.lower() or 'implicit' in name.lower():
            print(f"  {name}: {type(module).__name__}")

            # If it's a rendering network, try to get dims
            if hasattr(module, 'dims'):
                print(f"    -> dims: {module.dims}")
            if hasattr(module, 'lin0'):
                print(f"    -> lin0: {module.lin0.in_features} → {module.lin0.out_features}")

    # 3. Specifically check nodes ModuleDict
    print("\n[3] Nodes ModuleDict contents:")
    if hasattr(model.model, 'nodes'):
        print(f"  Type: {type(model.model.nodes).__name__}")
        if isinstance(model.model.nodes, torch.nn.ModuleDict):
            print(f"  Keys: {list(model.model.nodes.keys())}")
            for key in model.model.nodes.keys():
                node = model.model.nodes[key]
                print(f"\n  [{key}] type: {type(node).__name__}")
                # Check for rendering-related attributes
                for attr in ['rendering_network', 'implicit_network', 'sdf_network', 'model']:
                    if hasattr(node, attr):
                        obj = getattr(node, attr)
                        print(f"    -> {attr}: {type(obj).__name__}")
                        if hasattr(obj, 'dims'):
                            print(f"       dims: {obj.dims}")
                        if hasattr(obj, 'lin0'):
                            print(f"       lin0: {obj.lin0.in_features} → {obj.lin0.out_features}")

    # 4. Check background explicitly
    print("\n[4] Background module:")
    if hasattr(model.model, 'background'):
        bg = model.model.background
        print(f"  Type: {type(bg).__name__}")
        if hasattr(bg, 'bg_rendering_network'):
            rn = bg.bg_rendering_network
            print(f"  bg_rendering_network: {type(rn).__name__}")
            if hasattr(rn, 'dims'):
                print(f"    dims: {rn.dims}")
            if hasattr(rn, 'lin0'):
                print(f"    lin0: {rn.lin0.in_features} → {rn.lin0.out_features}")

    print("="*70 + "\n")

    # 5) Print rendering network dimensions
    print("\n[DEBUG] Rendering networks:")

    # Body rendering network
    try:
        body_net = model.model.implicit_network.rendering_network
        if hasattr(body_net, 'dims'):
            print("  body rendering dims:", body_net.dims)
        elif hasattr(body_net, 'lin0'):
            in_dim = body_net.lin0.in_features
            out_dim = body_net.lin0.out_features
            print(f"  body rendering lin0: {in_dim} → {out_dim}")
        else:
            print("  body rendering: cannot infer dims from attributes")
    except AttributeError as e:
        print("  body rendering dims: ERROR:", e)

    # Background rendering network
    try:
        bg_net = model.model.background.bg_rendering_network
        if hasattr(bg_net, 'dims'):
            print("  bg rendering dims:", bg_net.dims)
        elif hasattr(bg_net, 'lin0'):
            in_dim = bg_net.lin0.in_features
            out_dim = bg_net.lin0.out_features
            print(f"  bg rendering lin0: {in_dim} → {out_dim}")
        else:
            print("  bg rendering: cannot infer dims from attributes")
    except AttributeError as e:
        print("  bg rendering dims: ERROR:", e)


if __name__ == "__main__":
    main()