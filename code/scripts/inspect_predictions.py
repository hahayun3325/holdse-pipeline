import os
import sys
import argparse
import torch

# Disable Comet
os.environ["COMET_MODE"] = "disabled"

sys.path.insert(0, ".")
sys.path.insert(0, "../common")

from src.hold.hold import HOLD
# Instead of parser_args, import the config loader directly
# Check src/utils/config.py or src/utils/parser.py to see what load_config is called
# Assuming it is load_config based on typical structure:
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt")
    args = parser.parse_args()

    # 1. Load config directly
    print(f"[DEBUG] Loading config from: {args.config}")
    opt = load_config(args.config)

    # 2. Instantiate model
    # HOLD(opt) might expect 'opt' to have certain structure
    # If HOLD expects (opt, args) like in train.py, we might need to fake 'args'
    # Let's try minimal instantiation first

    # Mock an 'args' object if HOLD.__init__ requires it for non-critical things
    class MockArgs:
        def __getattr__(self, name):
            return None

    try:
        # Try initializing with just opt (standard usually)
        model = HOLD(opt)
    except TypeError:
        # If it demands 2 arguments: HOLD(opt, args)
        print("[DEBUG] HOLD requires (opt, args), providing mock args")
        model = HOLD(opt, MockArgs())

    # 3. Load checkpoint
    print(f"[DEBUG] Loading checkpoint from: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[DEBUG] Checkpoint loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    # 4. Inspect Dims
    print("\n" + "=" * 40)
    print(" NETWORK DIMENSION INSPECTION")
    print("=" * 40)

    try:
        print(f" Body Rendering Net 'dims': {model.model.rendering_network.dims}")
        # Also check input dimension specifically if stored
        if hasattr(model.model.rendering_network, 'd_in'):
            print(f" Body Rendering Net 'd_in': {model.model.rendering_network.d_in}")
    except Exception as e:
        print(f" Error inspecting body rendering net: {e}")

    try:
        print(f" BG Rendering Net 'dims':   {model.model.background.bg_rendering_network.dims}")
        if hasattr(model.model.background.bg_rendering_network, 'd_in'):
            print(f" BG Rendering Net 'd_in':   {model.model.background.bg_rendering_network.d_in}")
    except Exception as e:
        print(f" Error inspecting BG rendering net: {e}")

    print("=" * 40 + "\n")


if __name__ == "__main__":
    main()