import torch
import os
import os.path as op
import numpy as np
from PIL import Image
import argparse

import sys
sys.path = [".."] + sys.path

from src.hold.hold import HOLD
from src.utils.parser import parser_args  # Use original parser
from src.datasets.utils import create_dataset
import common.thing as thing
from common.torch_utils import reset_all_seeds


def save_normal_image(normal_tensor, save_path):
    """
    Save normal map as RGB image.
    Normals are typically in range [-1, 1] with (x, y, z) components.
    Map to [0, 255] for visualization: [-1, 1] -> [0, 255]
    """
    # Convert from [-1, 1] to [0, 1]
    normal_img = (normal_tensor.cpu().numpy() + 1.0) / 2.0
    # Clip and convert to uint8
    normal_img = (normal_img * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(normal_img).save(save_path)


def extract_normals_from_output(out):
    """
    Extract normal maps from model output.
    Returns dict with keys: hand, object, combined (or None if not present)
    """
    normals = {}

    # Try common normal output keys
    key_mapping = {
        'hand': ['hand_normals', 'normals_hand', 'hand_normal', 'normal_hand', 'normals_hand_pred'],
        'object': ['object_normals', 'normals_object', 'obj_normals', 'object_normal', 'normal_object', 'normals_obj_pred'],
        'combined': ['normals', 'normal_map', 'normals_pred', 'combined_normals', 'all_normals', 'normals_gt']
    }

    for category, possible_keys in key_mapping.items():
        for key in possible_keys:
            if key in out:
                normals[category] = out[key]
                break

    return normals


def main():
    args, opt = parser_args()

    # Fix: Set the data_dir for test dataset using the case argument
    # The config has dataset.dataset_path, need to set test.data_dir
    if hasattr(opt.dataset, 'dataset_path'):
        base_path = opt.dataset.dataset_path
        if not hasattr(opt.dataset, 'test'):
            from easydict import EasyDict
            opt.dataset.test = EasyDict()
        opt.dataset.test.data_dir = op.join(base_path, args.case)
        opt.dataset.test.seq_name = args.case

    # Override output directory if provided
    device = "cuda:0"
    if hasattr(args, 'output_dir') and args.output_dir:
        output_dir = args.output_dir
    else:
        exp_key = args.load_ckpt.split("/")[1]
        output_dir = op.join("logs", exp_key, "test", "normals")

    # Setup directories
    hand_dir = op.join(output_dir, "hand")
    object_dir = op.join(output_dir, "object")
    combined_dir = op.join(output_dir, "combined")

    for d in [hand_dir, object_dir, combined_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"Test data dir: {opt.dataset.test.data_dir}")
    print(f"Saving normals to: {output_dir}")

    # Initialize model
    model = HOLD(opt, args)
    testset = create_dataset(opt.dataset.test, args)
    print(f"Rendering {len(testset)} frames...")

    # Load checkpoint
    ckpt = torch.load(args.load_ckpt, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.to(device)
    model.eval()

    # Disable BARF masks
    if hasattr(model, 'model') and hasattr(model.model, 'nodes'):
        nodes = model.model.nodes
        for node in nodes.values():
            if hasattr(node, 'implicit_network'):
                node.implicit_network.embedder_obj.eval()

        if hasattr(model.model, 'background'):
            bg = model.model.background
            if hasattr(bg, 'bg_implicit_network'):
                bg.bg_implicit_network.embedder_obj.eval()
            if hasattr(bg, 'bg_rendering_network'):
                bg.bg_rendering_network.embedder_obj.eval()

    # Render loop
    reset_all_seeds(1)
    has_hand = has_object = has_combined = False

    for idx, batch in enumerate(testset):
        with torch.no_grad():
            batch = thing.thing2dev(batch, device)
            out = model.inference_step(batch)

            # Extract and save normals
            normals = extract_normals_from_output(out)

            # Process and save each normal type
            for category, normal_tensor in normals.items():
                if isinstance(normal_tensor, torch.Tensor):
                    # Handle batch dimension
                    if normal_tensor.dim() == 4:
                        normal_tensor = normal_tensor[0]  # [B, H, W, 3] -> [H, W, 3]

                    # Ensure shape is [H, W, 3]
                    if normal_tensor.shape[0] == 3:
                        normal_tensor = normal_tensor.permute(1, 2, 0)

                    if category == 'hand':
                        save_normal_image(normal_tensor, op.join(hand_dir, f"frame_{idx:04d}.png"))
                        has_hand = True
                    elif category == 'object':
                        save_normal_image(normal_tensor, op.join(object_dir, f"frame_{idx:04d}.png"))
                        has_object = True
                    elif category == 'combined':
                        save_normal_image(normal_tensor, op.join(combined_dir, f"frame_{idx:04d}.png"))
                        has_combined = True

            if idx % 10 == 0:
                print(f"Rendered frame {idx}/{len(testset)}")

    print(f"\nComplete!")
    print(f"  Hand normals:    {'Yes' if has_hand else 'No'}")
    print(f"  Object normals:  {'Yes' if has_object else 'No'}")
    print(f"  Combined normals:{'Yes' if has_combined else 'No'}")

    if not (has_hand or has_object or has_combined):
        print("\nWARNING: No normal outputs found in model inference!")
        print("Available output keys (from last frame):")
        print(list(out.keys()))


if __name__ == "__main__":
    main()

'''
export COMET_API_KEY="4hhuylWTxYQBirmxKwuwGv4Q5"
export COMET_WORKSPACE="cloudy"
python render_normals.py \
  --case hold_MC1_ho3d \
  --load_ckpt logs/7dacf8bc6_000036000/checkpoints/last.ckpt \
  --config confs/render_stage3_hold_MC1_ho3d_sds_from_official.yaml \
  --mute \
  --agent_id -1
'''