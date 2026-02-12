import torch
import os
import os.path as op
import numpy as np
from PIL import Image
import sys
from loguru import logger
try:
    import mcubes
except ImportError:
    mcubes = None
    logger.warning("mcubes not available, mesh extraction will be skipped")
sys.path = [".."] + sys.path
from loguru import logger

from src.hold.hold import HOLD
from src.utils.parser import parser_args
from src.datasets.utils import create_dataset
import common.thing as thing
from common.torch_utils import reset_all_seeds
from src.utils import vis_utils


def save_normal_image(normal_tensor, save_path, mask=None, norm_thresh=0.5):
    """
    Save normal map as RGB image with white background.
    Input: normal_tensor in range [-1, 1], shape [H, W, 3] or [3, H, W]
    """
    # Ensure shape is [H, W, 3]
    if normal_tensor.dim() == 3:
        if normal_tensor.shape[0] == 3:
            normal_tensor = normal_tensor.permute(1, 2, 0)

    # Convert from [-1, 1] to [0, 255]
    normal_img = (normal_tensor.detach().cpu().numpy() + 1.0) / 2.0

    # Create white background mask where normal is invalid/background
    # Method 1: Use explicit mask if provided by model output
    if mask is not None:
        bg_mask = ~mask.cpu().numpy().astype(bool)
        normal_img[bg_mask] = 1.0  # White background
    else:
        # Use SDF-gradient magnitude to define foreground
        # Convert back to [-1, 1] then compute norm
        normals_unit = normal_img * 2.0 - 1.0
        norm = np.linalg.norm(normals_unit, axis=-1)

        # Anything with large gradient is near surface → object
        fg_mask = norm > norm_thresh
        bg_mask = ~fg_mask
        normal_img[bg_mask] = 1.0

    # Clip and convert to uint8
    normal_img = (normal_img * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(normal_img).save(save_path)


def save_normal_image_raw(normal_tensor, save_path):
    """
    Save normal map WITHOUT any background masking (for debugging).
    Shows raw normals even if they would be masked as background.
    """
    # Ensure shape is [H, W, 3]
    if normal_tensor.dim() == 3:
        if normal_tensor.shape[0] == 3:
            normal_tensor = normal_tensor.permute(1, 2, 0)

    # Convert from [-1, 1] to [0, 255] without background masking
    normal_img = (normal_tensor.detach().cpu().numpy() + 1.0) / 2.0
    normal_img = (normal_img * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(normal_img).save(save_path)


def reshape_normal(normal, img_size):
    """
    Reshape flattened normal to [H, W, 3]
    normal: [N, 3] or [3, N] where N = H * W
    img_size: [H, W]
    """
    # Handle different input shapes
    if normal.dim() == 2:
        # If [3, N], transpose first
        if normal.shape[0] == 3:
            normal = normal.permute(1, 0)  # [N, 3]

        # Now reshape [N, 3] to [H, W, 3]
        H, W = img_size[0], img_size[1]
        normal = normal.view(H, W, 3)

    elif normal.dim() == 3:
        # If [3, H, W], permute to [H, W, 3]
        if normal.shape[0] == 3:
            normal = normal.permute(1, 2, 0)

    return normal


def main():
    args, opt = parser_args()
    device = "cuda:0"

    # Fix test data path
    if hasattr(opt.dataset, 'dataset_path'):
        base_path = opt.dataset.dataset_path
        if not hasattr(opt.dataset, 'test'):
            from easydict import EasyDict
            opt.dataset.test = EasyDict()
        opt.dataset.test.data_dir = op.join(base_path, args.case)
        opt.dataset.test.seq_name = args.case

    # Setup output directories
    exp_key = args.load_ckpt.split("/")[1]
    output_dir = op.join("logs", exp_key, "test", "normals")

    hand_dir = op.join(output_dir, "hand")
    object_dir = op.join(output_dir, "object")
    combined_dir = op.join(output_dir, "combined")

    for d in [hand_dir, object_dir, combined_dir]:
        os.makedirs(d, exist_ok=True)

    logger.info(f"Test data dir: {opt.dataset.test.data_dir}")
    logger.info(f"Saving normals to: {output_dir}")

    # Initialize model
    model = HOLD(opt, args)
    testset = create_dataset(opt.dataset.test, args)
    logger.info(f"Rendering {len(testset)} frames...")

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
    saved_count = 0

    for idx, batch in enumerate(testset):
        with torch.no_grad():
            batch = thing.thing2dev(batch, device)

            # Standard inference - now uses SDF grid internally for object
            out = model.inference_step(batch)

            # Extract image size first!
            img_size = out["img_size"]  # [H, W]

            # The object normals should now come from the fixed inference path
            if "object.normal" in out:
                normal = reshape_normal(out["object.normal"], img_size)
                # Save directly - no need for manual SDF processing
                save_normal_image(normal, op.join(object_dir, f"frame_{idx:04d}.png"))

            # Process hand normals
            if "right.normal" in out:
                normal = reshape_normal(out["right.normal"], img_size)
                hand_mask = out.get("right.mask", out.get("mask", None))
                save_normal_image(normal, op.join(hand_dir, f"frame_{idx:04d}.png"),
                                mask=hand_mask, norm_thresh=0.3)

            # Process combined normals
            if "normal" in out:
                combined = reshape_normal(out["normal"], img_size)
                save_normal_image(combined, op.join(combined_dir, f"frame_{idx:04d}.png"),
                                mask=None, norm_thresh=0.5)

    logger.info(f"\nComplete! Saved {saved_count} normal maps to {output_dir}")


if __name__ == "__main__":
    main()

'''
export COMET_API_KEY="4hhuylWTxYQBirmxKwuwGv4Q5"
export COMET_WORKSPACE="cloudy"
python render_normals.py \
  --case hold_MC1_ho3d \
  --load_ckpt logs/cb20a1702/checkpoints/last.ckpt \
  --config confs/render_stage3_hold_MC1_ho3d_sds_from_official.yaml \
  --mute \
  --agent_id -1
python render_normals.py \
  --case hold_MC1_ho3d \
  --load_ckpt logs/7e83a1f92_000035000/checkpoints/last.ckpt \
  --config confs/render_stage3_hold_MC1_ho3d_sds_from_official.yaml \
  --mute \
  --agent_id -1
'''

'''
3. If you want a thicker, filled‑in silhouette
If the band is too thin, you can “inflate” it by relaxing the threshold, e.g., norm_thresh = 0.5, or by post‑processing:

After computing fg_mask = norm > norm_thresh, apply a small dilation (e.g., using scipy.ndimage.binary_dilation) before setting background to white.

This will turn the thin band around the zero‑level set into a thicker, more visually apparent silhouette.

If you later want a true camera‑view silhouette, you’ll need to replace the mid‑slice trick with SDF ray‑marching along the camera rays, but the norm‑threshold change above is the minimal modification that uses your already‑correct SDF normals to give a recognizable object outline.
'''