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
            # Check which SDF source inference_step will actually use
            obj_model = model.model.nodes['object'].server.object_model
            print(f"SDF grid attr: {hasattr(obj_model, 'sdf_grid')}")
            print(f"SDF grid shape: {obj_model.sdf_grid.shape if hasattr(obj_model, 'sdf_grid') else 'N/A'}")
            print(f"SDF grid range: [{obj_model.sdf_grid.min():.4f}, {obj_model.sdf_grid.max():.4f}]" if hasattr(
                obj_model, 'sdf_grid') else 'N/A')
            # Replace the entire EXPERIMENTAL block (lines ~162-283) with:
            out = model.inference_step(batch)

            # The object normals should now come from the fixed inference path
            if "object.normal" in out:
                normal = reshape_normal(out["object.normal"], img_size)
                # Save directly - no need for manual SDF processing
                save_normal_image(normal, op.join(object_dir, f"frame_{idx:04d}.png"))

            # DEBUG: Check what's available in model structure
            logger.info(f"  [DEBUG] Checking model structure...")
            logger.info(f"  [DEBUG] hasattr(model, 'model'): {hasattr(model, 'model')}")
            if hasattr(model, 'model'):
                logger.info(f"  [DEBUG] hasattr(model.model, 'nodes'): {hasattr(model.model, 'nodes')}")
                if hasattr(model.model, 'nodes'):
                    logger.info(f"  [DEBUG] Number of nodes: {len(model.model.nodes)}")
                    for node in model.model.nodes.values():
                        logger.info(f"  [DEBUG] Node ID: {node.node_id}")
                        if "object" in node.node_id.lower():
                            logger.info(f"    [DEBUG] Object node found!")
                            logger.info(f"    [DEBUG] Attributes: {[attr for attr in dir(node) if not attr.startswith('_')]}")
                            logger.info(f"    [DEBUG] hasattr(node, 'sdf_grid'): {hasattr(node, 'sdf_grid')}")
                            # Check alternative attribute names
                            for attr in ['sdf_grid', 'grid', 'sdf', 'sdf_volume', 'volume']:
                                if hasattr(node, attr):
                                    logger.info(f"    [DEBUG] Found alternative attribute: {attr}")

            # Compare fresh SDF with checkpoint's stored sdf_grid
            if hasattr(model, 'model') and hasattr(model.model, 'nodes'):
                for node in model.model.nodes.values():
                    if "object" in node.node_id.lower():
                        # Check the correct path: node.server.object_model.sdf_grid
                        if hasattr(node, 'server') and hasattr(node.server, 'object_model'):
                            obj_model = node.server.object_model
                            if hasattr(obj_model, 'sdf_grid'):
                                stored_sdf = obj_model.sdf_grid
                                logger.info(f"  [COMPARISON] Found stored sdf_grid at server.object_model")
                                logger.info(f"  [COMPARISON] Stored SDF range: [{stored_sdf.min().item():.4f}, {stored_sdf.max().item():.4f}]")
                                logger.info(f"  [COMPARISON] Stored SDF shape: {stored_sdf.shape}")
                                logger.info(f"  [COMPARISON] Stored SDF std: {stored_sdf.std().item():.4f}")

                                # Zero-crossings on stored SDF
                                if stored_sdf.dim() == 4:
                                    B, H, W, D = stored_sdf.shape
                                    C = 1
                                elif stored_sdf.dim() == 5:
                                    B, C, H, W, D = stored_sdf.shape
                                flat = stored_sdf.view(-1)
                                zc = ((flat[:-1] * flat[1:]) < 0).sum().item()
                                logger.info(f"  [COMPARISON] Stored SDF zero-crossings: {zc}")

                            else:
                                logger.info(f"  [COMPARISON] server.object_model has no sdf_grid")
                        else:
                            logger.info(f"  [COMPARISON] node has no server or server has no object_model")


            # Get image size for reshaping
            img_size = out["img_size"]  # [H, W]

            # Process combined normal: prefer SDF-based object normals if available
            combined = None
            if "object.normal" in out:
                combined = out["object.normal"]        # SDF-based normals you just wrote
            elif "normal" in out:
                combined = out["normal"]               # Fallback to whatever the model produced

            if isinstance(combined, torch.Tensor):
                combined_img = reshape_normal(combined, img_size)
                # For combined, also let SDF gradient drive the silhouette
                save_normal_image(
                    combined_img,
                    op.join(combined_dir, f"frame_{idx:04d}.png"),
                    mask=None,
                    norm_thresh=0.5
                )
                saved_count += 1

            # Process hand normal - USE MODEL MASK
            if "right.normal" in out:
                normal = out["right.normal"]
                if isinstance(normal, torch.Tensor):
                    normal = reshape_normal(normal, img_size)
                    # Try to get hand-specific mask, fallback to generic mask
                    hand_mask = out.get("right.mask", out.get("mask", None))
                    save_normal_image(
                        normal,
                        op.join(hand_dir, f"frame_{idx:04d}.png"),
                        mask=hand_mask,      # Use model mask for hand
                        norm_thresh=0.3      # Lower threshold as safety
                    )

            # Process object normal - USE SDF WITH GRADIENT MASK
            if "object.normal" in out:
                # Extract mask ONCE at the beginning for both visualization and logging
                mask = out.get("mask", None)

                normal = reshape_normal(out["object.normal"], img_size)

                # Recompute gradient magnitude for masking if not already done
                if 'grad_mag_img' not in locals():
                    # Fallback: simple magnitude-based mask
                    object_mask = None
                    norm_thresh = 0.3  # More permissive for SDF normals
                else:
                    object_mask = grad_mag_img > 0.05  # Low threshold since SDF has strong gradients

                save_normal_image(
                    normal,
                    op.join(object_dir, f"frame_{idx:04d}.png"),
                    mask=object_mask,
                    norm_thresh=0.3
                )

                # Also save raw for debugging
                raw_dir = op.join(output_dir, "object_raw")
                os.makedirs(raw_dir, exist_ok=True)
                save_normal_image_raw(normal, op.join(raw_dir, f"frame_{idx:04d}.png"))

                # 3. If mask exists, also save mask visualization (now mask is defined!)
                if mask is not None:
                    mask_dir = op.join(output_dir, "mask_vis")
                    os.makedirs(mask_dir, exist_ok=True)
                    mask_img = (mask.cpu().numpy() * 255).astype(np.uint8)
                    Image.fromarray(mask_img).save(op.join(mask_dir, f"frame_{idx:04d}.png"))

                # Log information
                normal = out["object.normal"]
                logger.info(f"=== Frame {idx} Diagnostics ===")
                logger.info(f"  object.normal exists: True")
                logger.info(f"  object.normal shape: {normal.shape}")
                logger.info(f"  object.normal range: [{normal.min().item():.4f}, {normal.max().item():.4f}]")

                # Compute norms for debugging
                if normal.dim() == 2:
                    if normal.shape[0] == 3:
                        normal_for_norm = normal.permute(1, 0)
                    else:
                        normal_for_norm = normal
                else:
                    normal_for_norm = normal.view(-1, 3)

                norms = torch.norm(normal_for_norm, dim=-1)
                logger.info(f"  normal norms: min={norms.min().item():.4f}, max={norms.max().item():.4f}")
                logger.info(f"  near-zero normals (norm<0.1): {(norms < 0.1).sum().item()}/{norms.numel()}")

                # Log mask info (mask is already defined above)
                if mask is not None:
                    logger.info(f"  mask exists: True")
                    logger.info(f"  mask shape: {mask.shape}")
                    logger.info(f"  mask range: [{mask.min().item():.4f}, {mask.max().item():.4f}]")
                    logger.info(f"  mask true count: {mask.sum().item()}/{mask.numel()}")
                    logger.info(f"  mask false count: {(~mask).sum().item()}/{mask.numel()}")
                else:
                    logger.info(f"  mask exists: False (will use norm heuristic)")
            # Also try vis_utils extraction if raw normals not available
            if saved_count == 0 and idx == 0:
                try:
                    img_size = out["img_size"]
                    vis_dict = vis_utils.output2images([out], img_size)
                    logger.info(f"vis_utils extracted keys: {list(vis_dict.keys())}")

                    # Note: vis_utils outputs are already images in [0, 255]
                    if "normal" in vis_dict:
                        Image.fromarray(vis_dict["normal"]).save(
                            op.join(combined_dir, f"frame_{idx:04d}.png")
                        )
                        saved_count += 1
                except Exception as e:
                    if idx == 0:
                        logger.info(f"vis_utils extraction failed: {e}")

            if idx % 10 == 0:
                logger.info(f"Rendered frame {idx}/{len(testset)}, saved {saved_count} so far")

    logger.info(f"\nComplete! Saved {saved_count} normal maps to {output_dir}")


if __name__ == "__main__":
    main()

'''
export COMET_API_KEY="4hhuylWTxYQBirmxKwuwGv4Q5"
export COMET_WORKSPACE="cloudy"
python render_normals.py \
  --case hold_MC1_ho3d \
  --load_ckpt logs/a91d56935_000001000/checkpoints/last.ckpt \
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