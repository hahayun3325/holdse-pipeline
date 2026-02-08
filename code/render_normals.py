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


def save_normal_image(normal_tensor, save_path, mask=None):
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
        # Method 2: Detect by near-zero norm (unoccupied space)
        norm = np.linalg.norm(normal_img * 2 - 1, axis=-1)  # Back to [-1,1] then compute norm
        # Norm should be ~1.0 for valid normals, near 0 for empty/background
        bg_mask = norm < 0.1
        normal_img[bg_mask] = 1.0  # White background

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
            out = model.inference_step(batch)
            # Get image size early (needed for EXPERIMENTAL section)
            img_size = out["img_size"]  # [H, W]

            # ====================================================================
            # EXPERIMENTAL: Extract SDF directly from implicit network (hypothesis test)
            # This bypasses the checkpoint's stored sdf_grid parameter
            # ====================================================================
            try:
                # Create directory for extracted normals
                extracted_dir = op.join(output_dir, "object_extracted")
                os.makedirs(extracted_dir, exist_ok=True)

                # Use STORED SDF grid from checkpoint (not fresh extraction)
                # The stored SDF is valid, but the implicit network has drifted
                logger.info(f"  [EXPERIMENT] Using STORED sdf_grid from checkpoint...")

                # Get stored SDF from object_node.server.object_model.sdf_grid
                object_node = None
                for node in model.model.nodes.values():
                    if "object" in node.node_id.lower():
                        object_node = node
                        break

                if object_node is not None and hasattr(object_node.server, 'object_model'):
                    obj_model = object_node.server.object_model
                    if hasattr(obj_model, 'sdf_grid'):
                        # Use the stored SDF grid directly
                        sdf_grid_stored = obj_model.sdf_grid  # [1, 64, 64, 64]
                        logger.info(f"  [EXPERIMENT] Loaded stored SDF: shape={sdf_grid_stored.shape}, range=[{sdf_grid_stored.min():.4f}, {sdf_grid_stored.max():.4f}]")
                    else:
                        logger.error("  [EXPERIMENT] No sdf_grid found in object_model, falling back to fresh extraction")
                        sdf_grid_stored = model._extract_sdf_grid_from_nodes(batch, resolution=64)
                else:
                    logger.error("  [EXPERIMENT] Cannot access object node or object_model, falling back to fresh extraction")
                    sdf_grid_stored = model._extract_sdf_grid_from_nodes(batch, resolution=64)


                # Log the STORED SDF statistics (not fresh)
                sdf_min = sdf_grid_stored.min().item()
                sdf_max = sdf_grid_stored.max().item()
                sdf_std = sdf_grid_stored.std().item()
                logger.info(f"  [EXPERIMENT] STORED SDF stats: min={sdf_min:.4f}, max={sdf_max:.4f}, std={sdf_std:.4f}")

                # Compute zero-crossings on STORED SDF
                # Handle 4D shape [1, 64, 64, 64]
                if sdf_grid_stored.dim() == 4:
                    B, H, W, D = sdf_grid_stored.shape
                    sdf_flat = sdf_grid_stored.view(-1)
                else:  # 5D [B, C, H, W, D]
                    B, C, H, W, D = sdf_grid_stored.shape
                    sdf_flat = sdf_grid_stored.view(-1)

                sign_changes = ((sdf_flat[:-1] * sdf_flat[1:]) < 0).sum().item()
                logger.info(f"  [EXPERIMENT] STORED SDF zero-crossings: {sign_changes}")

                # ============================================================================
                # COMPUTE NORMALS FROM STORED SDF (bypass drifted implicit network)
                # ============================================================================
                try:
                    # Compute SDF gradient (which gives surface normals at the zero crossing)
                    # stored SDF shape: [1, 64, 64, 64]
                    sdf_grid = sdf_grid_stored[0]  # [64, 64, 64]

                    # Compute gradient using finite differences
                    grad_x = torch.zeros_like(sdf_grid)
                    grad_y = torch.zeros_like(sdf_grid)
                    grad_z = torch.zeros_like(sdf_grid)

                    grad_x[1:-1, :, :] = (sdf_grid[2:, :, :] - sdf_grid[:-2, :, :]) / 2.0
                    grad_y[:, 1:-1, :] = (sdf_grid[:, 2:, :] - sdf_grid[:, :-2, :]) / 2.0
                    grad_z[:, :, 1:-1] = (sdf_grid[:, :, 2:] - sdf_grid[:, :, :-2]) / 2.0

                    # Normalize to get unit normals
                    grad_norm = torch.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2 + 1e-8)
                    normals_grid = torch.stack([
                        grad_x / grad_norm,
                        grad_y / grad_norm,
                        grad_z / grad_norm
                    ], dim=-1)  # [64, 64, 64, 3]

                    # Sample normals at ray points (or use the grid directly for the view)
                    # For now, project the normals to the image plane
                    # This is a simplified projection - ideally you'd march rays through the grid

                    # Get image size
                    H_img, W_img = img_size[0], img_size[1]

                    # Create a simple orthographic projection of the normals
                    # Take the middle slice or average along one axis
                    normal_img = normals_grid[:, :, 32, :].permute(2, 0, 1)  # [3, 64, 64]

                    # Resize to match output image
                    import torch.nn.functional as F
                    normal_img = F.interpolate(
                        normal_img.unsqueeze(0),
                        size=(H_img, W_img),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).permute(1, 2, 0)  # [H, W, 3]

                    # Store in output dictionary for saving
                    # Delete existing key if present (from drifted implicit network)
                    if "object.normal" in out:
                        del out["object.normal"]
                    out["object.normal"] = normal_img.reshape(-1, 3)
                    out["normal"] = out["object.normal"].clone()

                    logger.info(f"  [EXPERIMENT] Computed normals from stored SDF: shape={out['object.normal'].shape}")
                    logger.info(
                        f"  [EXPERIMENT] Normal range: [{out['object.normal'].min():.4f}, {out['object.normal'].max():.4f}]")

                except Exception as e:
                    logger.error(f"  [EXPERIMENT] Failed to compute normals from SDF: {e}")
                    import traceback
                    logger.error(traceback.format_exc())

            except Exception as e:
                logger.error(f"  [EXPERIMENT] SDF extraction failed: {e}")
                import traceback
                logger.error(traceback.format_exc())

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
                save_normal_image(combined_img, op.join(combined_dir, f"frame_{idx:04d}.png"))
                saved_count += 1

            # Process hand normal (right.normal)
            if "right.normal" in out:
                normal = out["right.normal"]
                if isinstance(normal, torch.Tensor):
                    normal = reshape_normal(normal, img_size)
                    save_normal_image(normal, op.join(hand_dir, f"frame_{idx:04d}.png"))

            # In the render loop, extract mask if available
            img_size = out["img_size"]
            mask = out.get("mask", None)  # or "weights", "alpha", "acc_norm"

            if "object.normal" in out:
                normal = reshape_normal(out["object.normal"], img_size)

                # 1. Save standard masked version (existing behavior)
                save_normal_image(normal, op.join(object_dir, f"frame_{idx:04d}.png"), mask=mask)

                # 2. Save raw unmasked version (for hypothesis testing)
                raw_dir = op.join(output_dir, "object_raw")
                os.makedirs(raw_dir, exist_ok=True)
                save_normal_image_raw(normal, op.join(raw_dir, f"frame_{idx:04d}.png"))

                # 3. If mask exists, also save mask visualization
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

                # Log mask info
                mask = out.get("mask", None)
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
  --load_ckpt logs/4f425897c_000035000/checkpoints/last.ckpt \
  --config confs/render_stage3_hold_MC1_ho3d_sds_from_official.yaml \
  --mute \
  --agent_id -1
'''