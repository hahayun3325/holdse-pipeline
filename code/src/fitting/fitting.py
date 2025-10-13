import sys
import numpy as np
import sys
import numpy as np


sys.path = [".."] + sys.path
import pickle as pkl

with open("./body_models/contact_zones.pkl", "rb") as f:
    contact_zones = pkl.load(f)
contact_zones = contact_zones["contact_zones"]
contact_idx = np.array([item for sublist in contact_zones.values() for item in sublist])

import os
from src.fitting.utils import scaling_masks_K, extract_batch_data
from src.fitting.model import Model
# ========================================================================
# PHASE 4: GHOP Contact Refinement Imports
# ========================================================================
from src.model.ghop.contact_refinement import GHOPContactRefinement
from loguru import logger
# ========================================================================

def optimize_batch(
    batch_idx,
    args,
    pbar,
    out,
    device,
    obj_scale=None,
    freeze_scale=False,
    freeze_shape=False,
    hand_shapes=None,  # Added for optimize_ckpt.py compatibility
    # Phase 4 parameters
    use_phase4_contact=False,
    contact_thresh=0.01,
    collision_thresh=0.005,
):
    """Optimize batch with optional Phase 4 GHOP contact refinement.

    This function performs iterative optimization of hand pose, hand shape, and
    object parameters from multi-view images. With Phase 4 enabled, it applies
    GHOP contact refinement for physically plausible hand-object interaction.

    Args:
        batch_idx: List of frame indices to optimize
        args: Argument namespace containing:
            - iters: Number of optimization iterations
            - vis_every: Visualization frequency
            - write_gif: Whether to write GIF visualization
            - itw: In-the-wild mode flag
            - phase4_contact: Enable Phase 4 (optional, can use use_phase4_contact)
        pbar: tqdm progress bar instance (can be None)
        out: Data dictionary from load_data() containing:
            - servers: MANO/object model servers
            - fnames: Image file paths
            - K: Camera intrinsics
            - faces: Mesh face indices
        device: Torch device ('cuda' or 'cpu')
        obj_scale: Initial object scale (optional)
        freeze_scale: Whether to freeze object scale during optimization
        freeze_shape: Whether to freeze hand shape during optimization
        hand_shapes: Pre-computed hand shape parameters (for optimize_ckpt.py)
        use_phase4_contact: Enable Phase 4 contact refinement
        contact_thresh: Contact distance threshold in meters (default: 0.01 = 1cm)
        collision_thresh: Collision penetration threshold in meters (default: 0.005 = 5mm)

    Returns:
        model: Optimized Model instance with fitted parameters
    """
    print("=" * 70)
    print(f"Optimizing batch with indices: {batch_idx}")
    print("=" * 70)

    # Prepare mask paths by replacing '/image/' with '/mask/'
    mask_ps = [fname.replace("/image/", "/mask/") for fname in out["fnames"]]

    # Extract batch-specific data
    masks_batch, scene_scale, param_batch, fnames_batch, w2c_batch = extract_batch_data(
        batch_idx, out, mask_ps, device, args.itw
    )

    # Prepare output paths for visualization
    out_paths = [f"./vis/{idx:05d}.gif" for idx in batch_idx]
    os.makedirs(os.path.dirname(out_paths[0]), exist_ok=True)

    # Scale masks and camera intrinsics to target resolution
    masks_batch, K_scaled = scaling_masks_K(masks_batch, out["K"], target_dim=300)

    # Create optimization model
    model = Model(
        out["servers"],
        scene_scale,
        obj_scale,
        param_batch,
        device,
        masks_batch,
        w2c_batch,
        K_scaled,
        fnames_batch,
        out["faces"],
    )

    # Setup optimization parameters
    model.pbar = pbar
    model.defrost_all()
    model.obj_scale.requires_grad = not freeze_scale

    # Configure parameter gradients
    for k in model.param_dict.keys():
        if "betas" in k and freeze_shape:
            model.param_dict[k].requires_grad = False
        if "pose" in k:
            model.param_dict[k].requires_grad = False
        if "global_orient" in k and "object" not in k:
            model.param_dict[k].requires_grad = False
        if "scene_scale" in k:
            model.param_dict[k].requires_grad = False
    model.print_requires_grad()
    model.setup_optimizer()

    # ====================================================================
    # PHASE 4: Contact-Aware Fitting
    # ====================================================================
    # Determine if Phase 4 contact refinement should be used
    # Priority: 1) use_phase4_contact parameter, 2) args.phase4_contact, 3) disabled
    phase4_enabled = use_phase4_contact

    if not phase4_enabled and hasattr(args, 'phase4_contact'):
        phase4_enabled = args.phase4_contact

    if phase4_enabled:
        logger.info("\n[Phase 4] Contact-aware fitting enabled")
        logger.info(f"[Phase 4] Contact threshold: {contact_thresh}m")
        logger.info(f"[Phase 4] Collision threshold: {collision_thresh}m")

        # Initialize GHOP contact refiner
        try:
            contact_refiner = GHOPContactRefinement(
                contact_thresh=contact_thresh,
                collision_thresh=collision_thresh,
                contact_zones='zones'
            )
            logger.info("[Phase 4] ✓ Contact refiner initialized successfully")

            # Check if model has fit_with_contact method
            if hasattr(model, 'fit_with_contact'):
                # Use Phase 4 contact-aware fitting
                logger.info("[Phase 4] Running contact-aware optimization...")
                model.fit_with_contact(
                    num_iterations=args.iters,
                    contact_refiner=contact_refiner,
                    vis_every=args.vis_every,
                    write_gif=args.write_gif,
                    out_ps=out_paths,
                )
                logger.info("[Phase 4] ✓ Contact-aware fitting complete\n")
            else:
                # Fallback: Model doesn't have fit_with_contact method yet
                logger.warning("[Phase 4] Model.fit_with_contact() not implemented yet")
                logger.warning("[Phase 4] Falling back to standard fitting")
                logger.warning("[Phase 4] To use contact refinement, implement fit_with_contact() in Model class\n")
                model.fit(
                    num_iterations=args.iters,
                    vis_every=args.vis_every,
                    write_gif=args.write_gif,
                    out_ps=out_paths,
                )

        except Exception as e:
            logger.error(f"[Phase 4] Contact refinement initialization failed: {e}")
            logger.error("[Phase 4] Falling back to standard fitting")
            import traceback
            traceback.print_exc()

            # Fallback to standard fitting
            model.fit(
                num_iterations=args.iters,
                vis_every=args.vis_every,
                write_gif=args.write_gif,
                out_ps=out_paths,
            )
    else:
        # Standard fitting (no Phase 4)
        logger.debug("[Fitting] Using standard optimization (Phase 4 disabled)")
        model.fit(
            num_iterations=args.iters,
            vis_every=args.vis_every,
            write_gif=args.write_gif,
            out_ps=out_paths,
        )
    # ====================================================================

    print("=" * 70)
    print("Batch optimization complete")
    print("=" * 70)

    return model
