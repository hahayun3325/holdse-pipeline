import numpy as np
import torch
from tqdm import tqdm
from src.fitting.fitting import optimize_batch
from src.utils.io.optim import load_data
from common.xdict import xdict

# ========================================================================
# PHASE 4: GHOP Contact Refinement Imports
# ========================================================================
from src.model.ghop.mesh_extraction import GHOPMeshExtractor
from src.model.ghop.contact_refinement import GHOPContactRefinement
from loguru import logger
import trimesh
from skimage import measure
# ========================================================================


def fit_ckpt(args):
    """Main checkpoint optimization function with optional Phase 4 contact refinement.

    Args:
        args: Argument namespace containing:
            - ckpt_p: Path to checkpoint file
            - batch_size: Number of frames to optimize simultaneously
            - iters: Number of iterations per frame
            - phase4_contact: Enable Phase 4 contact refinement (default: False)
            - contact_iters: Number of contact optimization iterations (default: 100)
            - mesh_resolution: Resolution for mesh extraction (default: 128)
            - vqvae_checkpoint: Path to GHOP VQ-VAE checkpoint (required for Phase 4)
    """
    device = "cuda"
    out, ckpt = load_data(args.ckpt_p)

    node_ids = out["node_ids"]
    num_frames = out["num_frames"]
    batch_size = args.batch_size

    obj_scale = out["servers"]["object"].object_model.obj_scale.cpu().detach().numpy()

    # Extract initial hand shapes
    hand_shapes = xdict()
    for node_id in node_ids:
        if "right" in node_id or "left" in node_id:
            shape_key = f"model.nodes.{node_id}.params.betas.weight"
            initial_shape = ckpt["state_dict"][shape_key].clone().cpu().detach().numpy()
            hand_shapes[node_id] = initial_shape

    batch_idx = (
        torch.linspace(0, num_frames - 1, steps=batch_size).floor().long().tolist()
    )

    # ====================================================================
    # STAGE 1: Optimize object scale and hand shape
    # ====================================================================
    print("\n" + "=" * 70)
    print("Stage [1/3]: Optimizing object scale and hand shape")
    print("=" * 70)
    pbar = None
    model = optimize_batch(
        batch_idx,
        args,
        pbar,
        out,
        device,
        obj_scale=obj_scale,
        freeze_scale=False,
        freeze_shape=False,
        hand_shapes=None,  # Not used in Stage 1
        # Phase 4 parameters (disabled in Stage 1 for faster convergence)
        use_phase4_contact=False,  # Disable contact in Stage 1
        contact_thresh=args.contact_thresh,
        collision_thresh=args.collision_thresh,
    )
    obj_scale = model.servers["object"].object_model.obj_scale.cpu().detach().numpy()

    for node_id in node_ids:
        if "right" in node_id or "left" in node_id:
            hand_shapes[node_id] = (
                model.nodes[node_id].params.betas.weight.cpu().detach().numpy()
            )

    # ====================================================================
    # STAGE 2: Optimize all frames
    # ====================================================================
    print("\n" + "=" * 70)
    print("Stage [2/3]: Optimizing all frames")
    if getattr(args, 'phase4_in_stage2', False):
        print("[Phase 4] Contact refinement enabled in Stage 2")
    print("=" * 70)
    final_preds = {}
    for frame_idx in tqdm(range(num_frames), desc="[Stage 2] Frame optimization"):
        final_pred = optimize_batch(
            [frame_idx],
            args,
            None,
            out,
            device,
            obj_scale=obj_scale,
            freeze_scale=True,
            freeze_shape=True,
            hand_shapes=hand_shapes,
            # Phase 4: Optional contact in Stage 2 (default: False)
            use_phase4_contact=getattr(args, 'phase4_in_stage2', False),
            contact_thresh=args.contact_thresh,
            collision_thresh=args.collision_thresh,
        )

        # only need to save the last iteration
        for key in final_pred.keys():
            if key.endswith("_seq"):
                val = final_pred[key]
                if isinstance(val, list):
                    val = val[-1]
                if key not in final_preds.keys():
                    final_preds[key] = []
                final_preds[key].append(val)

    print("\n[Stage 2] Optimization complete")

    # ====================================================================
    # PHASE 4: STAGE 3 - GHOP CONTACT REFINEMENT
    # ====================================================================
    if args.phase4_contact:
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 4: Stage [3/3] - GHOP Contact Refinement")
        logger.info("=" * 70)

        # Validate Phase 4 requirements
        if args.vqvae_checkpoint is None:
            logger.error("[Phase 4] --vqvae_checkpoint required for contact refinement")
            logger.error("[Phase 4] Skipping Stage 3...")
        else:
            try:
                # Initialize GHOP components
                logger.info("[Phase 4] Initializing GHOP mesh extractor...")

                # Load VQ-VAE wrapper
                from src.model.ghop.autoencoder import GHOPVQVAEWrapper
                device_torch = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                vqvae_wrapper = GHOPVQVAEWrapper(
                    vqvae_ckpt_path=args.vqvae_checkpoint,
                    device=device_torch,
                    use_hand_field=False  # Not needed for mesh extraction only
                )

                mesh_extractor = GHOPMeshExtractor(
                    vqvae_wrapper=vqvae_wrapper,
                    resolution=args.mesh_resolution
                )
                logger.info(f"✓ Mesh extractor initialized (resolution: {args.mesh_resolution}³)")

                # Initialize contact refiner
                logger.info("[Phase 4] Initializing contact refiner...")
                contact_refiner = GHOPContactRefinement(
                    contact_thresh=args.contact_thresh,
                    collision_thresh=args.collision_thresh,
                    contact_zones='zones'
                )
                logger.info("✓ Contact refiner initialized")

                # Extract meshes from optimized checkpoint
                logger.info("[Phase 4] Extracting meshes from optimized checkpoint...")

                # Reconstruct HOLD model from checkpoint
                from src.hold.hold import HOLD
                from src.utils.parser import parser_args

                # Create minimal args for model loading
                load_args, load_opt = parser_args()
                load_args.case = args.ckpt_p.split('/')[-3]  # Extract case name from path

                # Load HOLD model
                logger.info(f"[Phase 4] Loading HOLD model from {args.ckpt_p}")
                model_hold = HOLD.load_from_checkpoint(
                    args.ckpt_p,
                    opt=load_opt,
                    args=load_args,
                    strict=False
                )
                model_hold = model_hold.to(device_torch)
                model_hold.eval()

                logger.info("✓ HOLD model loaded from checkpoint")

                # Extract meshes for all frames
                logger.info(f"[Phase 4] Extracting meshes for {num_frames} frames...")

                hand_meshes = []
                object_meshes = []

                for frame_idx in tqdm(range(num_frames), desc="[Phase 4] Mesh extraction"):
                    # Extract meshes using helper functions
                    with torch.no_grad():
                        # Extract hand mesh from MANO parameters
                        hand_verts, hand_faces = extract_hand_mesh_from_checkpoint(
                            model_hold, frame_idx, device_torch
                        )

                        # Extract object mesh from implicit SDF
                        object_sdf = extract_object_sdf_from_checkpoint(
                            model_hold, frame_idx, device_torch, args.mesh_resolution
                        )

                        # Convert SDF to mesh via Marching Cubes
                        object_mesh = sdf_to_mesh(object_sdf, args.mesh_resolution)

                    hand_meshes.append((hand_verts, hand_faces))
                    object_meshes.append(object_mesh)

                logger.info(f"✓ Extracted {len(hand_meshes)} hand meshes and {len(object_meshes)} object meshes")

                # Optimize for contact across all frames
                logger.info(f"[Phase 4] Optimizing contact for {num_frames} frames...")
                logger.info(f"[Phase 4] Contact iterations: {args.contact_iters}")

                # Create optimizable parameters (hand vertices for each frame)
                hand_verts_all = [hv.clone().detach().requires_grad_(True) for hv, _ in hand_meshes]

                # Setup optimizer for all frames
                optimizer = torch.optim.Adam(hand_verts_all, lr=1e-4)

                # Contact refinement loop
                pbar = tqdm(range(args.contact_iters), desc="[Phase 4] Contact optimization")

                best_loss = float('inf')
                contact_history = []

                for iteration in pbar:
                    optimizer.zero_grad()

                    total_contact_loss = 0.0
                    total_penetration = 0.0
                    total_attraction = 0.0
                    valid_frames = 0

                    # Compute contact loss for each frame
                    for frame_idx in range(num_frames):
                        hand_verts = hand_verts_all[frame_idx]
                        _, hand_faces = hand_meshes[frame_idx]
                        obj_mesh = object_meshes[frame_idx]

                        # Skip if object mesh is empty
                        if obj_mesh.vertices.shape[0] == 0:
                            continue

                        obj_verts = torch.from_numpy(obj_mesh.vertices).float().to(device_torch)
                        obj_faces = torch.from_numpy(obj_mesh.faces).long().to(device_torch)

                        # Compute contact loss
                        contact_loss, contact_metrics = contact_refiner(
                            hand_verts=hand_verts.unsqueeze(0),  # [1, 778, 3]
                            hand_faces=hand_faces,  # [1538, 3]
                            obj_verts=obj_verts.unsqueeze(0),  # [1, V, 3]
                            obj_faces=obj_faces  # [F, 3]
                        )

                        total_contact_loss += contact_loss
                        total_penetration += contact_metrics.get('penetration', 0.0)
                        total_attraction += contact_metrics.get('attraction', 0.0)
                        valid_frames += 1

                    # Average over valid frames
                    if valid_frames > 0:
                        avg_contact_loss = total_contact_loss / valid_frames
                        avg_penetration = total_penetration / valid_frames
                        avg_attraction = total_attraction / valid_frames

                        # Backward pass
                        avg_contact_loss.backward()
                        optimizer.step()

                        # Track best loss
                        if avg_contact_loss.item() < best_loss:
                            best_loss = avg_contact_loss.item()

                        # Record history
                        contact_history.append({
                            'iteration': iteration,
                            'loss': avg_contact_loss.item(),
                            'penetration': avg_penetration,
                            'attraction': avg_attraction
                        })

                        # Update progress bar
                        pbar.set_postfix({
                            'loss': f'{avg_contact_loss.item():.4f}',
                            'pen': f'{avg_penetration:.4f}',
                            'attr': f'{avg_attraction:.4f}',
                            'best': f'{best_loss:.4f}'
                        })

                        # Log every 10 iterations
                        if iteration % 10 == 0:
                            logger.info(
                                f"[Phase 4] Iter {iteration}/{args.contact_iters}: "
                                f"Loss={avg_contact_loss.item():.4f}, "
                                f"Penetration={avg_penetration:.4f}, "
                                f"Attraction={avg_attraction:.4f}"
                            )

                logger.info(f"✓ Contact refinement complete (best loss: {best_loss:.4f})")

                # Write refined hand vertices back to checkpoint
                logger.info("[Phase 4] Updating checkpoint with refined meshes...")

                # Store refined vertices in final_preds
                final_preds['phase4_refined_hand_verts'] = [
                    hv.detach().cpu().numpy() for hv in hand_verts_all
                ]
                final_preds['phase4_contact_history'] = contact_history
                final_preds['phase4_best_loss'] = best_loss

                logger.info("✓ Checkpoint updated with contact-refined meshes")
                logger.info(f"[Phase 4] Final statistics:")
                logger.info(f"   - Total frames refined: {valid_frames}/{num_frames}")
                logger.info(f"   - Final avg loss: {contact_history[-1]['loss']:.4f}")
                logger.info(f"   - Best loss achieved: {best_loss:.4f}")
                logger.info(f"   - Final penetration: {contact_history[-1]['penetration']:.4f}")
                logger.info(f"   - Final attraction: {contact_history[-1]['attraction']:.4f}")
                logger.info("=" * 70 + "\n")

            except Exception as e:
                logger.error(f"[Phase 4] Contact refinement failed: {e}")
                import traceback
                traceback.print_exc()
                logger.error("[Phase 4] Continuing without contact refinement...")
    # ====================================================================

    # Save optimized checkpoint
    final_preds["obj_scale"] = obj_scale
    final_preds["hand_shapes"] = hand_shapes
    final_preds["node_ids"] = node_ids

    import pickle
    out_p = args.out_p
    print(f"\nSaving optimized checkpoint to: {out_p}")
    with open(out_p, "wb") as f:
        pickle.dump(final_preds, f)

    print("=" * 70)
    print("✓ Checkpoint optimization complete!")
    if args.phase4_contact:
        print(f"✓ Phase 4 contact refinement applied")
    print(f"✓ Results saved to: {out_p}")
    print("=" * 70)


# ========================================================================
# PHASE 4: Helper Functions for Mesh Extraction
# ========================================================================

def extract_hand_mesh_from_checkpoint(model, frame_idx, device):
    """Extract hand mesh from HOLD checkpoint for a specific frame.

    Args:
        model: HOLD model loaded from checkpoint
        frame_idx: Frame index to extract
        device: Torch device

    Returns:
        hand_verts: [778, 3] hand vertex positions
        hand_faces: [1538, 3] hand face indices
    """
    # Find hand node
    hand_node = None
    for node in model.model.nodes.values():
        if 'right' in node.node_id.lower() or 'left' in node.node_id.lower():
            hand_node = node
            break

    if hand_node is None:
        raise ValueError("No hand node found in checkpoint")

    # Get MANO parameters for this frame
    with torch.no_grad():
        frame_tensor = torch.tensor([[frame_idx]]).to(device)
        params = hand_node.params(frame_tensor)

        # Extract MANO output
        mano_server = hand_node.server

        # Get pose, shape, trans
        pose_key = f'{hand_node.node_id}.mano_pose'
        shape_key = f'{hand_node.node_id}.mano_shape'
        trans_key = f'{hand_node.node_id}.mano_trans'

        pose = params.get(pose_key, torch.zeros(1, 48, device=device))
        shape = params.get(shape_key, torch.zeros(1, 10, device=device))
        trans = params.get(trans_key, torch.zeros(1, 3, device=device))

        # Forward MANO
        hand_output = mano_server(
            pose=pose,
            shape=shape,
            trans=trans
        )

        if isinstance(hand_output, dict):
            hand_verts = hand_output['vertices'].squeeze(0)  # [778, 3]
        else:
            hand_verts = hand_output.squeeze(0)

        hand_faces = mano_server.faces  # [1538, 3]

    return hand_verts, hand_faces


def extract_object_sdf_from_checkpoint(model, frame_idx, device, resolution=128):
    """Extract object SDF from HOLD checkpoint.

    Args:
        model: HOLD model loaded from checkpoint
        frame_idx: Frame index
        device: Torch device
        resolution: SDF grid resolution

    Returns:
        sdf_grid: [resolution, resolution, resolution] SDF values
    """
    # Find object node
    object_node = None
    for node in model.model.nodes.values():
        if 'object' in node.node_id.lower():
            object_node = node
            break

    if object_node is None:
        raise ValueError("No object node found in checkpoint")

    # Create sampling grid in object canonical space
    x = torch.linspace(-1.5, 1.5, resolution)
    y = torch.linspace(-1.5, 1.5, resolution)
    z = torch.linspace(-1.5, 1.5, resolution)

    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    grid_coords = torch.stack([grid_x, grid_y, grid_z], dim=-1)
    grid_coords = grid_coords.reshape(-1, 3).to(device)

    # Query SDF at grid points
    with torch.no_grad():
        # Access implicit network
        sdf_values = object_node.implicit_network(
            grid_coords,
            compute_grad=False
        )

        # Extract SDF values
        if isinstance(sdf_values, dict):
            sdf = sdf_values['sdf'].squeeze(-1)
        else:
            sdf = sdf_values[:, 0]

    # Reshape to 3D grid
    sdf_grid = sdf.reshape(resolution, resolution, resolution)

    return sdf_grid.cpu().numpy()


def sdf_to_mesh(sdf_grid, resolution):
    """Convert SDF grid to triangle mesh using Marching Cubes.

    Args:
        sdf_grid: [H, W, D] numpy array with SDF values
        resolution: Grid resolution

    Returns:
        trimesh.Trimesh: Extracted mesh
    """
    try:
        # Apply Marching Cubes at zero-level set
        verts, faces, normals, values = measure.marching_cubes(
            sdf_grid,
            level=0.0,
            spacing=(3.0 / resolution, 3.0 / resolution, 3.0 / resolution)
        )

        # Shift to [-1.5, 1.5] coordinate system
        verts = verts - 1.5

        # Create trimesh object
        mesh = trimesh.Trimesh(
            vertices=verts,
            faces=faces,
            vertex_normals=normals
        )

        return mesh

    except Exception as e:
        logger.warning(f"[Phase 4] Marching Cubes failed: {e}")
        # Return empty mesh as fallback
        return trimesh.Trimesh(
            vertices=np.zeros((0, 3)),
            faces=np.zeros((0, 3), dtype=np.int32)
        )

# ========================================================================


def fetch_parser():
    """Parse command-line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Optimize HOLD checkpoint with optional Phase 4 contact refinement"
    )

    # Standard optimization arguments
    parser.add_argument("--inspect_idx", type=int, default=None,
                        help="Inspect specific frame index")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Number of frames to optimize simultaneously in Stage 1")
    parser.add_argument("--ckpt_p", type=str, required=True,
                        help="Path to checkpoint file (.ckpt)")
    parser.add_argument("--write_gif", action="store_true",
                        help="Write GIF visualization during optimization")
    parser.add_argument("--iters", type=int, default=500,
                        help="Number of optimization iterations per frame")
    parser.add_argument("--vis_every", type=int, default=5,
                        help="Visualization frequency during optimization")
    parser.add_argument("--itw", action="store_true",
                        help="In-the-wild mode flag")

    # ====================================================================
    # PHASE 4: Contact Refinement Arguments
    # ====================================================================
    parser.add_argument("--phase4_contact", action="store_true",
                        help="Enable Phase 4 GHOP contact refinement (Stage 3)")
    parser.add_argument("--contact_iters", type=int, default=100,
                        help="Number of contact refinement iterations")
    parser.add_argument("--mesh_resolution", type=int, default=128,
                        help="Mesh extraction resolution for contact refinement")
    parser.add_argument("--contact_thresh", type=float, default=0.01,
                        help="Contact distance threshold in meters (default: 1cm)")
    parser.add_argument("--collision_thresh", type=float, default=0.005,
                        help="Collision penetration threshold in meters (default: 5mm)")
    parser.add_argument("--vqvae_checkpoint", type=str, default=None,
                        help="Path to GHOP VQ-VAE checkpoint (required for Phase 4)")

    # Optional: Enable contact refinement during Stage 2 (slower but more accurate)
    parser.add_argument("--phase4_in_stage2", action="store_true",
                        help="Enable Phase 4 contact during Stage 2 frame optimization (experimental)")

    args = parser.parse_args()

    # Convert to EasyDict for convenience
    from easydict import EasyDict as edict
    args = edict(vars(args))

    # Set output path based on Phase 4 flag
    if args.phase4_contact:
        out_p = args.ckpt_p.replace(".ckpt", ".phase4_contact_ref")
        logger.info(f"\n[Phase 4] Contact refinement enabled")
        logger.info(f"[Phase 4] Output will be saved to: {out_p}\n")
    else:
        out_p = args.ckpt_p.replace(".ckpt", ".pose_ref")

    args.out_p = out_p

    return args


def main():
    """Main entry point."""
    args = fetch_parser()

    print("\n" + "=" * 70)
    print("HOLD CHECKPOINT OPTIMIZATION")
    print("=" * 70)
    print(f"Checkpoint: {args.ckpt_p}")
    print(f"Batch size: {args.batch_size}")
    print(f"Iterations per frame: {args.iters}")

    if args.phase4_contact:
        print(f"\n[Phase 4] Contact Refinement: ENABLED")
        print(f"  - Contact iterations: {args.contact_iters}")
        print(f"  - Mesh resolution: {args.mesh_resolution}³")
        print(f"  - Contact threshold: {args.contact_thresh}m")
        print(f"  - Collision threshold: {args.collision_thresh}m")
        print(f"  - VQ-VAE checkpoint: {args.vqvae_checkpoint}")
    else:
        print(f"\n[Phase 4] Contact Refinement: DISABLED")
        print(f"  (Use --phase4_contact to enable)")

    print("=" * 70 + "\n")

    # Run optimization
    fit_ckpt(args)


if __name__ == "__main__":
    main()
