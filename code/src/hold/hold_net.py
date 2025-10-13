import sys

import torch
import torch.nn as nn
from loguru import logger

from src.model.renderables.background import Background
from src.model.renderables.object_node import ObjectNode
from src.model.renderables.mano_node import MANONode

sys.path = [".."] + sys.path
from common.xdict import xdict

import src.hold.hold_utils as hold_utils
from src.hold.hold_utils import prepare_loss_targets_hand
from src.hold.hold_utils import prepare_loss_targets_object
from src.hold.hold_utils import volumetric_render

# ========================================================================
# Phase 4: GHOP Mesh Extraction Imports
# ========================================================================
try:
    from skimage import measure  # For Marching Cubes
    SKIMAGE_AVAILABLE = True
except ImportError:
    logger.warning("[Phase 4] scikit-image not available - mesh extraction disabled")
    SKIMAGE_AVAILABLE = False

import numpy as np
# ========================================================================

class HOLDNet(nn.Module):
    def __init__(
        self,
        opt,
        betas_r,
        betas_l,
        num_frames,
        args,
    ):
        super().__init__()
        self.args = args
        self.opt = opt
        self.sdf_bounding_sphere = opt.scene_bounding_sphere
        self.threshold = 0.05

        # ====================================================================
        # Initialize Nodes (Hand + Object)
        # ====================================================================
        node_dict = {}

        if betas_r is not None:
            right_node = MANONode(args, opt, betas_r, self.sdf_bounding_sphere, "right")
            node_dict["right"] = right_node

        if betas_l is not None:
            left_node = MANONode(args, opt, betas_l, self.sdf_bounding_sphere, "left")
            node_dict["left"] = left_node

        object_node = ObjectNode(args, opt, self.sdf_bounding_sphere, "object")
        node_dict["object"] = object_node
        self.nodes = nn.ModuleDict(node_dict)
        self.background = Background(opt, args, num_frames, self.sdf_bounding_sphere)
        self.init_network()

        # ====================================================================
        # Phase 4: Flag (actual initialization happens in HOLD class)
        # ====================================================================
        # Note: Phase 4 components (mesh_extractor, contact_refiner) are
        # initialized in HOLD.__init__(), not here. HOLDNet just needs the flag.
        self.phase4_enabled = False  # Will be set by HOLD if Phase 4 is enabled

        # Cache for mesh extraction to avoid redundant computation
        self._mesh_cache = {}
        self._cache_step = -1

        logger.debug("[HOLDNet] Initialization complete")
        # ====================================================================

    def forward_fg(self, input):
        input = xdict(input)
        out_dict = xdict()
        if self.training:
            out_dict["epoch"] = input["current_epoch"]
            out_dict["step"] = input["global_step"]

        torch.set_grad_enabled(True)
        sample_dict = None
        factors_dicts = {}
        sample_dicts = {}

        # Forward pass through all nodes
        for node in self.nodes.values():
            factors, sample_dict = node(input)
            factors_dicts[node.node_id] = factors
            sample_dicts[node.node_id] = sample_dict

        import src.utils.debug as debug
        debug.debug_deformer(sample_dicts, self)

        # Compute canonical SDF and features
        out_dict = self.prepare_loss_targets(out_dict, sample_dicts)

        factors_list = list(factors_dicts.values())
        factors = hold_utils.merge_factors(factors_list, check=True)

        for myfactors in factors_dicts.values():
            myfactors["z_max"] = myfactors["z_vals"][:, -1]

        # Volumetric rendering: all
        comp_out_dict = volumetric_render(factors, self.training)
        out_dict.merge(comp_out_dict)

        for node_id, myfactors in factors_dicts.items():
            my_out_dict = volumetric_render(myfactors, self.training)
            out_dict.merge(my_out_dict.prefix(f"{node_id}."))

        # Background
        bg_z_vals = self.background.inverse_sphere_sampler.inverse_sample(
            sample_dict["ray_dirs"],
            sample_dict["cam_loc"],
            self.training,
            self.sdf_bounding_sphere,
        )

        out_dict["bg_z_vals"] = bg_z_vals
        out_dict["ray_dirs"] = sample_dict["ray_dirs"]
        out_dict["cam_loc"] = sample_dict["cam_loc"]
        out_dict["index"] = input["idx"]

        return out_dict

    def step_embedding(self):
        # Step on BARF counter
        for node in self.nodes.values():
            node.step_embedding()
        self.background.step_embedding()

    def forward(self, input):
        fg_dict = self.forward_fg(input)
        bg_dict = self.background(
            fg_dict["bg_weights"],
            fg_dict["ray_dirs"],
            fg_dict["cam_loc"],
            fg_dict["bg_z_vals"],
            fg_dict["index"],
        )

        out_dict = self.composite(fg_dict, bg_dict)

        if self.training:
            self.step_embedding()

        return out_dict

    def composite(self, fg_dict, bg_dict):
        out_dict = fg_dict

        # Composite foreground and background
        out_dict["rgb"] = fg_dict["fg_rgb"] + bg_dict["bg_rgb"]
        out_dict["semantics"] = fg_dict["fg_semantics"] + bg_dict["bg_semantics"]

        if not self.training:
            out_dict["bg_rgb_only"] = bg_dict["bg_rgb_only"]

        out_dict["instance_map"] = torch.argmax(out_dict["semantics"], dim=1)

        return out_dict

    def init_network(self):
        if self.args.shape_init != "":
            model_state = torch.load(
                f"./saved_models/{self.args.shape_init}/checkpoints/last.ckpt"
            )

            sd = model_state["state_dict"]
            sd = {
                k.replace("model.", ""): v
                for k, v in sd.items()
                if "implicit_network" in k
                and "bg_implicit_network." not in k
                and ".embedder_obj." not in k
            }

            logger.warning("Using MANO init that is for h2o, not the one in CVPR.")
            self.load_state_dict(sd, strict=False)
        else:
            logger.warning("Skipping INIT human models!")

    def prepare_loss_targets(self, out_dict, sample_dicts):
        if not self.training:
            return out_dict

        step = out_dict["step"]
        assert [node.node_id for node in self.nodes.values()] == [
            key for key in sample_dicts.keys()
        ]

        if step % 200 == 0 and step > 0:
            for node, node_id in zip(self.nodes.values(), sample_dicts):
                if node.node_id in ["right", "left"]:
                    node.spawn_cano_mano(sample_dicts[node_id])

        for node in self.nodes.values():
            node_id = node.node_id
            sample_dict = sample_dicts[node_id]

            if "right" in node_id or "left" in node_id:
                prepare_loss_targets_hand(out_dict, sample_dict, node)
            elif "object" in node_id:
                prepare_loss_targets_object(out_dict, sample_dict, node)
            else:
                raise ValueError(f"Unknown node_id: {node_id}")

        return out_dict

    # ====================================================================
    # Phase 4: Mesh Extraction Interface
    # ====================================================================

    def extract_meshes(self, batch, current_step=None):
        """Extract explicit meshes for Phase 4 contact refinement.

        This method is called from HOLD.training_step() when Phase 4 is enabled.

        Args:
            batch: Input batch with MANO parameters and frame data
            current_step: Current training step for caching

        Returns:
            dict containing:
                - hand_verts: [B, 778, 3] hand vertices
                - hand_faces: [1538, 3] hand triangle faces (shared)
                - obj_verts: List of [V, 3] object vertices per batch
                - obj_faces: List of [F, 3] object faces per batch
        """
        if not self.phase4_enabled:
            raise RuntimeError(
                "[Phase 4] Mesh extraction not enabled. "
                "Set phase4.enabled=true in config and ensure Phase 3 is initialized."
            )

        if not SKIMAGE_AVAILABLE:
            raise RuntimeError(
                "[Phase 4] scikit-image not available. "
                "Install with: pip install scikit-image"
            )

        # Check cache to avoid redundant extraction
        if current_step is not None and current_step == self._cache_step:
            logger.debug(f"[Phase 4] Using cached meshes for step {current_step}")
            return self._mesh_cache

        logger.debug(f"[Phase 4] Extracting meshes for step {current_step}")

        try:
            # Extract hand mesh from MANO
            hand_verts, hand_faces = self.extract_hand_geometry(batch)

            # Extract object mesh via implicit SDF
            obj_verts_list, obj_faces_list = self.extract_object_mesh(batch)

            mesh_dict = {
                'hand_verts': hand_verts,
                'hand_faces': hand_faces,
                'obj_verts': obj_verts_list,
                'obj_faces': obj_faces_list
            }

            # Update cache
            if current_step is not None:
                self._mesh_cache = mesh_dict
                self._cache_step = current_step

            logger.debug(
                f"[Phase 4] Mesh extraction complete: "
                f"hand {hand_verts.shape}, "
                f"obj {len(obj_verts_list)} batches"
            )

            return mesh_dict

        except Exception as e:
            logger.error(f"[Phase 4] Mesh extraction failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def extract_hand_geometry(self, batch):
        """Extract hand mesh from MANO node.

        Args:
            batch: Input batch with MANO parameters

        Returns:
            hand_verts: [B, 778, 3] vertex positions in world space
            hand_faces: [1538, 3] triangle face indices (shared across batch)
        """
        # Determine handedness (right or left)
        if 'right' in self.nodes:
            hand_node = self.nodes['right']
            node_id = 'right'
        elif 'left' in self.nodes:
            hand_node = self.nodes['left']
            node_id = 'left'
        else:
            raise ValueError("[Phase 4] No hand node found in HOLDNet")

        # Extract MANO parameters from batch
        # HOLD batch structure: {node_id}.{param_name}
        mano_pose_key = f"{node_id}.mano_pose"
        mano_rot_key = f"{node_id}.mano_rot"
        mano_shape_key = f"{node_id}.mano_shape"
        mano_trans_key = f"{node_id}.mano_trans"

        mano_pose = batch.get(mano_pose_key, None)
        mano_rot = batch.get(mano_rot_key, None)
        mano_shape = batch.get(mano_shape_key, None)
        mano_trans = batch.get(mano_trans_key, None)

        if mano_pose is None:
            raise ValueError(
                f"[Phase 4] Missing MANO pose in batch. "
                f"Expected key: {mano_pose_key}, available keys: {list(batch.keys())}"
            )

        # Get MANO vertices from the node
        with torch.no_grad():
            mano_server = hand_node.server

            # Construct full pose [B, 48] = [global_orient(3) + hand_pose(45)]
            if mano_rot is not None and mano_pose.shape[-1] == 45:
                full_pose = torch.cat([mano_rot, mano_pose], dim=-1)
            elif mano_pose.shape[-1] == 48:
                full_pose = mano_pose
            else:
                # Add zero rotation
                device = mano_pose.device
                zero_rot = torch.zeros(mano_pose.shape[0], 3, device=device)
                full_pose = torch.cat([zero_rot, mano_pose], dim=-1)

            # Prepare shape parameters
            if mano_shape is None:
                mano_shape = torch.zeros(full_pose.shape[0], 10, device=full_pose.device)

            # Prepare translation
            if mano_trans is None:
                mano_trans = torch.zeros(full_pose.shape[0], 3, device=full_pose.device)

            # Forward MANO to get mesh
            mano_output = mano_server(
                pose=full_pose,
                shape=mano_shape,
                trans=mano_trans
            )

            # Extract vertices and faces
            if isinstance(mano_output, dict):
                hand_verts = mano_output['vertices']  # [B, 778, 3]
            else:
                # Fallback if mano_output is tensor
                hand_verts = mano_output

            # Get faces (shared across batch)
            hand_faces = mano_server.faces  # [1538, 3]

        logger.debug(
            f"[Phase 4] Extracted hand mesh: {hand_verts.shape[0]} batches, "
            f"{hand_verts.shape[1]} vertices, {hand_faces.shape[0]} faces"
        )

        return hand_verts, hand_faces

    def extract_object_mesh(self, batch, resolution=128):
        """Extract object mesh from implicit SDF via Marching Cubes.

        Args:
            batch: Input batch
            resolution: Grid resolution for SDF sampling

        Returns:
            obj_verts_list: List of [V, 3] object vertices per batch
            obj_faces_list: List of [F, 3] object triangle faces per batch
        """
        # Determine batch size
        batch_size = self._get_batch_size(batch)

        # Get object node
        if 'object' not in self.nodes:
            raise ValueError("[Phase 4] No object node found in HOLDNet")

        object_node = self.nodes['object']

        # Query object SDF on a dense grid
        with torch.no_grad():
            # Create sampling grid in canonical space [-1, 1]³
            grid_coords = self._create_sampling_grid(resolution)
            grid_coords = grid_coords.to(self.args.device)

            # Query SDF at grid points
            sdf_values = self._query_object_sdf(object_node, grid_coords, batch_size)

            # Reshape to 3D grid [B, H, W, D]
            sdf_grid = sdf_values.reshape(batch_size, resolution, resolution, resolution)

        # Apply Marching Cubes to extract mesh
        obj_verts_list = []
        obj_faces_list = []

        for b in range(batch_size):
            sdf_numpy = sdf_grid[b].cpu().numpy()

            # Run Marching Cubes
            try:
                verts, faces, _, _ = measure.marching_cubes(
                    sdf_numpy,
                    level=0.0,  # Zero-level set
                    spacing=(2.0 / resolution, 2.0 / resolution, 2.0 / resolution)
                )

                # Shift to [-1, 1] coordinate system
                verts = verts - 1.0

                # Convert to tensors and move to device
                obj_verts_list.append(
                    torch.from_numpy(verts).float().to(self.args.device)
                )
                obj_faces_list.append(
                    torch.from_numpy(faces).long().to(self.args.device)
                )

                logger.debug(
                    f"[Phase 4] Batch {b}: extracted {verts.shape[0]} vertices, "
                    f"{faces.shape[0]} faces"
                )

            except Exception as e:
                logger.warning(f"[Phase 4] Marching Cubes failed for batch {b}: {e}")
                # Fallback to empty mesh
                obj_verts_list.append(torch.zeros((0, 3), device=self.args.device))
                obj_faces_list.append(
                    torch.zeros((0, 3), dtype=torch.long, device=self.args.device)
                )

        avg_verts = np.mean([v.shape[0] for v in obj_verts_list])
        logger.debug(
            f"[Phase 4] Extracted object mesh: {len(obj_verts_list)} batches, "
            f"avg {avg_verts:.0f} vertices"
        )

        return obj_verts_list, obj_faces_list

    def _get_batch_size(self, batch):
        """Infer batch size from batch dictionary."""
        # Try common keys
        for key in ['rgb', 'gt.rgb', 'idx']:
            if key in batch:
                return batch[key].shape[0]

        # Fallback: check node parameters
        for node_id in ['right', 'left', 'object']:
            mano_pose_key = f"{node_id}.mano_pose"
            if mano_pose_key in batch:
                return batch[mano_pose_key].shape[0]

        # Default to 1
        logger.warning("[Phase 4] Could not infer batch size, defaulting to 1")
        return 1

    def _create_sampling_grid(self, resolution):
        """Create 3D sampling grid in [-1, 1]³.

        Args:
            resolution: Grid resolution (H = W = D)

        Returns:
            grid_coords: [N, 3] where N = resolution³
        """
        x = torch.linspace(-1, 1, resolution)
        y = torch.linspace(-1, 1, resolution)
        z = torch.linspace(-1, 1, resolution)

        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
        grid_coords = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        grid_coords = grid_coords.reshape(-1, 3)  # [N, 3]

        return grid_coords

    def _query_object_sdf(self, object_node, grid_coords, batch_size):
        """Query object SDF at grid coordinates.

        Args:
            object_node: ObjectNode instance
            grid_coords: [N, 3] sampling coordinates
            batch_size: Number of batch elements

        Returns:
            sdf_values: [B, N] SDF values at grid points
        """
        num_points = grid_coords.shape[0]
        sdf_list = []

        for b in range(batch_size):
            # Query implicit network
            # Note: ObjectNode interface may vary; adjust as needed
            try:
                # Method 1: Direct implicit_network call
                sdf_output = object_node.implicit_network(
                    grid_coords,
                    compute_grad=False
                )

                if isinstance(sdf_output, dict):
                    sdf = sdf_output['sdf']  # [N, 1]
                else:
                    sdf = sdf_output[:, 0:1]  # [N, 1]

            except Exception as e:
                # Method 2: Fallback to server
                try:
                    sdf_output = object_node.server.forward_sdf(grid_coords)
                    if isinstance(sdf_output, dict):
                        sdf = sdf_output['sdf']
                    else:
                        sdf = sdf_output
                except:
                    logger.error(f"[Phase 4] SDF query failed for batch {b}: {e}")
                    # Return zero SDF as fallback
                    sdf = torch.zeros(num_points, 1, device=grid_coords.device)

            sdf_list.append(sdf.squeeze(-1))  # [N]

        sdf_values = torch.stack(sdf_list, dim=0)  # [B, N]
        return sdf_values

    # ====================================================================
    # End Phase 4 Mesh Extraction Interface
    # ====================================================================