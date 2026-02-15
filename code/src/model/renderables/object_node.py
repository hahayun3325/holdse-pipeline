import torch
import numpy as np
import src.engine.volsdf_utils as volsdf_utils
import src.utils.debug as debug
from src.model.renderables.node import Node
from src.datasets.utils import get_camera_params
from src.utils.meshing import generate_mesh
from kaolin.ops.mesh import index_vertices_by_faces
import torch.nn as nn
from src.model.obj.deformer import ObjectDeformer
from src.model.obj.server import ObjectServer
from src.model.obj.specs import object_specs
from src.model.obj.params import ObjectParams
import src.hold.hold_utils as hold_utils
from loguru import logger

class ObjectNode(Node):
    def __init__(self, args, opt, sdf_bounding_sphere, node_id):
        time_code_dim = 32
        opt.rendering_network.d_in = opt.rendering_network.d_in + time_code_dim
        deformer = ObjectDeformer()
        server = ObjectServer(args.case, None)
        class_id = 1
        params = ObjectParams(
            args.n_images,
            {
                "global_orient": 3,
                "transl": 3,
            },
            node_id,
        )

        # ========== CONDITIONAL LOADING ==========
        # Only load from dataset if not loading from checkpoint
        if getattr(args, 'loading_from_checkpoint', False):
            logger.info(f"[ObjectNode:{node_id}] Skipping load_params - will preserve checkpoint values")
            params.preserve_checkpoint_values()
        else:
            params.load_params(args.case)
        # ==========================================
        super(ObjectNode, self).__init__(
            args,
            opt,
            object_specs,
            sdf_bounding_sphere,
            getattr(opt, 'obj_implicit_network', opt.implicit_network),
            opt.rendering_network,
            deformer,
            server,
            class_id,
            node_id,
            params,
        )
        self.frame_latent_encoder = nn.Embedding(args.n_images, time_code_dim)
        self.is_test = False
        self.mesh_o = None
        v3d_cano = server.object_model.v3d_cano.cpu().detach().numpy()
        self.v_min_max = np.array([v3d_cano.min(axis=0), v3d_cano.max(axis=0)]) * 2.0

    # In src/model/renderables/object_node.py, after __init__
    # Add property to access geometry latent from object_model
    @property
    def z_geo_refined(self):
        """Expose geometry latent from object_model for implicit network conditioning."""
        if hasattr(self.server, 'object_model'):
            return self.server.object_model.get_z_geo_refined()
        return None

    # Optional: Add method to force recomputation
    def update_geometry_latent(self, voxel_grid=None):
        """Recompute z_geo_refined after v3d_cano updates."""
        if hasattr(self.server, 'object_model'):
            return self.server.object_model.update_geometry_latent(voxel_grid)
        return False

    def forward(self, input):
        time_code = self.frame_latent_encoder(input["idx"])
        input["time_code"] = time_code
        return super().forward(input)

    def sample_points(self, input):
        """Sample points with correct shape handling."""
        node_id = self.node_id

        # ================================================================
        # ✅ Handle 3D parameter tensors - squeeze to 2D
        # ================================================================
        params = input[f"{node_id}.params"]          # [B, 1, 1]
        transl = input[f"{node_id}.transl"]          # [B, 1, 3]
        global_orient = input[f"{node_id}.global_orient"]  # [B, 1, 3]

        # Squeeze middle dimension if 3D
        if params.dim() == 3:
            scene_scale = params.squeeze(1).squeeze(1)  # [B, 1, 1] -> [B]
            transl = transl.squeeze(1)                   # [B, 1, 3] -> [B, 3]
            global_orient = global_orient.squeeze(1)     # [B, 1, 3] -> [B, 3]
        else:
            scene_scale = params[:, 0]

        # After line ~30 (after squeeze operations)
        logger.info(f"\n[ObjectNode.sample_points] Object parameters:")
        logger.info(f"  scene_scale: {scene_scale}")
        logger.info(f"  transl: {transl}")
        logger.info(f"  global_orient: {global_orient}")
        # logger.info(f"  global_orient (degrees): {(global_orient * 180 / np.pi).cpu().numpy()}")

        # ✅ Call server (returns output dict)
        output = self.server(scene_scale, transl, global_orient)

        # ✅ Create cond dictionary (2D after squeezing)
        cond = {"pose": global_orient / np.pi}  # [B, 3]
        if hasattr(self, 'z_geo_refined') and self.z_geo_refined is not None:
            cond['geo'] = self.z_geo_refined  # Add FiLM conditioning here

        # Get camera parameters
        ray_dirs, cam_loc = get_camera_params(
            input["uv"], input["extrinsics"], input["intrinsics"]
        )
        batch_size, num_pixels, _ = ray_dirs.shape
        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)

        pose = cond["pose"]

        # Ensure 2D
        if pose.dim() == 1:
            pose = pose.unsqueeze(0)
        elif pose.dim() > 2:
            # Squeeze all middle singleton dimensions
            original_batch = pose.shape[0]
            pose = pose.reshape(original_batch, -1)

        num_pixels = ray_dirs.shape[0]  # ensure this is defined earlier

        # Expand pose: [B, D] -> [B*num_pixels, D]
        cond_expanded = pose.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, pose.shape[-1])

        # Build **final** cond, including FiLM geometry
        cond = {"pose": cond_expanded}
        if hasattr(self, "z_geo_refined") and self.z_geo_refined is not None:
            cond["geo"] = self.z_geo_refined

        # ================================================================
        # Build deform_info using the FINAL cond
        # ================================================================
        deform_info = {
            "cond": cond,
            "verts": output.get("verts", None),
        }
        if "tfs" in output:
            deform_info["tfs"] = output["tfs"]
        if self.is_test and "obj_verts" in output:
            deform_info["verts"] = output["obj_verts"]

        # Now call ray_sampler with consistent cond
        z_vals = self.ray_sampler.get_z_vals(
            volsdf_utils.sdf_func_with_deformer,
            self.deformer,
            self.implicit_network,
            ray_dirs,
            cam_loc,
            self.density,
            self.training,
            deform_info,
        )

        # Compute sample points
        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)

        out = {}
        out["idx"] = input["idx"]
        out["output"] = output
        out["cond"] = cond           # uses the same cond as deform_info["cond"]
        out["ray_dirs"] = ray_dirs
        out["cam_loc"] = cam_loc
        out["deform_info"] = deform_info
        out["z_vals"] = z_vals
        out["points"] = points

        # ✅ FIX: Handle tfs - use output dict, handle missing key
        if "tfs" in output:
            out["tfs"] = output["tfs"]
        elif "obj_tfs" in output:
            out["tfs"] = output["obj_tfs"]
        else:
            # No tfs available, set to None or skip
            out["tfs"] = None

        out["batch_size"] = batch_size
        out["num_pixels"] = num_pixels

        return out

    def meshing_cano(self):
        """Extract canonical object mesh without gradient tracking.

        Returns:
            trimesh.Trimesh: Canonical object mesh
        """
        # ================================================================
        # FIX 3: Wrap entire operation in torch.no_grad()
        # ================================================================
        # This prevents gradient graph accumulation during:
        # 1. Condition tensor creation
        # 2. SDF network queries (thousands of them)
        # 3. Marching Cubes algorithm
        # 4. Canonical mesh update
        # ================================================================
        cond = {"pose": torch.zeros(1, self.specs.pose_dim).float().cuda()}
        if hasattr(self, 'z_geo_refined') and self.z_geo_refined is not None:
            cond['geo'] = self.z_geo_refined

        # Ensure condition tensor doesn't have gradients
        # cond["pose"] = cond["pose"].detach()

        # ================================================================
        # FIX 3: query_oc is called within no_grad context
        # ================================================================
        # The lambda function captures the no_grad context, so all
        # implicit_network evaluations are gradient-free
        mesh_canonical = generate_mesh(
            lambda x: hold_utils.query_oc(self.implicit_network, x, cond),
            self.v_min_max,
            point_batch=10000,
            res_up=2,
        )

        # Update canonical mesh (also gradient-free)
        self.update_cano(mesh_canonical)

        return mesh_canonical

    def update_cano(self, mesh_canonical):
        self.mesh_vo_cano = torch.tensor(
            mesh_canonical.vertices[None],
            device="cuda",
        ).float()
        self.mesh_fo_cano = torch.tensor(
            mesh_canonical.faces.astype(np.int64),
            device="cuda",
        )
        self.mesh_o = index_vertices_by_faces(self.mesh_vo_cano, self.mesh_fo_cano)
