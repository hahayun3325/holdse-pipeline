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
        params.load_params(args.case)
        super(ObjectNode, self).__init__(
            args,
            opt,
            object_specs,
            sdf_bounding_sphere,
            opt.implicit_network,
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

        # ✅ Call server (returns output dict)
        output = self.server(scene_scale, transl, global_orient)

        # ✅ Create cond dictionary (2D after squeezing)
        cond = {"pose": global_orient / np.pi}  # [B, 3]

        # Get camera parameters
        ray_dirs, cam_loc = get_camera_params(
            input["uv"], input["extrinsics"], input["intrinsics"]
        )
        batch_size, num_pixels, _ = ray_dirs.shape
        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)

        # ================================================================
        # ✅ FIX: Create deform_info with correct variable names
        # ================================================================
        deform_info = {
            "cond": cond,
            "verts": output.get("verts", None),  # Use 'output' not 'obj_output'
        }

        # Add tfs only if server provides it
        if "tfs" in output:
            deform_info["tfs"] = output["tfs"]

        # Handle test mode
        if self.is_test and "obj_verts" in output:
            deform_info["verts"] = output["obj_verts"]

        # Ray sampling
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

        # ✅ ADD THIS: Expand cond to match ray batch
        if batch_size > 1:
            # Repeat cond for each pixel in the batch
            # cond: [B, 3] -> [B, num_pixels, 3] -> [B*num_pixels, 3]
            cond_expanded = cond["pose"].unsqueeze(1).repeat(1, num_pixels, 1)
            cond_expanded = cond_expanded.reshape(-1, 3)
            cond = {"pose": cond_expanded}

        # ================================================================
        # ✅ FIX: Build output dict with correct variable names
        # ================================================================
        out = {}
        out["idx"] = input["idx"]
        out["output"] = output              # ✅ Changed from obj_output to output
        out["cond"] = cond                  # ✅ Changed from obj_cond to cond
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
        with torch.no_grad():
            cond = {"pose": torch.zeros(1, self.specs.pose_dim).float().cuda()}

            # Ensure condition tensor doesn't have gradients
            cond["pose"] = cond["pose"].detach()

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
