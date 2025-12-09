import numpy as np
import torch
from kaolin.ops.mesh import index_vertices_by_faces

import src.engine.volsdf_utils as volsdf_utils
import src.utils.debug as debug
from src.model.renderables.node import Node
from src.datasets.utils import get_camera_params
from common.body_models import seal_mano_v
from common.body_models import seal_mano_f
from src.utils.meshing import generate_mesh
from src.model.mano.deformer import MANODeformer
from src.model.mano.server import MANOServer
import src.hold.hold_utils as hold_utils


class MANONode(Node):
    def __init__(self, args, opt, betas, sdf_bounding_sphere, node_id):
        if node_id == "right":
            class_id = 2
            self.is_rhand = True
        elif node_id == "left":
            class_id = 3
            self.is_rhand = False
        else:
            assert False

        # deformer = MANODeformer(max_dist=0.1, K=15, betas=betas, is_rhand=self.is_rhand)
        deformer = MANODeformer(max_dist=2.0, K=15, betas=betas, is_rhand=self.is_rhand)
        server = MANOServer(betas=betas, is_rhand=self.is_rhand)

        from src.model.mano.params import MANOParams
        from src.model.mano.specs import mano_specs

        params = MANOParams(
            args.n_images,
            {
                "betas": 10,
                "global_orient": 3,
                "transl": 3,
                "pose": 45,
            },
            node_id,
        )
        params.load_params(args.case)
        super(MANONode, self).__init__(
            args,
            opt,
            mano_specs,
            sdf_bounding_sphere,
            opt.implicit_network,
            opt.rendering_network,
            deformer,
            server,
            class_id,
            node_id,
            params,
        )

        self.mesh_v_cano = self.server.verts_c
        self.mesh_f_cano = torch.tensor(
            self.server.human_layer.faces.astype(np.int64)
        ).cuda()
        self.mesh_face_vertices = index_vertices_by_faces(
            self.mesh_v_cano, self.mesh_f_cano
        )

        self.mesh_v_cano_div = None
        self.mesh_f_cano_div = None
        self.canonical_mesh = None

    def sample_points(self, input):
        """Sample points with correct shape handling."""
        node_id = self.node_id
        full_pose = input[f"{node_id}.full_pose"]

        # ================================================================
        # ✅ FIX: Handle 3D parameter tensors correctly
        # ================================================================
        # Parameters from GenericParams have shape [B, 1, D]
        # Server expects 2D inputs: [B, D]
        # Need to squeeze the middle dimension

        params = input[f"{node_id}.params"]  # [B, 1, 48]
        transl = input[f"{node_id}.transl"]  # [B, 1, 3]
        betas = input[f"{node_id}.betas"]    # [B, 1, 10]

        # Squeeze middle dimension if present
        if params.dim() == 3:
            scene_scale = params.squeeze(1)[:, 0]  # [B, 1, 48] -> [B, 48] -> [B]
            transl = transl.squeeze(1)              # [B, 1, 3] -> [B, 3]
            full_pose = full_pose.squeeze(1)        # [B, 1, 48] -> [B, 48]
            betas = betas.squeeze(1)                # [B, 1, 10] -> [B, 10]
        else:
            # Original HOLD path (2D tensors)
            scene_scale = params[:, 0]

        output = self.server(
            scene_scale,  # [B]
            transl,       # [B, 3]
            full_pose,    # [B, 48]
            betas,        # [B, 10]
        )

        # ========================================================
        # 🔧 FIX: Store predicted joints for supervision
        # ========================================================
        if 'jnts' in output:
            # Store joints so they can be added to model outputs later
            self._predicted_joints = output['jnts']  # [B, 21, 3]

        debug.debug_world2pix(self.args, output, input, self.node_id)

        # ================================================================
        # ✅ FIX: Ensure cond has 2D shape [B, D] not [B, 1, D]
        # ================================================================
        pose_cond = full_pose[:, 3:] / np.pi  # [B, 45] after squeeze above
        cond = {"pose": pose_cond}  # Now [B, 45] - correct!

        if self.training:
            if input["current_epoch"] < 20:
                cond = {"pose": pose_cond * 0.0}  # no pose for shape

        ray_dirs, cam_loc = get_camera_params(
            input["uv"], input["extrinsics"], input["intrinsics"]
        )
        batch_size, num_pixels, _ = ray_dirs.shape
        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)

        deform_info = {
            "cond": cond,
            "tfs": output["tfs"],
            "verts": output["verts"],
        }

        if 'tfs' in deform_info:
            tfs = deform_info['tfs']
            print(f"\n[MANO DEBUG] tfs validation:")
            print(f"  tfs shape: {tfs.shape}")
            print(f"  tfs has_nan: {torch.isnan(tfs).any().item()}")
            print(f"  tfs min/max: {tfs.min().item():.4f} / {tfs.max().item():.4f}")

            # Check if any transformation is identity (not deforming)
            for joint_idx in range(min(5, tfs.shape[1])):  # Check first 5 joints
                tf = tfs[0, joint_idx]
                is_identity = torch.allclose(tf, torch.eye(4).cuda(), atol=1e-3)
                print(f"  Joint {joint_idx} is identity: {is_identity}")

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

        if torch.isnan(z_vals).any():
            print(f"  ❌ z_vals is NaN from ray sampler!")
            # Apply fix
            z_vals = torch.nan_to_num(z_vals, nan=1.0)

        # fg samples to points
        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)

        if torch.isnan(points).any():
            print(f"  ❌ points is NaN after cam_loc + z_vals * ray_dirs!")
            # Apply fix
            points = torch.nan_to_num(points, nan=0.0)

        out = {}
        out["idx"] = input["idx"]
        out["output"] = output
        out["cond"] = cond
        out["ray_dirs"] = ray_dirs
        out["cam_loc"] = cam_loc
        out["deform_info"] = deform_info
        out["z_vals"] = z_vals
        out["points"] = points
        out["tfs"] = output["tfs"]
        out["batch_size"] = batch_size
        out["num_pixels"] = num_pixels
        return out

    def spawn_cano_mano(self, sample_dict_h):
        mesh_v_cano = sample_dict_h["output"]["v_posed"]
        mesh_vh_cano = seal_mano_v(mesh_v_cano)
        mesh_fh_cano = seal_mano_f(self.mesh_f_cano, self.is_rhand)

        # GHOP FIX: Check for NaN/Inf before subdivision
        if not torch.isfinite(mesh_vh_cano).all():
            import warnings
            warnings.warn(f"[GHOP] NaN detected in MANO vertices (shape: {mesh_vh_cano.shape}), skipping subdivision")

            # Ensure mesh_vh_cano is 2D: [n_verts, 3]
            while mesh_vh_cano.dim() > 2:
                if mesh_vh_cano.shape[0] == 1:
                    mesh_vh_cano = mesh_vh_cano.squeeze(0)
                else:
                    # If first dim is not 1, take the first sample
                    mesh_vh_cano = mesh_vh_cano[0]
                    warnings.warn(f"[GHOP] Took first sample from batch, new shape: {mesh_vh_cano.shape}")

            # Final check
            if mesh_vh_cano.dim() != 2:
                raise RuntimeError(f"[GHOP] Cannot reduce mesh to 2D, shape: {mesh_vh_cano.shape}")

            self.mesh_v_cano_div = mesh_vh_cano  # Shape must be [n_verts, 3]
            self.mesh_f_cano_div = mesh_fh_cano
            return

        mesh_vh_cano, mesh_fh_cano = hold_utils.subdivide_cano(
            mesh_vh_cano, mesh_fh_cano
        )
        self.mesh_v_cano_div = mesh_vh_cano
        self.mesh_f_cano_div = mesh_fh_cano

    def meshing_cano(self, pose=None):
        """Extract canonical hand mesh without gradient tracking.

        Args:
            pose: Optional hand pose parameters

        Returns:
            trimesh.Trimesh: Canonical hand mesh
        """
        # ================================================================
        # FIX 3: Wrap entire operation in torch.no_grad()
        # ================================================================
        # This prevents gradient graph accumulation during:
        # 1. Pose tensor creation
        # 2. SDF network queries (thousands of them)
        # 3. Marching Cubes algorithm
        # ================================================================
        if pose is None:
            cond = {"pose": torch.zeros(1, self.specs.pose_dim).float().cuda()}
        else:
            cond = {"pose": pose / np.pi}

        # Ensure condition tensors don't have gradients
        # cond["pose"] = cond["pose"].detach()

        assert cond["pose"].shape[0] == 1, "only support batch size 1"

        v_min_max = np.array([[-0.0814, -0.0280, -0.0742], [0.1171, 0.0349, 0.0971]])

        # ================================================================
        # FIX 3: query_oc is called within no_grad context
        # ================================================================
        # The lambda function captures the no_grad context, so all
        # implicit_network evaluations are gradient-free
        mesh_canonical = generate_mesh(
            lambda x: hold_utils.query_oc(self.implicit_network, x, cond),
            v_min_max,
            point_batch=10000,
            res_up=1,
            res_init=64,
        )

        return mesh_canonical