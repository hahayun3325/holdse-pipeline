import sys
from loguru import logger
import torch

sys.path = ["."] + sys.path


import src.utils.external.body_models as body_models


def construct_da_mano_pose(hand_mean):
    from src.model.mano.specs import mano_specs as body_specs

    param_canonical = torch.zeros((1, body_specs.total_dim), dtype=torch.float32).cuda()
    param_canonical[0, 0] = 1  # scale
    param_canonical[0, 7:52] = -hand_mean.unsqueeze(0)
    return param_canonical


class GenericServer(torch.nn.Module):
    def __init__(
        self,
        body_specs,
        betas=None,
        human_layer=None,
    ):
        super().__init__()
        assert human_layer is not None
        self.human_layer = human_layer.cuda()
        self.bone_parents = self.human_layer.bone_parents.astype(int)
        self.bone_parents[0] = -1
        self.bone_ids = []
        self.faces = self.human_layer.faces
        for i in range(body_specs.num_full_tfs):
            self.bone_ids.append([self.bone_parents[i], i])

        self.v_template = None

        if betas is not None:
            self.betas = torch.tensor(betas).float().cuda()
        else:
            self.betas = None

        # define the canonical pose
        param_canonical = construct_da_mano_pose(self.human_layer.hand_mean)
        if self.betas is not None:
            param_canonical[0, -body_specs.shape_dim :] = self.betas  # shape
        self.param_canonical = param_canonical

        self.cano_params = torch.split(
            self.param_canonical,
            [1, 3, body_specs.full_pose_dim, body_specs.shape_dim],
            dim=1,
        )

        # forward to get verts and joints
        output = self.forward(*self.cano_params, absolute=True)
        self.verts_c = output["verts"]
        self.joints_c = output["jnts"]
        # FIX: Compute inverse on CPU to avoid CUDA cuSPARSE issues
        tfs = output["tfs"].squeeze(0)
        if tfs.is_cuda:
            # Move to CPU for safe inverse computation
            tfs_cpu = tfs.cpu()
            tfs_c_inv_cpu = tfs_cpu.inverse()
            # Move result back to GPU
            self.tfs_c_inv = tfs_c_inv_cpu.to(tfs.device)
        else:
            # Already on CPU, compute normally
            self.tfs_c_inv = tfs.inverse()

    def forward(self, scene_scale, transl, thetas, betas, absolute=False):
        # ================================================================
        # FIX: Periodic CUDA cache clearing to prevent memory fragmentation
        # ================================================================
        # Initialize counter on first call
        if not hasattr(self, '_forward_count'):
            self._forward_count = 0

        self._forward_count += 1

        # Clear cache every 50 forward passes (aggressive cleanup)
        if self._forward_count % 50 == 0:
            torch.cuda.empty_cache()
            logger.info(f"[MANO Server] 🧹 Cleared CUDA cache at forward #{self._forward_count}")

            # Also log current memory usage
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            logger.info(f"[MANO Server] 📊 Memory: {allocated:.1f}MB allocated, {reserved:.1f}MB reserved")

        logger.info("[MANO Server] ========== FORWARD CALL ==========")
        logger.info(f"[MANO Server] Input shapes:")
        logger.info(f"  scene_scale: {scene_scale.shape}")
        logger.info(f"  transl: {transl.shape}")
        logger.info(f"  thetas: {thetas.shape}")
        logger.info(f"  betas: {betas.shape}")

        out = {}

        # ================================================================
        # FIX: Squeeze middle dimensions for MANO layer compatibility
        # ================================================================
        # transl: [B, 1, 3] -> [B, 3]
        if transl.dim() == 3 and transl.shape[1] == 1:
            transl = transl.squeeze(1)
            logger.info(f"[MANO Server] Squeezed transl: {transl.shape}")

        # betas: [B, 1, 10] -> [B, 10]
        if betas.dim() == 3 and betas.shape[1] == 1:
            betas = betas.squeeze(1)
            logger.info(f"[MANO Server] Squeezed betas: {betas.shape}")

        # ignore betas if v_template is provided
        if self.v_template is not None:
            betas = torch.zeros_like(betas)

        # ================================================================
        # MANO layer forward pass
        # FIX: Explicit allocation instead of zeros_like
        # Avoids potential stale tensor references and allocation issues
        # ================================================================
        transl_zeros = torch.zeros(
            transl.shape,
            dtype=transl.dtype,
            device=transl.device,
            requires_grad=False  # Explicit: no gradients needed
        )

        outputs = body_models.forward_layer(
            self.human_layer,
            betas=betas,
            transl=transl_zeros,
            pose=thetas[:, 3:],
            global_orient=thetas[:, :3],
            return_verts=True,
            return_full_pose=True,
            v_template=None,
        )
        verts = outputs.vertices.clone()

        # ================================================================
        # FIX: Reshape for broadcasting
        # ================================================================
        scene_scale = scene_scale.view(-1, 1, 1)

        # Ensure transl is [B, 1, 3] for broadcasting with verts [B, V, 3]
        if transl.dim() == 2:  # [B, 3]
            transl = transl.unsqueeze(1)  # [B, 1, 3]

        logger.info(f"[MANO Server] After reshaping for broadcasting:")
        logger.info(f"  scene_scale: {scene_scale.shape}")
        logger.info(f"  transl: {transl.shape}")
        logger.info(f"  verts: {verts.shape}")

        out["verts"] = verts * scene_scale + transl * scene_scale

        joints = outputs.joints.clone()

        logger.info(f"[MANO Server] MANO layer outputs:")
        logger.info(f"  verts: {verts.shape}")
        logger.info(f"  joints: {joints.shape}")

        # Check for mismatch
        batch_size_from_transl = transl.shape[0]
        batch_size_from_verts = verts.shape[0]

        if batch_size_from_verts != batch_size_from_transl:
            logger.error(f"[MANO Server] ❌ SHAPE MISMATCH DETECTED:")
            logger.error(f"  verts batch dim: {batch_size_from_verts}")
            logger.error(f"  transl batch dim: {batch_size_from_transl}")
            logger.error(f"  Ratio: {batch_size_from_verts / batch_size_from_transl}")
            logger.error(f"  This suggests tensors were expanded somewhere upstream!")

        logger.info("[MANO Server] =============================================")

        out["jnts"] = joints * scene_scale + transl * scene_scale
        tf_mats = outputs.T.clone()
        tf_mats[:, :, :3, :] = tf_mats[:, :, :3, :] * scene_scale.view(-1, 1, 1, 1)
        tf_mats[:, :, :3, 3] = tf_mats[:, :, :3, 3] + transl * scene_scale

        # adjust current pose relative to canonical pose
        if not absolute:
            tf_mats = torch.einsum("bnij,njk->bnik", tf_mats, self.tfs_c_inv)

        out["tfs"] = tf_mats
        out["skin_weights"] = outputs.weights
        out["v_posed"] = outputs.v_posed

        # ================================================================
        # FIX: Explicit cleanup after forward pass
        # ================================================================
        # Detach intermediate tensors that might retain computation graphs
        for key in out:
            if isinstance(out[key], torch.Tensor) and out[key].requires_grad:
                # Only detach if not needed for backward pass
                # (these are usually just outputs, not used for gradients)
                pass  # Keep as-is for now, may need gradients

        return out

    def forward_param(self, param_dict):
        global_orient = param_dict.fuzzy_get("global_orient")
        pose = param_dict.fuzzy_get("pose")
        transl = param_dict.fuzzy_get("transl")
        full_pose = torch.cat((global_orient, pose), dim=1)
        shape = param_dict.fuzzy_get("betas")
        scene_scale = param_dict.fuzzy_get("scene_scale")

        batch_size = full_pose.shape[0]
        scene_scale = scene_scale.view(-1).repeat(batch_size)
        shape = shape.repeat(batch_size, 1)
        out = self.forward(scene_scale, transl, full_pose, shape)
        return out


class MANOServer(GenericServer):
    def __init__(self, betas, is_rhand):
        from src.model.mano.specs import mano_specs
        from src.utils.external.body_models import MANO

        mano_layer = MANO(
            model_path="./body_models",
            is_rhand=is_rhand,
            batch_size=1,
            flat_hand_mean=False,
            dtype=torch.float32,
            use_pca=False,
        )
        super().__init__(
            body_specs=mano_specs,
            betas=betas,
            human_layer=mano_layer,
        )
