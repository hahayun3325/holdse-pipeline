import torch
from loguru import logger
from src.model.generic.params import GenericParams


class MANOParams(GenericParams):
    def forward(self, frame_ids):
        """
        Forward pass to retrieve MANO parameters.

        Args:
            frame_ids: [B] tensor of frame indices

        Returns:
            dict with keys:
                - {node_id}.betas: [B, 10]
                - {node_id}.global_orient: [B, 3]
                - {node_id}.transl: [B, 3]
                - {node_id}.pose: [B, 45]
                - {node_id}.full_pose: [B, 48] (global_orient + pose)
        """
        logger.info("="*70)
        logger.info("[MANOParams.forward] ENTRY")
        logger.info("="*70)
        logger.info(f"[MANOParams.forward] node_id: {self.node_id}")
        logger.info(f"[MANOParams.forward] Input frame_ids shape: {frame_ids.shape}")

        # Check internal parameter storage
        logger.info(f"[MANOParams.forward] Internal parameter storage:")
        for param_name in self.param_names:
            if hasattr(self, param_name):
                embedding_layer = getattr(self, param_name)
                logger.info(f"  {param_name}: {embedding_layer.weight.shape}")

        # Call parent forward
        logger.info(f"[MANOParams.forward] Calling super().forward(frame_ids)...")
        params = super().forward(frame_ids)

        logger.info(f"[MANOParams.forward] super().forward() returned {len(params)} keys:")
        node_id = self.node_id
        for k, v in params.items():
            if isinstance(v, torch.Tensor):
                logger.info(f"  {k}: {v.shape}")

        # ================================================================
        # FIXED: Concatenate global_orient and pose
        # ================================================================
        logger.info(f"[MANOParams.forward] Concatenating global_orient + pose...")

        global_orient = params[f"{node_id}.global_orient"]
        pose = params[f"{node_id}.pose"]

        logger.info(f"  {node_id}.global_orient BEFORE squeeze: {global_orient.shape}")
        logger.info(f"  {node_id}.pose BEFORE squeeze: {pose.shape}")

        # Squeeze middle dimension if present: [B, 1, D] -> [B, D]
        if global_orient.dim() == 3 and global_orient.shape[1] == 1:
            global_orient = global_orient.squeeze(1)
            logger.info(f"  Squeezed global_orient: {global_orient.shape}")

        if pose.dim() == 3 and pose.shape[1] == 1:
            pose = pose.squeeze(1)
            logger.info(f"  Squeezed pose: {pose.shape}")

        # Concatenate on last dimension
        full_pose = torch.cat((global_orient, pose), dim=-1)
        params[f"{node_id}.full_pose"] = full_pose

        logger.info(f"  Created {node_id}.full_pose: {full_pose.shape}")

        logger.info(f"[MANOParams.forward] Final params dictionary:")
        for k, v in params.items():
            if isinstance(v, torch.Tensor):
                logger.info(f"  {k}: {v.shape}")

        logger.info("="*70)
        return params

    def load_params(self, case):
        import os

        import numpy as np

        # load parameter from preprocessing
        params_h = {param_name: [] for param_name in self.param_names}
        data_root = os.path.join("./data", case, f"build/data.npy")
        data = np.load(data_root, allow_pickle=True).item()["entities"][self.node_id]

        mean_shape = data["mean_shape"]
        params_h["betas"] = torch.tensor(
            mean_shape[None],
            dtype=torch.float32,
        )

        poses = data["hand_poses"]
        trans = data["hand_trans"]

        params_h["global_orient"] = torch.tensor(
            poses[:, :3],
            dtype=torch.float32,
        )
        params_h["pose"] = torch.tensor(
            poses[:, 3:],
            dtype=torch.float32,
        )
        params_h["transl"] = torch.tensor(
            trans,
            dtype=torch.float32,
        )
        for param_name in params_h.keys():
            self.init_parameters(param_name, params_h[param_name], requires_grad=False)
