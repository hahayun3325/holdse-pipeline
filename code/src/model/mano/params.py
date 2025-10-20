import torch
from loguru import logger
from src.model.generic.params import GenericParams


class MANOParams(GenericParams):
    def forward(self, frame_ids):
        """Forward with diagnostics and correct concatenation."""
        logger.info("="*70)
        logger.info("[MANOParams.forward] ENTRY")
        logger.info("="*70)
        logger.info(f"[MANOParams.forward] node_id: {self.node_id}")
        logger.info(f"[MANOParams.forward] Input frame_ids shape: {frame_ids.shape}")
        logger.info(f"[MANOParams.forward] Input frame_ids dtype: {frame_ids.dtype}")
        logger.info(f"[MANOParams.forward] Input frame_ids values: {frame_ids.flatten()[:10]}")

        # Check internal parameter storage
        logger.info(f"[MANOParams.forward] Internal parameter storage:")
        for param_name in self.param_names:
            if hasattr(self, param_name):
                embedding_layer = getattr(self, param_name)
                # nn.Embedding has .weight attribute, not .shape
                logger.info(f"  {param_name}: {embedding_layer.weight.shape}")

        # Call parent forward
        logger.info(f"[MANOParams.forward] Calling super().forward(frame_ids)...")
        params = super().forward(frame_ids)

        logger.info(f"[MANOParams.forward] super().forward() returned {len(params)} keys:")
        node_id = self.node_id
        for k, v in params.items():
            if isinstance(v, torch.Tensor):
                logger.info(f"  {k}: {v.shape}")

        # Concatenate global_orient and pose
        logger.info(f"[MANOParams.forward] Concatenating global_orient + pose...")
        logger.info(f"  {node_id}.global_orient: {params[f'{node_id}.global_orient'].shape}")
        logger.info(f"  {node_id}.pose: {params[f'{node_id}.pose'].shape}")

        # ================================================================
        # ✅ FIX: Concatenate on dim=2 (feature dimension), not dim=1
        # ================================================================
        # Shapes: global_orient [B, 1, 3], pose [B, 1, 45]
        # Result: full_pose [B, 1, 48]

        full_pose = torch.cat(
            (params[f"{node_id}.global_orient"], params[f"{node_id}.pose"]),
            dim=2  # ← CHANGED from dim=1 to dim=2
        )
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
