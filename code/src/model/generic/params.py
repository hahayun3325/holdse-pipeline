import torch
import torch.nn as nn
from common.xdict import xdict
from loguru import logger


class GenericParams(nn.Module):
    def __init__(self, num_frames, params_dim, node_id):
        super(GenericParams, self).__init__()
        self.num_frames = num_frames

        # parameter dims to keep track
        self.params_dim = params_dim

        self.node_id = node_id

        self.param_names = self.params_dim.keys()

        # ========== ADD THIS FLAG ==========
        self._preserve_checkpoint = False
        # ===================================

        # init variables based on dim specs
        for param_name in self.param_names:
            if param_name == "betas":
                param = nn.Embedding(1, self.params_dim[param_name])
            else:
                param = nn.Embedding(num_frames, self.params_dim[param_name])
            param.weight.data.fill_(0)
            param.weight.requires_grad = False
            setattr(self, param_name, param)

    def init_parameters(self, param_name, data, requires_grad=False):
        getattr(self, param_name).weight.data = data[..., : self.params_dim[param_name]]
        getattr(self, param_name).weight.requires_grad = requires_grad

    def set_requires_grad(self, param_name, requires_grad=True):
        getattr(self, param_name).weight.requires_grad = requires_grad

    def forward(self, frame_ids):
        """
        Given frame_ids, return parameters for the frames.

        DIAGNOSTIC VERSION to find shape mismatch.

        Args:
            frame_ids: Tensor of frame indices
                - Expected: [B] or [B, 1] where B is batch size
                - Example: tensor([[0], [1]]) for batch=2

        Returns:
            params: xdict with node_id prefix
                - Keys: "{node_id}.{param_name}"
                - Values: Retrieved parameters for frame_ids
        """
        logger.info("="*70)
        logger.info("[GenericParams.forward] ENTRY")
        logger.info("="*70)
        logger.info(f"[GenericParams.forward] node_id: {self.node_id}")
        logger.info(f"[GenericParams.forward] num_frames: {self.num_frames}")

        # ============================================================
        # CRITICAL DEBUG: Input frame_ids analysis
        # ============================================================
        logger.info(f"[GenericParams.forward] Input frame_ids:")
        logger.info(f"  shape: {frame_ids.shape}")
        logger.info(f"  dtype: {frame_ids.dtype}")
        logger.info(f"  device: {frame_ids.device}")
        logger.info(f"  ndim: {frame_ids.ndim}")

        # Show actual values
        if frame_ids.numel() <= 20:
            logger.info(f"  values: {frame_ids}")
        else:
            logger.info(f"  first 10 values: {frame_ids.flatten()[:10]}")

        # Check for unexpected shapes
        expected_batch_size = frame_ids.shape[0]
        logger.info(f"  Expected batch size: {expected_batch_size}")

        # ============================================================
        # CRITICAL DEBUG: Stored parameter analysis
        # ============================================================
        logger.info(f"[GenericParams.forward] Stored parameters (nn.Embedding):")
        for param_name in self.param_names:
            embedding = getattr(self, param_name)
            logger.info(f"  {param_name}:")
            logger.info(f"    num_embeddings: {embedding.num_embeddings}")
            logger.info(f"    embedding_dim: {embedding.embedding_dim}")
            logger.info(f"    weight.shape: {embedding.weight.shape}")

        # ============================================================
        # CRITICAL: Parameter retrieval with detailed logging
        # ============================================================
        params = xdict()
        for param_name in self.param_names:
            logger.info("-"*70)
            logger.info(f"[GenericParams.forward] Retrieving: {param_name}")

            embedding_layer = getattr(self, param_name)
            logger.info(f"  Embedding layer: {embedding_layer}")
            logger.info(f"  Weight shape: {embedding_layer.weight.shape}")

            if param_name == "betas":
                # Special case: betas uses zeros
                indices = torch.zeros_like(frame_ids)
                logger.info(f"  Using zero indices for betas: {indices.shape}")
                logger.info(f"  Calling embedding(zeros): embedding.weight[[0, 0, ...]]")

                retrieved = embedding_layer(indices)

                logger.info(f"  Retrieved {param_name}: {retrieved.shape}")

            else:
                # Normal case: use frame_ids directly
                logger.info(f"  Using frame_ids directly: {frame_ids.shape}")
                logger.info(f"  Calling embedding(frame_ids)")
                logger.info(f"  This will index: embedding.weight[frame_ids]")

                # CRITICAL: This is where the issue might be
                # nn.Embedding(frame_ids) behavior:
                # - Input: [B, S] -> Output: [B, S, embedding_dim]
                # - Input: [B] -> Output: [B, embedding_dim]

                retrieved = embedding_layer(frame_ids)

                logger.info(f"  Retrieved {param_name}: {retrieved.shape}")

                # ====================================================
                # CHECK FOR SHAPE MISMATCH
                # ====================================================
                if retrieved.shape[0] != expected_batch_size:
                    logger.error("="*70)
                    logger.error(f"[GenericParams.forward] ❌ SHAPE MISMATCH DETECTED!")
                    logger.error("="*70)
                    logger.error(f"  Parameter: {param_name}")
                    logger.error(f"  Expected batch dim: {expected_batch_size}")
                    logger.error(f"  Got batch dim: {retrieved.shape[0]}")
                    logger.error(f"  Expansion factor: {retrieved.shape[0] / expected_batch_size}")
                    logger.error(f"  frame_ids input: {frame_ids}")
                    logger.error(f"  frame_ids shape: {frame_ids.shape}")
                    logger.error(f"  Retrieved shape: {retrieved.shape}")
                    logger.error(f"  Embedding weight shape: {embedding_layer.weight.shape}")
                    logger.error("="*70)

                    # Additional diagnostics
                    logger.error(f"[GenericParams.forward] Debugging info:")
                    logger.error(f"  frame_ids.numel(): {frame_ids.numel()}")
                    logger.error(f"  frame_ids.flatten(): {frame_ids.flatten()}")
                    logger.error(f"  torch.unique(frame_ids): {torch.unique(frame_ids)}")

            params[param_name] = retrieved

        # ============================================================
        # Add node_id prefix
        # ============================================================
        logger.info("="*70)
        logger.info(f"[GenericParams.forward] BEFORE prefix - params keys:")
        for k in params.keys():
            logger.info(f"  {k}: {params[k].shape}")

        params = params.prefix(self.node_id + ".")

        logger.info(f"[GenericParams.forward] AFTER prefix - params keys:")
        for k in params.keys():
            logger.info(f"  {k}: {params[k].shape}")

        logger.info("="*70)
        logger.info("[GenericParams.forward] RETURNING")
        logger.info("="*70)

        return params

    def load_params(self, case):
        raise NotImplementedError

    def preserve_checkpoint_values(self):
        """Mark that checkpoint values should be preserved"""
        self._preserve_checkpoint = True
        logger.info(f"[{self.node_id}] Checkpoint preservation enabled")

    def defrost(self, keys=None):
        if keys is None:
            keys = self.param_names
        for param_name in keys:
            self.set_requires_grad(param_name, requires_grad=True)

    def freeze(self, keys=None):
        if keys is None:
            keys = self.param_names
        for param_name in keys:
            self.set_requires_grad(param_name, requires_grad=False)
