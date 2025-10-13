"""
Interaction grid builder: combines object latent (3 ch) + hand SKDF (15 ch).
"""
import torch
import torch.nn as nn
from src.model.ghop.hand_field import HandSkeletalField
from src.utils.meshing import create_sdf_grid


class InteractionGridBuilder(nn.Module):
    """
    Builds 18-channel interaction grid from HOLD outputs.
    """

    def __init__(self, vqvae, config):
        super().__init__()
        self.vqvae = vqvae
        self.config = config

        # Initialize hand field
        self.hand_field = HandSkeletalField(
            mano_dir=config.get('mano_dir', 'assets/mano'),
            device=config.get('device', 'cuda')
        )

        # Spatial parameters
        self.resolution = config.get('grid_resolution', 16)
        self.spatial_lim = config.get('spatial_lim', 1.5)

    def forward(self, obj_sdf, hand_pose):
        """
        Build interaction grid.

        Args:
            obj_sdf: [B, 1, 64, 64, 64] - Object SDF
            hand_pose: [B, 48 or 45] - MANO parameters

        Returns:
            interaction_grid: [B, 18, 16, 16, 16]
            components: Dict with obj_latent and hand_skdf
        """
        B = len(obj_sdf)

        # Extract pose parameters (first 45 dims)
        if hand_pose.shape[-1] == 48:
            hand_pose_only = hand_pose[:, :45]
        else:
            hand_pose_only = hand_pose

        # 1. Encode object to 3-channel latent (no quantization)
        with torch.no_grad():
            obj_latent = self.vqvae.encode_to_prequant(obj_sdf)  # [B, 3, 16, 16, 16]

        # 2. Compute 15-channel hand skeletal distance field
        hand_skdf = self.hand_field(
            hand_pose=hand_pose_only,
            resolution=self.resolution,
            spatial_lim=self.spatial_lim,
            rtn_wrist=False
        )  # [B, 15, 16, 16, 16]

        # 3. Concatenate
        interaction_grid = torch.cat([obj_latent, hand_skdf], dim=1)  # [B, 18, 16, 16, 16]

        components = {
            'obj_latent': obj_latent,
            'hand_skdf': hand_skdf
        }

        return interaction_grid, components

    def extract_object_sdf(self, object_node, resolution=64):
        """
        Extract object SDF from HOLD's ObjectNode.

        Args:
            object_node: HOLD's ObjectNode with implicit_network
            resolution: SDF grid resolution

        Returns:
            obj_sdf: [B, 1, 64, 64, 64]
        """
        device = next(object_node.parameters()).device

        # Create query grid
        xyz_grid = create_sdf_grid(1, resolution, self.spatial_lim, device=device)
        xyz_flat = xyz_grid.reshape(-1, 3)  # [64³, 3]

        # Query implicit surface
        with torch.no_grad():
            sdf_values = object_node.implicit_network(xyz_flat)  # [64³, 1]

        obj_sdf = sdf_values.reshape(1, 1, resolution, resolution, resolution)

        return obj_sdf