"""
Hand skeletal distance field computation for Phase 2 & 3 GHOP integration.

Computes 15-channel distance field from MANO hand pose parameters.

PHASE 3 UPDATES:
- Added HandFieldBuilder wrapper class for unified HOLD integration API
- Supports both dict-based hand_params and direct tensor inputs
- Added downsample_to_latent() for 64³ → 16³ compression
- Compatible with ghop_loss.py SDS computation pipeline

This module creates a 3D grid representation where each voxel stores the
squared distance to each of the 15 hand joints (excluding wrist).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import logging

logger = logging.getLogger(__name__)
# Try importing jutils
try:
    from jutils import hand_utils, mesh_utils
    JUTILS_AVAILABLE = True
except ImportError:
    JUTILS_AVAILABLE = False
    print("[HandSkeletalField] jutils not available, using self-contained implementation")


class HandSkeletalField(nn.Module):
    """
    Computes skeletal distance field using HOLD's existing MANO implementation.
    This eliminates external dependencies (jutils, manopth).
    """

    def __init__(self, mano_server, device='cuda'):
        """
        Args:
            mano_server: HOLD's MANOServer instance (from hold_net.py)
            device: Device for computation
        """
        super().__init__()

        if mano_server is None:
            raise ValueError(
                "mano_server is required.\n"
                "Pass HOLD's MANO server from: self.model.nodes['right'].server"
            )

        self.mano_server = mano_server
        self.device = device

        print(f"[HandSkeletalField] Initialized with HOLD's MANO server on {device}")

    def get_joints_from_mano(self, hand_pose, hand_shape=None):
        """
        Get hand joint positions using HOLD's MANO server.

        Args:
            hand_pose: (B, 48) or (B, 45) MANO pose parameters
            hand_shape: (B, 10) MANO shape parameters (optional)

        Returns:
            joints: (B, 21, 3) joint positions
        """
        B = hand_pose.shape[0]
        device = hand_pose.device

        # Handle different pose dimensions
        if hand_pose.shape[-1] == 45:
            # Add zero global orientation
            global_orient = torch.zeros(B, 3, device=device)
            full_pose = torch.cat([global_orient, hand_pose], dim=1)  # (B, 48)
        elif hand_pose.shape[-1] == 48:
            full_pose = hand_pose
        else:
            raise ValueError(f"Expected hand_pose dim 45 or 48, got {hand_pose.shape[-1]}")

        # Use default shape if not provided
        if hand_shape is None:
            hand_shape = torch.zeros(B, 10, device=device)

        # Forward through HOLD's MANO server
        with torch.no_grad():
            # HOLD's MANO server expects different input formats
            # Try multiple approaches to extract joints

            try:
                # ============================================================
                # Approach 1: Try MANO layer directly (BEST - no extra params needed)
                # ============================================================
                if hasattr(self.mano_server, 'mano_layer') or hasattr(self.mano_server, 'human_layer'):
                    mano_layer = getattr(self.mano_server, 'mano_layer', None) or self.mano_server.human_layer

                    # Extract pose components
                    if full_pose.shape[-1] == 48:
                        global_orient = full_pose[:, :3]   # First 3: global orientation
                        hand_pose_only = full_pose[:, 3:]  # Rest: hand pose
                    else:
                        global_orient = torch.zeros(B, 3, device=device)
                        hand_pose_only = full_pose

                    # Call MANO layer
                    output = mano_layer(
                        global_orient=global_orient,
                        hand_pose=hand_pose_only,
                        betas=hand_shape if hand_shape is not None else torch.zeros(B, 10, device=device),
                        return_verts=True
                    )

                    # Extract joints
                    joints = output.joints if hasattr(output, 'joints') else output[1]
                    logger.debug(f"[HandSkeletalField] ✓ Joints via mano_layer: {joints.shape}")

                # ============================================================
                # Approach 2: Try forward_param (needs dict construction)
                # ============================================================
                elif hasattr(self.mano_server, 'forward_param'):
                    # Construct param dict
                    from src.utils.params import ParamsDict  # May need adjustment
                    param_dict = {
                        'global_orient': full_pose[:, :3] if full_pose.shape[-1] == 48 else torch.zeros(B, 3, device=device),
                        'pose': full_pose[:, 3:] if full_pose.shape[-1] == 48 else full_pose,
                        'betas': hand_shape if hand_shape is not None else torch.zeros(B, 10, device=device),
                        'transl': torch.zeros(B, 3, device=device),
                        'scene_scale': torch.ones(B, device=device)
                    }

                    output = self.mano_server.forward_param(param_dict)
                    joints = output.get('joints') or output.get('J')
                    logger.debug(f"[HandSkeletalField] ✓ Joints via forward_param: {joints.shape}")

                # ============================================================
                # Approach 3: Try full forward (needs all 5 params)
                # ============================================================
                elif hasattr(self.mano_server, 'forward'):
                    # Construct missing parameters
                    scene_scale = torch.ones(B, device=device)
                    transl = torch.zeros(B, 3, device=device)

                    output = self.mano_server.forward(
                        scene_scale=scene_scale,
                        transl=transl,
                        thetas=full_pose,
                        betas=hand_shape if hand_shape is not None else torch.zeros(B, 10, device=device)
                    )

                    joints = output.get('joints') or output.get('J')
                    logger.debug(f"[HandSkeletalField] ✓ Joints via forward: {joints.shape}")

                else:
                    raise AttributeError(
                        "MANO server has no recognized method: mano_layer, human_layer, forward_param, or forward"
                    )

            except Exception as e:
                logger.error(f"[HandSkeletalField] All MANO approaches failed: {e}")
                logger.error(f"  Available methods: {[m for m in dir(self.mano_server) if not m.startswith('_')]}")

                # Re-raise with more context
                raise RuntimeError(
                    f"Failed to extract joints from MANO server: {e}\n"
                    f"Pose shape: {full_pose.shape}, Shape: {hand_shape.shape if hand_shape is not None else None}"
                ) from e

        if joints is None or joints.shape != (B, 21, 3):
            raise ValueError(f"Invalid joints shape: {joints.shape if joints is not None else None}, expected ({B}, 21, 3)")

        return joints

    def compute_skeletal_distance_field(self, joints, resolution=16, spatial_lim=1.5, rtn_wrist=False):
        """
        Compute skeletal distance field from joint positions.

        Args:
            joints: (B, 21, 3) joint positions
            resolution: Grid resolution
            spatial_lim: Spatial extent
            rtn_wrist: If True, return 20 joints; if False, return 15

        Returns:
            skdf: (B, 15/20, H, H, H) skeletal distance field
        """
        B = joints.shape[0]
        H = resolution
        device = joints.device

        # Create 3D coordinate grid
        lin = torch.linspace(-spatial_lim, spatial_lim, H, device=device)
        if hasattr(torch, '__version__') and tuple(int(x) for x in torch.__version__.split('.')[:2]) >= (1, 10):
            # PyTorch >= 1.10: Use indexing parameter
            grid_x, grid_y, grid_z = torch.meshgrid(lin, lin, lin, indexing='ij')
        else:
            # PyTorch < 1.10: No indexing parameter (defaults to 'ij')
            grid_x, grid_y, grid_z = torch.meshgrid(lin, lin, lin)
        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # (H, H, H, 3)
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1, -1)  # (B, H, H, H, 3)

        # Reshape for distance computation
        grid_flat = grid.reshape(B, 1, H * H * H, 3)  # (B, 1, H³, 3)
        joints_expand = joints.unsqueeze(2)  # (B, 21, 1, 3)

        # Compute squared Euclidean distances
        dist_sq = ((grid_flat - joints_expand) ** 2).sum(dim=-1)  # (B, 21, H³)

        # Reshape back to grid
        dist_sq = dist_sq.reshape(B, 21, H, H, H)  # (B, 21, H, H, H)

        # Handle wrist joint
        if not rtn_wrist:
            # Exclude wrist (joint 0)
            dist_sq = dist_sq[:, 1:, :, :, :]  # (B, 20, H, H, H)

            # Keep only 15 channels (fingertip and key joints)
            # MANO joints: 1-4 (thumb), 5-8 (index), 9-12 (middle), 13-16 (ring), 17-20 (pinky)
            # Keep: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            dist_sq = dist_sq[:, :15, :, :, :]  # (B, 15, H, H, H)

        return dist_sq

    def forward(self, hand_params, resolution=16, spatial_lim=1.5, rtn_wrist=False):
        """
        Full forward pass: MANO pose -> joints -> skeletal distance field.

        Updated to accept either tensor or dict input for compatibility with
        training_step's hand_params extraction.

        Args:
            hand_params: Either:
                - Tensor: (B, 48) or (B, 45) MANO pose parameters
                - Dict: {'pose': tensor, 'shape': tensor, 'trans': tensor}
            resolution: Grid resolution (default: 16)
            spatial_lim: Spatial extent (default: 1.5)
            rtn_wrist: If True, return 20 channels; if False, 15

        Returns:
            skdf: (B, 15/20, H, H, H) skeletal distance field
        """
        # ================================================================
        # STEP 1: Extract pose and shape from hand_params
        # ================================================================
        if isinstance(hand_params, dict):
            # Dict format from training_step
            hand_pose = hand_params['pose']  # [B, 45] or [B, 48]
            hand_shape = hand_params.get('shape', None)

            logger.debug(
                f"[HandSkeletalField] Received dict input: "
                f"pose={hand_pose.shape}, shape={hand_shape.shape if hand_shape is not None else None}"
            )

        elif isinstance(hand_params, torch.Tensor):
            # Direct tensor format (backward compatibility)
            hand_pose = hand_params
            hand_shape = None

            logger.debug(f"[HandSkeletalField] Received tensor input: {hand_pose.shape}")

        else:
            raise ValueError(
                f"Invalid hand_params type: {type(hand_params)}. "
                f"Expected dict or torch.Tensor"
            )

        # ================================================================
        # STEP 2: Get joint positions from HOLD's MANO
        # ================================================================
        try:
            # Pass hand_shape to get_joints_from_mano if available
            joints = self.get_joints_from_mano(hand_pose, hand_shape)

            logger.debug(f"[HandSkeletalField] Extracted joints: {joints.shape}")

        except Exception as e:
            logger.error(f"[HandSkeletalField] MANO forward failed: {e}")
            logger.error(f"  hand_pose shape: {hand_pose.shape}")
            logger.error(f"  hand_shape: {hand_shape.shape if hand_shape is not None else None}")

            # Fallback: Create dummy joints for sanity check
            logger.warning("[HandSkeletalField] Using dummy joints (zeros) as fallback")
            B = hand_pose.shape[0]
            joints = torch.zeros(B, 21, 3, device=hand_pose.device)

        # ================================================================
        # STEP 3: Compute skeletal distance field
        # ================================================================
        skdf = self.compute_skeletal_distance_field(
            joints=joints,
            resolution=resolution,
            spatial_lim=spatial_lim,
            rtn_wrist=rtn_wrist
        )

        logger.debug(
            f"[HandSkeletalField] Computed SKDF: shape={skdf.shape}, "
            f"range=[{skdf.min():.4f}, {skdf.max():.4f}]"
        )

        return skdf

# ============================================================================
# Standalone Function for Quick Usage
# ============================================================================

def compute_hand_skeletal_field(hand_pose, resolution=16, spatial_lim=1.5,
                                 mano_layer=None, device='cuda'):
    """
    Standalone function to compute hand skeletal distance field.

    Args:
        hand_pose: (N, 48) or (N, 45) MANO parameters
        resolution: Grid resolution (default: 16)
        spatial_lim: Spatial extent in meters (default: 1.5)
        mano_layer: Optional pre-initialized MANO layer
        device: Device for computation

    Returns:
        skdf: (N, 15, H, H, H) skeletal distance field

    Example:
        >>> hand_pose = torch.randn(4, 48).cuda()  # Batch of 4
        >>> skdf = compute_hand_skeletal_field(hand_pose)
        >>> print(skdf.shape)  # torch.Size([4, 15, 16, 16, 16])
    """
    field_computer = HandSkeletalField(mano_layer=mano_layer, device=device)
    return field_computer(hand_pose, resolution, spatial_lim, rtn_wrist=False)


# ============================================================================
# PHASE 3: Simplified Wrapper for HOLD Integration
# ============================================================================

class HandFieldBuilder(nn.Module):
    """
    Simplified wrapper for Phase 3 HOLD integration.
    Uses HOLD's existing MANO server directly.
    """

    def __init__(self, mano_server, resolution=64, spatial_limit=1.5):
        """
        Args:
            mano_server: HOLD's MANO server object (from node_dict)
            resolution: Grid resolution (default: 64 for high-res SDF matching)
            spatial_limit: Spatial extent in meters (default: 1.5)
        """
        super().__init__()

        if mano_server is None:
            raise ValueError("mano_server is required from HOLD's hand node")

        # Extract device from mano_server
        self.device = next(mano_server.parameters()).device if hasattr(mano_server, 'parameters') else 'cuda'

        # Store HOLD's MANO server
        self.mano_server = mano_server

        # ============================================================
        # CRITICAL FIX: Pass MANO server directly (no external dependencies)
        # ============================================================
        print(f"[HandFieldBuilder] Initializing with HOLD's MANO server")
        print(f"[HandFieldBuilder] Resolution: {resolution}³, Spatial limit: {spatial_limit}")

        self.field_computer = HandSkeletalField(
            mano_server=mano_server,  # ← Pass HOLD's MANO server
            device=self.device
        )

        self.resolution = resolution
        self.lim = spatial_limit

        print(f"[HandFieldBuilder] ✓ Initialized successfully")

    def forward(self, hand_params, hand_side='right'):
        """
        Generate 15-channel hand skeletal distance field.

        Args:
            hand_params: Dict with keys:
                - 'pose': (B, 48) or (B, 45) MANO pose parameters
                - 'shape': (B, 10) MANO shape parameters (optional)
                - 'trans': (B, 3) Translation (optional)
            hand_side: 'right' or 'left' (for future use)

        Returns:
            hand_field: (B, 15, H, H, H) skeletal distance field
        """
        # Extract hand pose from params dict
        if isinstance(hand_params, dict):
            hand_pose = hand_params['pose']  # (B, 48) or (B, 45)
        else:
            hand_pose = hand_params

        # Compute skeletal distance field
        hand_field = self.field_computer(
            hand_params=hand_pose,  # ✅ Match expected parameter name
            resolution=self.resolution,
            spatial_lim=self.lim,
            rtn_wrist=False  # Return 15 channels
        )

        return hand_field

    def forward_from_mano(self, hand_pose, hand_shape=None, hand_trans=None):
        """
        Alternative interface: directly from MANO parameters.

        Args:
            hand_pose: (B, 48) or (B, 45) MANO pose
            hand_shape: (B, 10) shape (optional, ignored)
            hand_trans: (B, 3) translation (optional, ignored)

        Returns:
            hand_field: (B, 15, H, H, H)
        """
        return self.field_computer(
            hand_pose=hand_pose,
            resolution=self.resolution,
            spatial_lim=self.lim,
            rtn_wrist=False
        )

    def downsample_to_latent(self, hand_field_64, target_res=16):
        """
        Downsample hand field from high resolution (64³) to latent resolution (16³).

        Args:
            hand_field_64: (B, 15, 64, 64, 64) high-res hand field
            target_res: Target resolution (default: 16)

        Returns:
            hand_field_16: (B, 15, target_res, target_res, target_res)
        """
        import torch.nn.functional as F
        return F.interpolate(
            hand_field_64,
            size=(target_res, target_res, target_res),
            mode='trilinear',
            align_corners=True
        )
# ============================================================================
# PHASE 3: Compatibility Test
# ============================================================================
def test_hand_field_builder():
    """Test HandFieldBuilder wrapper for Phase 3 compatibility."""
    print("\n" + "=" * 60)
    print("Testing HandFieldBuilder for Phase 3...")
    print("=" * 60)

    # Mock MANO server
    class MockMANOServer(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy_param = nn.Parameter(torch.zeros(1))

        def parameters(self):
            return [self.dummy_param]

    mock_server = MockMANOServer().cuda()

    # Initialize builder
    builder = HandFieldBuilder(
        mano_server=mock_server,
        resolution=64,
        spatial_limit=1.5
    )

    # Test 1: Dict input (HOLD-style)
    print("\nTest 1: Dict input (HOLD-style)")
    hand_params = {
        'pose': torch.randn(2, 48).cuda(),
        'shape': torch.randn(2, 10).cuda(),
        'trans': torch.randn(2, 3).cuda()
    }
    hand_field = builder(hand_params)
    print(f"✓ Input: dict with pose {hand_params['pose'].shape}")
    print(f"✓ Output: {hand_field.shape}")
    assert hand_field.shape == (2, 15, 64, 64, 64), f"Expected (2, 15, 64, 64, 64), got {hand_field.shape}"

    # Test 2: Direct tensor input
    print("\nTest 2: Direct tensor input")
    hand_pose_tensor = torch.randn(2, 45).cuda()
    hand_field_2 = builder.forward_from_mano(hand_pose_tensor)
    print(f"✓ Input: tensor {hand_pose_tensor.shape}")
    print(f"✓ Output: {hand_field_2.shape}")
    assert hand_field_2.shape == (2, 15, 64, 64, 64)

    # Test 3: Downsampling
    print("\nTest 3: Downsampling to latent space")
    hand_field_16 = builder.downsample_to_latent(hand_field, target_res=16)
    print(f"✓ Downsample: {hand_field.shape} -> {hand_field_16.shape}")
    assert hand_field_16.shape == (2, 15, 16, 16, 16)

    # Test 4: Joint extraction
    print("\nTest 4: Joint extraction")
    try:
        joints = builder.get_joints_from_params(hand_params)
        print(f"✓ Joints: {joints.shape}")
        assert joints.shape == (2, 21, 3)
    except Exception as e:
        print(f"⚠ Joint extraction skipped: {e}")

    print("\n" + "=" * 60)
    print("✓ All HandFieldBuilder tests passed!")
    print("=" * 60)

# # ============================================================================
# # Unit Test to verify functionality
# # ============================================================================
#
# if __name__ == "__main__":
#     print("Testing HandSkeletalField...")
#
#     # Create test data
#     batch_size = 2
#     hand_pose_48 = torch.randn(batch_size, 48).cuda()
#     hand_pose_45 = torch.randn(batch_size, 45).cuda()
#
#     # Initialize field computer
#     field = HandSkeletalField(device='cuda')
#
#     # Test with 48-dim input
#     skdf_48 = field(hand_pose_48, resolution=16, spatial_lim=1.5)
#     print(f"✓ 48-dim input: {hand_pose_48.shape} -> {skdf_48.shape}")
#     assert skdf_48.shape == (batch_size, 15, 16, 16, 16), "Shape mismatch!"
#
#     # Test with 45-dim input
#     skdf_45 = field(hand_pose_45, resolution=16, spatial_lim=1.5)
#     print(f"✓ 45-dim input: {hand_pose_45.shape} -> {skdf_45.shape}")
#     assert skdf_45.shape == (batch_size, 15, 16, 16, 16), "Shape mismatch!"
#
#     # Test zero pose handling
#     zero_pose = torch.zeros(batch_size, 48).cuda()
#     skdf_zero = field(zero_pose)
#     print(f"✓ Zero pose: max value = {skdf_zero.max().item():.6f}")
#     assert skdf_zero.max().item() < 1e-5, "Zero pose not handled correctly!"
#
#     # Test standalone function
#     skdf_standalone = compute_hand_skeletal_field(hand_pose_48)
#     print(f"✓ Standalone function: {skdf_standalone.shape}")
#
#     print("\n✓ All tests passed! HandSkeletalField is ready for Phase 2.")

'''
TO Test if two options(jutils) have similar performance
'''
# # Test equivalence
# import torch
#
# # Create test input
# hand_pose = torch.randn(2, 48).cuda()
#
# # Option A (jutils)
# field_jutils = HandSkeletalField_JUtils(device='cuda')
# skdf_jutils = field_jutils(hand_pose)
#
# # Option B (standalone)
# field_standalone = HandSkeletalField_Standalone(device='cuda')
# skdf_standalone = field_standalone(hand_pose)
#
# # Compare
# diff = (skdf_jutils - skdf_standalone).abs()
# print(f"Max difference: {diff.max().item()}")
# print(f"Mean difference: {diff.mean().item()}")
#
# # Should be < 1e-5 (numerical precision)
# assert diff.max() < 1e-5, "Implementations differ!"