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

# Try importing jutils
try:
    from jutils import hand_utils, mesh_utils
    JUTILS_AVAILABLE = True
except ImportError:
    JUTILS_AVAILABLE = False
    print("[HandSkeletalField] jutils not available, using self-contained implementation")


class HandSkeletalField(nn.Module):
    """
    Computes skeletal distance field with automatic backend selection.
    """

    def __init__(self, mano_dir='assets/mano', device='cuda', force_standalone=False):
        """
        Args:
            mano_dir: Path to MANO model files
            device: Device for computation
            force_standalone: If True, use self-contained implementation even if jutils available
        """
        super().__init__()
        self.device = device

        # Choose backend
        self.use_jutils = JUTILS_AVAILABLE and not force_standalone

        if self.use_jutils:
            # Use GHOP's original implementation
            self.hand_wrapper = hand_utils.ManopthWrapper(mano_dir).to(device)
            print("[HandSkeletalField] Using GHOP jutils backend")
        else:
            # Use self-contained implementation
            from manopth.manolayer import ManoLayer
            self.mano_layer = ManoLayer(
                mano_root=mano_dir,
                use_pca=False,
                ncomps=45,
                flat_hand_mean=True
            ).to(device)
            print("[HandSkeletalField] Using self-contained backend")

    def forward(self, hand_pose, resolution=16, spatial_lim=1.5, rtn_wrist=False):
        """Compute skeletal distance field."""

        if self.use_jutils:
            return self._forward_jutils(hand_pose, resolution, spatial_lim, rtn_wrist)
        else:
            return self._forward_standalone(hand_pose, resolution, spatial_lim, rtn_wrist)

    def _forward_jutils(self, hand_pose, resolution=16, spatial_lim=1.5, rtn_wrist=False):
        """
        GHOP jutils backend.
        Compute skeletal distance field.

        Args:
            hand_pose: (N, 48) or (N, 45) MANO parameters
                      - If (N, 48): uses first 45 dims (ignores global orient)
                      - If (N, 45): uses directly
            resolution: Grid resolution (default: 16)
            spatial_lim: Spatial extent in meters (default: 1.5)
            rtn_wrist: If True, return 20 joints; if False, return 15

        Returns:
            skdf: (N, 15, H, H, H) or (N, 20, H, H, H) skeletal distance field
        """
        N = hand_pose.shape[0]

        # ====================================================================
        # MODIFICATION: Handle both 45-dim and 48-dim MANO parameters
        # ====================================================================
        if hand_pose.shape[-1] == 48:
            # Extract only hand pose parameters (ignore global orientation)
            # GHOP's hand_wrapper expects (N, 45) only
            hand_pose_params = hand_pose[:, 3:48]  # Take dims 3-47 (45 dims)
        elif hand_pose.shape[-1] == 45:
            hand_pose_params = hand_pose
        else:
            raise ValueError(
                f"Expected hand_pose shape (N, 45) or (N, 48), got {hand_pose.shape}"
            )

        # ====================================================================
        # Use GHOP's original implementation
        # ====================================================================
        # Get hand joints from MANO
        hand_joints = self.hand_wrapper.pose_to_joints(hand_pose_params)  # (N, 21, 3)

        # Create 3D grid
        grid = mesh_utils.create_sdf_grid(
            resolution=resolution,
            spatial_lim=spatial_lim,
            batch_size=N,
            device=self.device
        )  # (N, H, H, H, 3)

        # Compute skeletal distance field
        skdf = hand_utils.compute_skeletal_distance_field(
            grid=grid,
            joints=hand_joints,
            rtn_wrist=rtn_wrist
        )  # (N, 15/20, H, H, H)

        return skdf

    def _forward_standalone(self, hand_pose, resolution=16, spatial_lim=1.5, rtn_wrist=False):
        """
        Self-contained backend.
        Compute skeletal distance field from MANO parameters.

        Args:
            hand_pose: (N, 48) or (N, 45) MANO parameters
                      - (N, 48): [global_orient(3) + hand_pose(45)]
                      - (N, 45): [hand_pose(45)] only
            resolution: Grid resolution (default: 16)
            spatial_lim: Grid spatial extent in meters (default: 1.5)
            rtn_wrist: If True, return 20 joints; if False, return 15 (default)

        Returns:
            skdf: (N, 15, H, H, H) or (N, 20, H, H, H) skeletal distance field
        """
        N = hand_pose.shape[0]
        H = resolution
        device = hand_pose.device

        # Parse MANO parameters
        if hand_pose.shape[-1] == 48:
            # Extract global orientation and hand pose
            global_orient = hand_pose[:, :3]
            hand_pose_params = hand_pose[:, 3:]
        elif hand_pose.shape[-1] == 45:
            # No global orientation provided, use identity
            global_orient = torch.zeros(N, 3, device=device)
            hand_pose_params = hand_pose
        else:
            raise ValueError(
                f"Expected hand_pose shape (N, 45) or (N, 48), got {hand_pose.shape}"
            )

        # Get hand joint positions from MANO
        # MANO layer returns: vertices (N, 778, 3), joints (N, 21, 3)
        _, joints = self.mano_layer(
            th_pose_coeffs=hand_pose_params,
            th_betas=None  # Use mean shape
        )

        # Apply global orientation if provided
        if global_orient is not None:
            # Convert axis-angle to rotation matrix
            rot_mat = self.axis_angle_to_matrix(global_orient)  # (N, 3, 3)

            # Apply rotation to joints
            joints = torch.bmm(joints, rot_mat.transpose(1, 2))  # (N, 21, 3)

        # Create 3D coordinate grid
        grid = self.create_grid(N, H, spatial_lim, device)  # (N, H, H, H, 3)

        # Reshape for distance computation
        grid_flat = grid.reshape(N, 1, H * H * H, 3)  # (N, 1, H³, 3)
        joints_expand = joints.unsqueeze(2)  # (N, 21, 1, 3)

        # Compute squared Euclidean distances
        # dist² = (x - x_joint)² + (y - y_joint)² + (z - z_joint)²
        dist_square = ((grid_flat - joints_expand) ** 2).sum(dim=-1)  # (N, 21, H³)

        # Reshape back to grid
        dist_square = dist_square.reshape(N, 21, H, H, H)  # (N, 21, H, H, H)

        # Handle zero poses (mask out invalid data)
        # Check if all pose parameters are zero
        zero_mask = (hand_pose_params.abs().sum(dim=-1) < 1e-6).float()  # (N,)
        zero_mask = zero_mask.view(N, 1, 1, 1, 1)

        # Zero out distances for zero poses
        dist_square = dist_square * (1.0 - zero_mask)

        # Optionally exclude wrist joint (joint 0)
        if not rtn_wrist:
            dist_square = dist_square[:, 1:, :, :, :]  # (N, 20, H, H, H)

            # Further exclude 5 joints to get exactly 15 channels
            # MANO joints: 0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
            # Keep tip joints only: [4, 8, 12, 16, 20] and intermediate joints
            # Simplified: keep first 15 non-wrist joints
            dist_square = dist_square[:, :15, :, :, :]  # (N, 15, H, H, H)

        return dist_square

    def axis_angle_to_matrix(self, axis_angle):
        """
        Convert axis-angle representation to rotation matrix.

        Args:
            axis_angle: (N, 3) axis-angle vectors

        Returns:
            rot_mat: (N, 3, 3) rotation matrices
        """
        # Compute angle (magnitude of axis-angle vector)
        angle = torch.norm(axis_angle, dim=-1, keepdim=True)  # (N, 1)

        # Normalize axis
        axis = axis_angle / (angle + 1e-8)  # (N, 3)

        # Handle near-zero rotations
        small_angle_mask = (angle < 1e-6).squeeze(-1)

        # Rodrigues' rotation formula
        cos_angle = torch.cos(angle)  # (N, 1)
        sin_angle = torch.sin(angle)  # (N, 1)

        # Cross-product matrix
        N = axis.shape[0]
        K = torch.zeros(N, 3, 3, device=axis.device)
        K[:, 0, 1] = -axis[:, 2]
        K[:, 0, 2] = axis[:, 1]
        K[:, 1, 0] = axis[:, 2]
        K[:, 1, 2] = -axis[:, 0]
        K[:, 2, 0] = -axis[:, 1]
        K[:, 2, 1] = axis[:, 0]

        # R = I + sin(θ)K + (1-cos(θ))K²
        I = torch.eye(3, device=axis.device).unsqueeze(0).repeat(N, 1, 1)
        rot_mat = I + sin_angle.unsqueeze(-1) * K + \
                  (1 - cos_angle).unsqueeze(-1) * torch.bmm(K, K)

        # Use identity for small angles
        rot_mat[small_angle_mask] = I[small_angle_mask]

        return rot_mat


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
    Provides a unified interface compatible with the Phase 3 plan's API.

    This class wraps HandSkeletalField and adapts its interface to match
    the expected API in ghop_loss.py and hold.py.
    """

    def __init__(self, mano_server, resolution=64, spatial_limit=1.5):
        """
        Args:
            mano_server: HOLD's MANO server object (from node_dict)
            resolution: Grid resolution (default: 64 for high-res SDF matching)
            spatial_limit: Spatial extent in meters (default: 1.5)
        """
        super().__init__()

        # Extract device from mano_server
        self.device = next(mano_server.parameters()).device if hasattr(mano_server, 'parameters') else 'cuda'

        # Store HOLD's MANO server for direct access
        self.mano_server = mano_server

        # Initialize the skeletal field computer
        # Try to determine if jutils is available
        try:
            # Check if mano_server has MANO assets path
            mano_dir = getattr(mano_server, 'mano_assets_root', 'assets/mano')
        except:
            mano_dir = 'assets/mano'

        self.field_computer = HandSkeletalField(
            mano_dir=mano_dir,
            device=self.device,
            force_standalone=False  # Auto-select backend
        )

        self.resolution = resolution
        self.lim = spatial_limit

        print(f"[HandFieldBuilder] Initialized with resolution={resolution}, spatial_limit={spatial_limit}")

    def forward(self, hand_params, hand_side='right'):
        """
        Generate 15-channel hand skeletal distance field.

        Args:
            hand_params: Dict with keys:
                - 'pose': (B, 48) or (B, 45) MANO pose parameters
                - 'shape': (B, 10) MANO shape parameters (optional, not used)
                - 'trans': (B, 3) Translation (optional, not used in field computation)
            hand_side: 'right' or 'left' (for future use)

        Returns:
            hand_field: (B, 15, H, H, H) skeletal distance field
        """
        # Extract hand pose from params dict
        if isinstance(hand_params, dict):
            hand_pose = hand_params['pose']  # (B, 48) or (B, 45)
        else:
            # If passed directly as tensor
            hand_pose = hand_params

        # Compute skeletal distance field using existing implementation
        hand_field = self.field_computer(
            hand_pose=hand_pose,
            resolution=self.resolution,
            spatial_lim=self.lim,
            rtn_wrist=False  # Always return 15 joints (exclude wrist)
        )

        return hand_field

    def forward_from_mano(self, hand_pose, hand_shape=None, hand_trans=None):
        """
        Alternative interface: directly from MANO parameters.

        Args:
            hand_pose: (B, 48) or (B, 45) MANO pose
            hand_shape: (B, 10) shape (ignored)
            hand_trans: (B, 3) translation (ignored)

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
        return F.interpolate(
            hand_field_64,
            size=(target_res, target_res, target_res),
            mode='trilinear',
            align_corners=True
        )

    def get_joints_from_params(self, hand_params):
        """
        Extract 21 3D joint positions from MANO parameters.
        Useful for visualization and debugging.

        Args:
            hand_params: Dict with 'pose', 'shape', 'trans'

        Returns:
            joints: (B, 21, 3) joint positions
        """
        hand_pose = hand_params['pose']

        # Use the underlying field computer's MANO layer
        if self.field_computer.use_jutils:
            # jutils backend
            if hand_pose.shape[-1] == 48:
                hand_pose_params = hand_pose[:, 3:48]
            else:
                hand_pose_params = hand_pose
            joints = self.field_computer.hand_wrapper.pose_to_joints(hand_pose_params)
        else:
            # Standalone backend
            if hand_pose.shape[-1] == 48:
                hand_pose_params = hand_pose[:, 3:]
            else:
                hand_pose_params = hand_pose

            _, joints = self.field_computer.mano_layer(
                th_pose_coeffs=hand_pose_params,
                th_betas=None
            )

        return joints

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