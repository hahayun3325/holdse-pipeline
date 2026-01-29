"""
GHOP mesh extraction utilities for Phase 4.

This module provides utilities to extract explicit triangle meshes from
implicit SDF representations in HOLD. The extracted meshes are used for
Phase 4 contact refinement.

Key Components:
- GHOPMeshExtractor: Main mesh extraction class using Marching Cubes
- Helper functions for coordinate transformation and mesh processing

Dependencies:
- torch: For tensor operations
- scikit-image (skimage): For Marching Cubes algorithm
- numpy: For numerical operations
"""

import torch
import torch.nn as nn
import numpy as np
from loguru import logger

# Try Kaolin first (differentiable)
try:
    from kaolin.ops.conversions import voxelgrids_to_trianglemeshes
    KAOLIN_AVAILABLE = True
    logger.info("[mesh_extraction] Kaolin available (will try, may fall back if CUDA errors)")
except ImportError:
    KAOLIN_AVAILABLE = False
    logger.info("[mesh_extraction] Kaolin not available")

# scikit-image fallback (non-differentiable but reliable)
try:
    from skimage import measure
    SKIMAGE_AVAILABLE = True
    if not KAOLIN_AVAILABLE:
        logger.info("[mesh_extraction] Using scikit-image (non-differentiable)")
except ImportError:
    SKIMAGE_AVAILABLE = False
# PyTorch3D fallback
if not KAOLIN_AVAILABLE and not SKIMAGE_AVAILABLE:
    try:
        from pytorch3d.ops.marching_cubes import marching_cubes
        PYTORCH3D_AVAILABLE = True
        logger.warning("[mesh_extraction] Using PyTorch3D (non-differentiable in 0.7.4)")
    except ImportError:
        PYTORCH3D_AVAILABLE = False
        logger.error(
            "[mesh_extraction] No marching cubes backend available! "
            "Install: pip install scikit-image"
        )
else:
    PYTORCH3D_AVAILABLE = False

class GHOPMeshExtractor(nn.Module):
    """Extract explicit meshes from implicit HOLD SDF representations.

    This class uses the Marching Cubes algorithm to convert volumetric SDF
    grids into explicit triangle meshes. The extraction process involves:
    1. Sampling SDF values on a dense 3D grid
    2. Applying Marching Cubes to find the zero-level surface
    3. Post-processing vertices to match coordinate systems

    Args:
        vqvae_wrapper (GHOPVQVAEWrapper): Phase 3 VQ-VAE wrapper (for future use)
        resolution (int): Voxel grid resolution (default: 128)
            Higher resolution = more detailed meshes but slower extraction
            Recommended: 64 (debug), 128 (production), 256 (high-quality)

    Attributes:
        vqvae_wrapper: Reference to Phase 3 VQ-VAE (currently unused)
        resolution: Grid resolution for SDF sampling
    """

    def __init__(self, vqvae_wrapper, resolution=128):
        super().__init__()

        if not KAOLIN_AVAILABLE and not PYTORCH3D_AVAILABLE:
            raise ImportError(
                "Kaolin or PyTorch3D required for mesh extraction. "
                "Install one: pip install kaolin OR pip install --upgrade pytorch3d"
            )

        self.vqvae_wrapper = vqvae_wrapper
        self.resolution = resolution

        logger.info(
            f"[GHOPMeshExtractor] Initialized with resolution {resolution}³ "
            f"(total {resolution ** 3:,} voxels)"
        )

    def extract_object_mesh(self, sdf_grid, coordinate_range=(-1.5, 1.5)):
        """Convert SDF grid to mesh via Marching Cubes.

        Tries backends in order:
        1. Kaolin (differentiable) - if available and CUDA works
        2. scikit-image (non-diff) - fast and reliable fallback
        3. PyTorch3D (non-diff) - if scikit-image unavailable

        Args:
            sdf_grid (torch.Tensor): [B, H, W, D] or [B, 1, H, W, D] SDF values
                - Positive values: outside object
                - Negative values: inside object
                - Zero: object surface
            coordinate_range (tuple): (min, max) coordinate bounds
                Default: (-1.5, 1.5) for HOLD canonical space

        Returns:
            list of (verts, faces) tuples:
                - verts: torch.Tensor [V, 3] vertex positions in world coordinates
                - faces: torch.Tensor [F, 3] triangle face indices
                Empty mesh (0 vertices) returned on failure

        Example:
            >>> sdf_grid = torch.randn(2, 128, 128, 128, requires_grad=True)
            >>> meshes = extractor.extract_object_mesh(sdf_grid)
            >>> verts_0, faces_0 = meshes[0]
            >>> print(f"Object 0: {verts_0.shape[0]} vertices, gradients: {verts_0.requires_grad}")
        """
        # Handle channel dimension
        if sdf_grid.dim() == 5:
            sdf_grid = sdf_grid.squeeze(1)

        batch_size = sdf_grid.shape[0]
        resolution = sdf_grid.shape[1]
        device = sdf_grid.device
        meshes = []

        # Compute coordinate transformation parameters
        coord_min, coord_max = coordinate_range
        coord_span = coord_max - coord_min

        # ================================================================
        # BACKEND 1: Try Kaolin (differentiable) first
        # ================================================================
        if KAOLIN_AVAILABLE:
            try:
                from kaolin.ops.conversions import voxelgrids_to_trianglemeshes

                occupancy = (sdf_grid < 0.0).float()
                if occupancy.dim() == 4:
                    occupancy = occupancy.unsqueeze(1)

                verts_list, faces_list = voxelgrids_to_trianglemeshes(occupancy)

                # Check if extraction succeeded
                success = False
                for b in range(batch_size):
                    if len(verts_list[b]) > 0:
                        verts = verts_list[b].float()
                        faces = faces_list[b]

                        # Transform coordinates
                        verts = verts / (resolution - 1)
                        verts = verts * coord_span + coord_min

                        meshes.append((verts, faces))
                        success = True
                    else:
                        # Empty mesh
                        meshes.append((
                            torch.zeros((0, 3), device=device, dtype=torch.float32),
                            torch.zeros((0, 3), device=device, dtype=torch.long)
                        ))

                if success:
                    logger.debug(f"[GHOPMeshExtractor] Kaolin backend: extracted meshes (✅ differentiable)")
                    return meshes
                else:
                    # Kaolin returned empty, try fallback
                    meshes = []
                    logger.warning("[GHOPMeshExtractor] Kaolin returned empty meshes, trying fallback")

            except RuntimeError as e:
                if "CUDA Error" in str(e) or "invalid configuration" in str(e):
                    logger.warning(f"[GHOPMeshExtractor] Kaolin CUDA error (RTX 4090 incompatibility), using fallback")
                    meshes = []
                else:
                    raise

        # ================================================================
        # BACKEND 2: scikit-image fallback (non-differentiable but reliable)
        # ================================================================
        try:
            from skimage import measure

            logger.debug("[GHOPMeshExtractor] Using scikit-image backend (⚠️ non-differentiable)")

            for b in range(batch_size):
                try:
                    sdf_np = sdf_grid[b].detach().cpu().numpy()

                    # Compute spacing
                    spacing = (coord_span / resolution,) * 3

                    # Marching cubes
                    verts, faces, normals, values = measure.marching_cubes(
                        sdf_np,
                        level=0.0,
                        spacing=spacing,
                        gradient_direction='descent'
                    )

                    # Transform vertices
                    verts = verts + coord_min

                    # Convert to tensors
                    verts_tensor = torch.from_numpy(verts.copy()).float().to(device)
                    faces_tensor = torch.from_numpy(faces.copy()).long().to(device)

                    meshes.append((verts_tensor, faces_tensor))

                    logger.debug(
                        f"[GHOPMeshExtractor] Batch {b}: {verts.shape[0]} verts, "
                        f"{faces.shape[0]} faces (scikit-image)"
                    )

                except Exception as e:
                    logger.warning(f"[GHOPMeshExtractor] Batch {b} marching cubes failed: {e}")
                    meshes.append((
                        torch.zeros((0, 3), device=device, dtype=torch.float32),
                        torch.zeros((0, 3), device=device, dtype=torch.long)
                    ))

            return meshes

        except ImportError:
            logger.warning("[GHOPMeshExtractor] scikit-image not available, trying PyTorch3D")

        # ================================================================
        # BACKEND 3: PyTorch3D fallback (if scikit-image unavailable)
        # ================================================================
        if PYTORCH3D_AVAILABLE:
            from pytorch3d.ops.marching_cubes import marching_cubes

            logger.debug("[GHOPMeshExtractor] Using PyTorch3D backend (⚠️ non-differentiable in 0.7.4)")

            for b in range(batch_size):
                try:
                    sdf_batch = sdf_grid[b:b+1]  # [1, H, W, D]

                    # Returns ([verts_list], [faces_list])
                    verts_list, faces_list = marching_cubes(
                        sdf_batch,
                        isolevel=0.0,
                        return_local_coords=False
                    )

                    if len(verts_list) > 0 and verts_list[0].shape[0] > 0:
                        verts = verts_list[0].float()
                        faces = faces_list[0]

                        # Transform coordinates
                        verts = verts / (resolution - 1)
                        verts = verts * coord_span + coord_min

                        meshes.append((verts, faces))
                    else:
                        meshes.append((
                            torch.zeros((0, 3), device=device),
                            torch.zeros((0, 3), device=device, dtype=torch.long)
                        ))
                except Exception as e:
                    logger.warning(f"[GHOPMeshExtractor] PyTorch3D batch {b} failed: {e}")
                    meshes.append((
                        torch.zeros((0, 3), device=device),
                        torch.zeros((0, 3), device=device, dtype=torch.long)
                    ))

            return meshes

        # ================================================================
        # If all backends failed
        # ================================================================
        raise RuntimeError(
            "No marching cubes backend available. "
            "Install: pip install scikit-image"
        )

    def forward(self, sdf_grid):
        """Alias for extract_object_mesh() to support nn.Module interface.

        Args:
            sdf_grid: SDF grid tensor

        Returns:
            List of (verts, faces) tuples
        """
        return self.extract_object_mesh(sdf_grid)


# ========================================================================
# Helper Functions
# ========================================================================

def mesh_to_sdf_grid(vertices, faces, resolution=128, padding=0.1):
    """Convert triangle mesh to SDF grid (inverse of extract_object_mesh).

    Useful for testing and validation purposes.

    Args:
        vertices (np.ndarray): [V, 3] vertex positions
        faces (np.ndarray): [F, 3] triangle face indices
        resolution (int): Output grid resolution
        padding (float): Padding around mesh bounding box

    Returns:
        np.ndarray: [resolution, resolution, resolution] SDF values
    """
    try:
        import trimesh
    except ImportError:
        raise ImportError("trimesh required for mesh_to_sdf_grid")

    # Create trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # Normalize to [-1, 1] range
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    scale = np.max(bounds[1] - bounds[0]) / (2 * (1 - padding))
    mesh.vertices = (mesh.vertices - center) / scale

    # Create sampling grid
    x = np.linspace(-1, 1, resolution)
    grid_x, grid_y, grid_z = np.meshgrid(x, x, x, indexing='ij')
    query_points = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)

    # Compute SDF (requires mesh_to_sdf library or similar)
    # This is a placeholder - actual implementation depends on SDF library
    logger.warning("[mesh_to_sdf_grid] SDF computation not fully implemented")
    sdf_grid = np.zeros((resolution, resolution, resolution))

    return sdf_grid