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

try:
    from skimage import measure

    SKIMAGE_AVAILABLE = True
except ImportError:
    logger.error(
        "[mesh_extraction] scikit-image not available. "
        "Install with: pip install scikit-image"
    )
    SKIMAGE_AVAILABLE = False


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

        if not SKIMAGE_AVAILABLE:
            raise ImportError(
                "scikit-image is required for mesh extraction. "
                "Install with: pip install scikit-image"
            )

        self.vqvae_wrapper = vqvae_wrapper
        self.resolution = resolution

        logger.info(
            f"[GHOPMeshExtractor] Initialized with resolution {resolution}³ "
            f"(total {resolution ** 3:,} voxels)"
        )

    def extract_object_mesh(self, sdf_grid, coordinate_range=(-1.5, 1.5)):
        """Convert SDF grid to mesh via Marching Cubes.

        This method applies the Marching Cubes algorithm to extract the zero-level
        isosurface from a volumetric SDF grid. Each batch element is processed
        independently since object meshes may have different topologies.

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
            >>> sdf_grid = torch.randn(2, 128, 128, 128)  # Batch of 2
            >>> meshes = extractor.extract_object_mesh(sdf_grid)
            >>> verts_0, faces_0 = meshes[0]
            >>> print(f"Object 0: {verts_0.shape[0]} vertices")
        """
        if not SKIMAGE_AVAILABLE:
            raise RuntimeError("scikit-image not available for Marching Cubes")

        # Handle channel dimension
        if sdf_grid.dim() == 5:  # [B, 1, H, W, D]
            sdf_grid = sdf_grid.squeeze(1)

        batch_size = sdf_grid.shape[0]
        resolution = sdf_grid.shape[1]  # Assume cubic grid
        meshes = []

        # Compute spacing for coordinate transformation
        coord_min, coord_max = coordinate_range
        coord_span = coord_max - coord_min
        spacing = (coord_span / resolution,) * 3

        for b in range(batch_size):
            sdf = sdf_grid[b].cpu().numpy()

            try:
                # Apply Marching Cubes algorithm
                verts, faces, normals, values = measure.marching_cubes(
                    sdf,
                    level=0.0,  # Zero-level set (object surface)
                    spacing=spacing,
                    gradient_direction='descent'
                )

                # Transform vertices from grid coordinates to world coordinates
                # Grid coords: [0, resolution] -> World coords: [coord_min, coord_max]
                verts = verts + coord_min

                # Convert to tensors and move to original device
                verts_tensor = torch.from_numpy(verts).float().to(sdf_grid.device)
                faces_tensor = torch.from_numpy(faces).long().to(sdf_grid.device)

                meshes.append((verts_tensor, faces_tensor))

                logger.debug(
                    f"[GHOPMeshExtractor] Batch {b}: extracted {verts.shape[0]} verts, "
                    f"{faces.shape[0]} faces"
                )

            except Exception as e:
                # Marching Cubes can fail for various reasons:
                # - No zero-crossing in SDF grid
                # - Degenerate geometry
                # - Numerical issues
                logger.warning(
                    f"[GHOPMeshExtractor] Marching Cubes failed for batch {b}: {e}. "
                    f"Returning empty mesh."
                )

                # Empty mesh fallback
                meshes.append((
                    torch.zeros((0, 3), device=sdf_grid.device, dtype=torch.float32),
                    torch.zeros((0, 3), device=sdf_grid.device, dtype=torch.long)
                ))

        # Log statistics
        num_verts = [m[0].shape[0] for m in meshes]
        avg_verts = np.mean(num_verts) if num_verts else 0
        logger.debug(
            f"[GHOPMeshExtractor] Extracted {len(meshes)} meshes, "
            f"avg {avg_verts:.0f} vertices"
        )

        return meshes

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