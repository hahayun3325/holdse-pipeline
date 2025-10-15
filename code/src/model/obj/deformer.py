import torch
from pytorch3d import ops


class ObjectDeformer:
    def __init__(self):
        super().__init__()
        self.max_dist = 0.1

    def forward(self, x, tfs, return_weights=None, inverse=False, verts=None):
        """
        Apply object deformation transformation.

        Args:
            tfs: (batch, 4, 4) transformation matrices
            x: (batch, N, 3) points to transform
            return_weights: (unused, kept for compatibility)
            inverse: If True, apply inverse transformation (deform -> cano)
            verts: Optional vertices for outlier detection

        Returns:
            x_tf: (batch, N, 3) transformed points
            outlier_mask: Optional mask for outlier points
        """
        assert len(x.shape) == 3
        assert x.shape[2] == 3
        tfs = tfs.view(-1, 4, 4)

        # inverse: deform -> cano
        # not inverse: cano -> deform
        if inverse:
            # ============================================================
            # REPLACE LINE 18 WITH THIS BLOCK (Lines 18-70)
            # FIX: CPU inverse workaround for CUDA cuSPARSE compatibility
            # Issue: RTX 4090 + CUDA 11.1 doesn't support cusparseCreate
            # Solution: Compute matrix inverse on CPU, then move to GPU
            # Context: Object transformation matrix inversion
            # ============================================================

            device = tfs.device  # Remember original device

            if tfs.is_cuda:
                # GPU tensor - move to CPU for inverse computation
                tfs_cpu = tfs.cpu()

                try:
                    # Compute inverse on CPU (stable and supported)
                    obj_tfs_cpu = torch.inverse(tfs_cpu)
                except RuntimeError as e:
                    # Fallback 1: Regularization for near-singular matrices
                    print(f"[ObjectDeformer] Warning: Matrix inversion failed, applying regularization: {e}")
                    epsilon = 1e-6
                    batch_size = tfs_cpu.shape[0]
                    identity = torch.eye(4, device='cpu').unsqueeze(0).expand(batch_size, -1, -1)
                    tfs_cpu_reg = tfs_cpu + epsilon * identity

                    try:
                        obj_tfs_cpu = torch.inverse(tfs_cpu_reg)
                    except RuntimeError as e2:
                        # Fallback 2: Pseudo-inverse (most stable)
                        print(f"[ObjectDeformer] Warning: Using pseudo-inverse: {e2}")
                        obj_tfs_cpu = torch.linalg.pinv(tfs_cpu)

                # Move result back to GPU
                obj_tfs = obj_tfs_cpu.to(device)
            else:
                # CPU tensor - compute directly
                try:
                    obj_tfs = torch.inverse(tfs)
                except RuntimeError as e:
                    # Fallback with regularization
                    print(f"[ObjectDeformer] Warning: Matrix inversion failed, applying regularization: {e}")
                    epsilon = 1e-6
                    batch_size = tfs.shape[0]
                    identity = torch.eye(4, device=device).unsqueeze(0).expand(batch_size, -1, -1)
                    tfs_reg = tfs + epsilon * identity

                    try:
                        obj_tfs = torch.inverse(tfs_reg)
                    except RuntimeError as e2:
                        # Fallback: Pseudo-inverse
                        print(f"[ObjectDeformer] Warning: Using pseudo-inverse: {e2}")
                        obj_tfs = torch.linalg.pinv(tfs)

            # ============================================================
            # END OF FIX
            # ============================================================
        else:
            obj_tfs = tfs

        # apply transformation
        x_pad = torch.cat([x, torch.ones_like(x[:, :, :1])], dim=-1).permute(
            0, 2, 1
        )  # (N, 4)
        obj_x_tf = torch.bmm(obj_tfs, x_pad).permute(0, 2, 1)
        x_tf = obj_x_tf[:, :, :3]

        outlier_mask = None
        if verts is not None and inverse:  # points in deform space
            distance_batch, index_batch, neighbor_points = ops.knn_points(
                x, verts, K=1, return_nn=True
            )
            distance_batch = torch.clamp(distance_batch, max=4)
            distance_batch = torch.sqrt(distance_batch)
            distance_batch = distance_batch.min(dim=2).values
            outlier_mask = distance_batch > self.max_dist

        return x_tf, outlier_mask

    def forward_skinning(self, xc, cond, tfs):
        """
        Forward skinning transformation (canonical -> deformed).

        Args:
            xc: (batch, N, 3) canonical points
            cond: (unused, kept for compatibility)
            tfs: (batch, 4, 4) transformation matrices

        Returns:
            x_transformed: (batch, N, 3) deformed points
        """
        # cano -> deform
        x_transformed = self.forward(xc, tfs, inverse=False)[0]
        return x_transformed
