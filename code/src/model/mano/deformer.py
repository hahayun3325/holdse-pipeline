import torch
import torch.nn.functional as F
from pytorch3d import ops


class KNNDeformer:
    def __init__(
        self,
        body_specs,
        max_dist=0.1,
        K=1,
        betas=None,
        server=None,
    ):
        super().__init__()

        self.max_dist = max_dist
        self.K = K
        self.server = server
        params_canoical = self.server.param_canonical.clone()
        params_canoical[:, -body_specs.shape_dim :] = (
            torch.tensor(betas).float().to(self.server.param_canonical.device)
        )
        cano_scale, cano_transl, cano_thetas, cano_betas = torch.split(
            params_canoical,
            [1, 3, body_specs.full_pose_dim, body_specs.shape_dim],
            dim=1,
        )
        output = self.server(cano_scale, cano_transl, cano_thetas, cano_betas)
        #  canonical vertices
        self.verts = output["verts"]
        self.skin_weights = output["skin_weights"]

    def forward(self, x, tfs, return_weights=True, inverse=False, verts=None):
        """
        # transform query points from one space to another given tfs

        if not inverse:
            cano -> deform
        else:
            deform -> cano

        if tfs is None:
            use canonical pose tfs
        else:
            use the given tfs
        """
        assert len(x.shape) == 3
        assert len(tfs.shape) == 4
        assert x.shape[0] == tfs.shape[0]
        assert tfs.shape[2] == 4
        assert tfs.shape[3] == 4
        curr_verts = self.verts.repeat(x.shape[0], 1, 1)
        skin_weights = self.skin_weights.repeat(x.shape[0], 1, 1)
        if x.shape[0] == 0:
            return x
        if verts is None:
            weights, outlier_mask = self.query_skinning_weights_multi(
                x, verts=curr_verts, skin_weights=skin_weights
            )
        else:
            weights, outlier_mask = self.query_skinning_weights_multi(
                x, verts=verts, skin_weights=skin_weights
            )
        if return_weights:
            return weights
        x_transformed = skinning(x, weights, tfs, inverse=inverse)
        return x_transformed, outlier_mask

    def forward_skinning(self, xc, cond, tfs):
        num_images = xc.shape[0]
        verts = self.verts.repeat(num_images, 1, 1)
        skin_weights = self.skin_weights.repeat(num_images, 1, 1)
        # cano -> deformed
        # query skining weights in cano
        weights, _ = self.query_skinning_weights_multi(
            xc, verts=verts, skin_weights=skin_weights
        )

        # LBS
        x_transformed = skinning(xc, weights, tfs, inverse=False)
        return x_transformed

    def query_skinning_weights_multi(self, pts, verts, skin_weights):
        distance_batch, index_batch, neighbor_points = ops.knn_points(
            pts, verts, K=self.K, return_nn=True
        )
        distance_batch = torch.clamp(distance_batch, max=4)
        weights_conf = torch.exp(-distance_batch)
        distance_batch = torch.sqrt(distance_batch)
        weights_conf = weights_conf / weights_conf.sum(-1, keepdim=True)

        num_parts = skin_weights.shape[2]

        # Expand index_batch for all parts
        expanded_index = index_batch[:, :, :, None].repeat(1, 1, 1, num_parts)
        skin_weights = skin_weights[:, :, None, :].repeat(1, 1, self.K, 1)
        weights_k = torch.gather(skin_weights, 1, expanded_index)

        # Multiply weights by their respective confidences and sum along the K dimension
        weights = (weights_k * weights_conf.unsqueeze(-1)).sum(dim=2).detach()

        distance_batch = distance_batch.min(dim=2).values
        outlier_mask = distance_batch > self.max_dist
        return weights, outlier_mask

    def query_weights(self, xc):
        weights = self.forward(xc, None, return_weights=True, inverse=False)
        return weights

    def forward_skinning_normal(self, xc, normal, cond, tfs, inverse=False):
        """
        Apply skinning transformation to normal vectors.

        Args:
            xc: Canonical points [B, N, 3]
            normal: Normal vectors [N, 3] or [B, N, 3]
            cond: Condition input
            tfs: Transformation matrices [B, J, 4, 4]
            inverse: If True, apply inverse transformation

        Returns:
            Transformed normals [B, N, 3]
        """
        if normal.ndim == 2:
            normal = normal.unsqueeze(0)

        w = self.query_weights(xc[0], cond)

        p_h = F.pad(normal, (0, 1), value=0)

        if inverse:
            # p:num_point, n:num_bone, i,j: num_dim+1
            tf_w = torch.einsum("bpn,bnij->bpij", w.double(), tfs.double())

            # ============================================================
            # FIX: CPU inverse workaround for CUDA cuSPARSE compatibility
            # Issue: RTX 4090 + CUDA 11.1 doesn't support cusparseCreate
            # Solution: Compute matrix inverse on CPU, then move to GPU
            # Context: Per-point transformation inverse for normal skinning
            # ============================================================

            device = tf_w.device  # Remember original device

            if tf_w.is_cuda:
                # GPU tensor - move to CPU for inverse computation
                tf_w_cpu = tf_w.cpu()

                try:
                    # Compute inverse on CPU (stable and supported)
                    tf_w_inv_cpu = tf_w_cpu.inverse()
                except RuntimeError as e:
                    # Fallback 1: Regularization for near-singular matrices
                    print(f"[KNNDeformer.forward_skinning_normal] Warning: Matrix inversion failed, applying regularization: {e}")
                    epsilon = 1e-6
                    batch_size = tf_w_cpu.shape[0]
                    num_points = tf_w_cpu.shape[1]
                    dim = tf_w_cpu.shape[-1]

                    # Create identity matrix: [B, P, D, D]
                    identity = torch.eye(dim, device='cpu').unsqueeze(0).unsqueeze(0)
                    identity = identity.expand(batch_size, num_points, -1, -1)

                    tf_w_cpu_reg = tf_w_cpu + epsilon * identity

                    try:
                        tf_w_inv_cpu = tf_w_cpu_reg.inverse()
                    except RuntimeError as e2:
                        # Fallback 2: Pseudo-inverse (most stable)
                        print(f"[KNNDeformer.forward_skinning_normal] Warning: Using pseudo-inverse: {e2}")
                        tf_w_inv_cpu = torch.linalg.pinv(tf_w_cpu)

                # Move result back to GPU
                tf_w_inv = tf_w_inv_cpu.to(device)
            else:
                # CPU tensor - compute directly
                try:
                    tf_w_inv = tf_w.inverse()
                except RuntimeError as e:
                    # Fallback with regularization
                    print(f"[KNNDeformer.forward_skinning_normal] Warning: Matrix inversion failed, applying regularization: {e}")
                    epsilon = 1e-6
                    batch_size = tf_w.shape[0]
                    num_points = tf_w.shape[1]
                    dim = tf_w.shape[-1]

                    identity = torch.eye(dim, device=device).unsqueeze(0).unsqueeze(0)
                    identity = identity.expand(batch_size, num_points, -1, -1)

                    tf_w_reg = tf_w + epsilon * identity

                    try:
                        tf_w_inv = tf_w_reg.inverse()
                    except RuntimeError as e2:
                        print(f"[KNNDeformer.forward_skinning_normal] Warning: Using pseudo-inverse: {e2}")
                        tf_w_inv = torch.linalg.pinv(tf_w)

            # Apply inverse transformation
            p_h = torch.einsum("bpij,bpj->bpi", tf_w_inv, p_h.double()).float()

            # ============================================================
            # END OF FIX
            # ============================================================
        else:
            # Forward transformation (no changes needed)
            p_h = torch.einsum(
                "bpn, bnij, bpj->bpi", w.double(), tfs.double(), p_h.double()
            ).float()

        return p_h[:, :, :3]


class MANODeformer(KNNDeformer):
    def __init__(self, max_dist, K, betas, is_rhand):
        from src.model.mano.specs import mano_specs as body_specs
        from src.model.mano.server import MANOServer

        server = MANOServer(betas=betas, is_rhand=is_rhand)
        super().__init__(
            body_specs,
            max_dist=max_dist,
            K=K,
            betas=betas,
            server=server,
        )


def skinning(x, w, tfs, inverse=False):
    """Linear blend skinning
    Args:
        x (tensor): canonical points. shape: [B, N, D]
        w (tensor): conditional input. [B, N, J]
        tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
    Returns:
        x (tensor): skinned points. shape: [B, N, D]
    """
    assert len(x.shape) == 3
    assert len(w.shape) == 3
    assert len(tfs.shape) == 4
    assert x.shape[0] == w.shape[0] == tfs.shape[0]
    assert x.shape[1] == w.shape[1]

    # LBS and inverse LBS
    x_h = F.pad(x, (0, 1), value=1.0)

    if inverse:
        # p:n_point, n:n_bone, i,k: n_dim+1
        w_tf = torch.einsum("bpn,bnij->bpij", w, tfs)

        # ============================================================
        # FIX: CPU inverse workaround for CUDA cuSPARSE compatibility
        # Issue: RTX 4090 + CUDA 11.1 doesn't support cusparseCreate
        # Solution: Compute matrix inverse on CPU, then move to GPU
        # Performance: ~2-5ms overhead per batch (acceptable)
        # ============================================================

        device = w_tf.device  # Remember original device (cuda or cpu)

        if w_tf.is_cuda:
            # GPU tensor - move to CPU for inverse computation
            w_tf_cpu = w_tf.cpu()

            try:
                # Compute inverse on CPU (stable and supported)
                w_tf_inv_cpu = w_tf_cpu.inverse()
            except RuntimeError as e:
                # Fallback: Regularization for singular matrices
                print(f"[skinning] Warning: Matrix inversion failed, applying regularization: {e}")
                epsilon = 1e-6
                identity = torch.eye(4, device='cpu').unsqueeze(0).unsqueeze(0)
                w_tf_cpu_reg = w_tf_cpu + epsilon * identity
                w_tf_inv_cpu = w_tf_cpu_reg.inverse()

            # Move inverse back to GPU
            w_tf_inv = w_tf_inv_cpu.to(device)
        else:
            # CPU tensor - compute directly
            try:
                w_tf_inv = w_tf.inverse()
            except RuntimeError as e:
                # Fallback with regularization
                print(f"[skinning] Warning: Matrix inversion failed, applying regularization: {e}")
                epsilon = 1e-6
                identity = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0)
                w_tf_reg = w_tf + epsilon * identity
                w_tf_inv = w_tf_reg.inverse()

        # Apply inverse transformation using pre-computed inverse
        x_h = torch.einsum("bpij,bpj->bpi", w_tf_inv, x_h)

        # ============================================================
        # END OF FIX
        # ============================================================
    else:
        # standard LBS: cano -> deform
        x_h = torch.einsum("bpn,bnij,bpj->bpi", w, tfs, x_h)
    return x_h[:, :, :3]
