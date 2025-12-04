import kaolin
import torch
from torch.autograd import grad


def compute_gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0][:, :, -3:]
    return points_grad


def compute_gradient_samples(
    pt_in_space_sampler,
    implicit_network,
    cond,
    num_pixels,
    verts_c,
    local_sigma=0.008,
    global_ratio=0.20,
):
    if verts_c is not None:
        indices = torch.randperm(verts_c.shape[1])[:num_pixels].cuda()
        verts_c = torch.index_select(verts_c, 1, indices)
        sample = pt_in_space_sampler.get_points(
            verts_c,
            local_sigma=local_sigma,
            global_ratio=global_ratio,
        )  # sample around each verts_c
    else:
        num_images = cond["pose"].shape[0]
        device = cond["pose"].device

        # uniform[-sigma, sigma]
        sample = torch.rand(num_images, num_pixels, 3).to(device)
        global_sigma = 0.3
        sample = sample * (global_sigma * 2) - global_sigma

    sample.requires_grad_()
    local_pred = implicit_network(sample, cond)[..., 0:1]
    grad_theta = compute_gradient(sample, local_pred)
    return grad_theta


def extract_features(
    deformer, implicit_network, pnts_c, cond, tfs, create_graph=True, retain_graph=True
):
    if pnts_c.shape[0] == 0:
        return pnts_c.detach()
    pnts_c.requires_grad_(True)
    num_images = tfs.shape[0]
    assert len(tfs.shape) == 4
    assert tfs.shape[2] == 4
    assert tfs.shape[3] == 4
    pnts_c = pnts_c.view(num_images, -1, 3)
    pnts_d = deformer.forward_skinning(pnts_c, None, tfs)

    num_dim = pnts_d.shape[-1]
    grads = []
    for i in range(num_dim):
        d_out = torch.zeros_like(pnts_d, requires_grad=False, device=pnts_d.device)
        d_out[:, :, i] = 1
        grad = torch.autograd.grad(
            outputs=pnts_d,
            inputs=pnts_c,
            grad_outputs=d_out,
            # create_graph=create_graph,
            create_graph=False,  # ← Always False to prevent nested graphs
            # retain_graph=True if i < num_dim - 1 else retain_graph
            retain_graph=True,
            only_inputs=True,
        )[0]
        grads.append(grad)

    # Line 32: Stack gradients into Jacobian matrix
    grads = torch.stack(grads, dim=-2).reshape(-1, num_dim, num_dim)

    # ============================================================
    # LINES 33-60: REPLACE ORIGINAL LINE 33 WITH THIS BLOCK
    # FIX: CPU inverse workaround for CUDA cuSPARSE compatibility
    # Issue: RTX 4090 + CUDA 11.1 doesn't support cusparseCreate
    # Solution: Compute Jacobian inverse on CPU, then move to GPU
    # Context: Inverse Jacobian needed for normal transformation
    # ============================================================

    device = grads.device  # Remember original device

    if grads.is_cuda:
        # GPU tensor - move to CPU for inverse computation
        grads_cpu = grads.cpu()

        try:
            # Compute inverse on CPU (stable and supported)
            grads_inv_cpu = grads_cpu.inverse()
        except RuntimeError as e:
            # Fallback 1: Regularization for near-singular matrices
            print(f"[extract_features] Warning: Jacobian inversion failed, applying regularization: {e}")
            epsilon = 1e-6
            batch_size = grads_cpu.shape[0]
            identity = torch.eye(num_dim, device='cpu').unsqueeze(0).expand(batch_size, -1, -1)
            grads_cpu_reg = grads_cpu + epsilon * identity

            try:
                grads_inv_cpu = grads_cpu_reg.inverse()
            except RuntimeError as e2:
                # Fallback 2: Pseudo-inverse (most stable)
                print(f"[extract_features] Warning: Using pseudo-inverse: {e2}")
                grads_inv_cpu = torch.linalg.pinv(grads_cpu)

        # Move result back to GPU
        grads_inv = grads_inv_cpu.to(device)
    else:
        # CPU tensor - compute directly
        try:
            grads_inv = grads.inverse()
        except RuntimeError as e:
            # Fallback with regularization
            print(f"[extract_features] Warning: Jacobian inversion failed, applying regularization: {e}")
            epsilon = 1e-6
            batch_size = grads.shape[0]
            identity = torch.eye(num_dim, device=device).unsqueeze(0).expand(batch_size, -1, -1)
            grads_reg = grads + epsilon * identity

            try:
                grads_inv = grads_reg.inverse()
            except RuntimeError as e2:
                print(f"[extract_features] Warning: Using pseudo-inverse: {e2}")
                grads_inv = torch.linalg.pinv(grads)

    # ============================================================
    # END OF FIX
    # ============================================================

    # Lines 34-58: Rest of function UNCHANGED
    output = implicit_network(pnts_c, cond)

    # [0]
    sdf = output[:, :, :1]

    feature = output[:, :, 1:]
    d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
    gradients = torch.autograd.grad(
        outputs=sdf,
        inputs=pnts_c,
        grad_outputs=d_output,
        create_graph=create_graph,
        retain_graph=retain_graph,
        only_inputs=True,
    )[0]

    gradients = gradients.view(-1, 3)
    # ensure the gradient is normalized
    normals = torch.nn.functional.normalize(
        torch.einsum("bi,bij->bj", gradients, grads_inv), dim=1, eps=1e-6
    )
    grads = grads.reshape(grads.shape[0], -1)
    feature = feature.reshape(-1, feature.shape[2])
    return grads, normals, feature


def render_fg_rgb(
    deformer,
    implicit_network,
    rendering_network,
    points,
    view_dirs,
    cond,
    tfs,
    feature_vectors,
    is_training=True,
    time_code=None,
):
    pnts_c = points

    # features on samples for rendering
    _, normals, feature_vectors = extract_features(
        deformer,
        implicit_network,
        pnts_c,
        cond,
        tfs,
        create_graph=is_training,
        retain_graph=is_training,
    )

    # ================================================================
    # Handle time_code if present
    # ================================================================
    if time_code is not None:

        # ✅ FIX: Ensure time_code is EXACTLY 2D [B, D]
        # Squeeze ALL extra dimensions (handles [B, 1, D] → [B, D])
        while time_code.ndim > 2:
            time_code = time_code.squeeze(1)  # Remove dimension at index 1

        # After squeezing, ensure it's at least 2D
        if time_code.ndim == 1:
            time_code = time_code.unsqueeze(0)  # [D] → [1, D]

        # Now time_code is guaranteed to be [B, D]
        num_images = time_code.shape[0]
        num_samples = pnts_c.shape[0] // num_images

        # ✅ Now safe: time_code is [B, D], can do [:, None, :] → [B, 1, D]
        time_code = (
            time_code[:, None, :]                           # [B, 1, D]
            .repeat(1, num_samples, 1)                      # [B, N, D]
            .reshape(-1, time_code.shape[-1])               # [B*N, D]
        )

        # Concatenate with features
        feature_vectors = torch.cat([feature_vectors, time_code], dim=-1)

        if feature_vectors.shape[-1] != 288:
            print(f"  ⚠️  MISMATCH: {feature_vectors.shape[-1]} != 48")

    else:
        print(f"\n[DIAGNOSTIC] time_code is None - using features as-is")
        print(f"  feature_vectors shape: {feature_vectors.shape}")

    try:
        # Rendering
        fg_rendering_output = rendering_network(
            pnts_c, normals, view_dirs, cond["pose"], feature_vectors
        )
        rgb_vals = fg_rendering_output[:, :3]
        return rgb_vals, normals

    except RuntimeError as e:
        print(f"\n❌ ERROR in rendering network:")
        print(f"  Error: {str(e)}")
        print(f"  feature_vectors shape (last dimension is d_in): {feature_vectors.shape}")
        print(f"  Config d_in should be: {feature_vectors.shape[-1]}")
        raise


def sdf_func_with_deformer(deformer, sdf_fn, training, x, deform_info):
    """SDF function with deformer - handles optional tfs."""
    cond = deform_info["cond"]
    tfs = deform_info.get("tfs", None)
    verts = deform_info.get("verts", None)

    # Handle cond being a dict or tensor
    if isinstance(cond, dict):
        print(f"  cond is dict with keys: {list(cond.keys())}")
        for k, v in cond.items():
            if isinstance(v, torch.Tensor):
                print(f"    cond['{k}'] has_nan: {torch.isnan(v).any().item()}")
    elif isinstance(cond, torch.Tensor):
        print(f"  cond has_nan: {torch.isnan(cond).any().item()}")
    else:
        print(f"  cond type: {type(cond)}")

    if tfs is not None:
        print(f"  tfs has_nan: {torch.isnan(tfs).any().item()}")
        print(f"  tfs has_inf: {torch.isinf(tfs).any().item()}")
    else:
        print(f"  tfs: None (object node)")

    if tfs is not None:
        # Original path: use deformer with tfs
        num_images = tfs.shape[0]
        x = x.view(num_images, -1, 3)
        x_c, outlier_mask = deformer.forward(
            x, tfs, return_weights=False, inverse=True, verts=verts
        )

        # ✅ NEW DEBUG 2: Check deformer output
        print(f"\n[sdf_func_with_deformer] After deformer.forward(inverse=True):")
        print(f"  x_c has_nan: {torch.isnan(x_c).any().item()}")
        if torch.isnan(x_c).any():
            print(f"  ❌ Deformer produced NaN canonical points!")
    else:
        # No tfs available: skip deformation
        x_c = x
        print(f"\n[sdf_func_with_deformer] No tfs, using x directly as x_c")

    # Continue with SDF computation
    output = sdf_fn(x_c, cond)

    sdf = output[:, :, 0:1]
    feature = output[:, :, 1:]

    return sdf, x_c, feature


def compute_mano_cano_sdf(mesh_v_cano, mesh_f_cano, mesh_face_vertices, x_cano):
    distance, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(
        x_cano.contiguous(), mesh_face_vertices
    )

    distance = torch.sqrt(distance)  # kaolin outputs squared distance

    # inside or not
    sign = kaolin.ops.mesh.check_sign(mesh_v_cano, mesh_f_cano, x_cano).float()

    # inside: 1 -> 1 - 2 = -1, negative
    # outside: 0 -> 1 - 0 = 1, positive
    sign = 1 - 2 * sign
    signed_distance = sign * distance  # SDF of points to mesh
    return signed_distance


def check_off_in_surface_points_cano_mesh(
    mesh_v_cano,
    mesh_f_cano,
    mesh_face_vertices,
    x_cano,
    num_pixels_total,
    threshold=0.05,
):
    distance, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(
        x_cano.contiguous(), mesh_face_vertices
    )

    distance = torch.sqrt(distance)  # kaolin outputs squared distance

    # inside or not
    sign = kaolin.ops.mesh.check_sign(mesh_v_cano, mesh_f_cano, x_cano).float()

    # inside: 1 -> 1 - 2 = -1, negative
    # outside: 0 -> 1 - 0 = 1, positive
    sign = 1 - 2 * sign
    signed_distance = sign * distance  # SDF of points to mesh

    # num_rays, samples, 1
    signed_distance = signed_distance.reshape(num_pixels_total, -1, 1)

    minimum = torch.min(signed_distance, 1)[0]
    index_off_surface = (minimum > threshold).squeeze(1)
    index_in_surface = (minimum <= 0.0).squeeze(1)
    return index_off_surface, index_in_surface


def density2weight(density_flat, z_vals, z_max):
    density = density_flat.reshape(-1, z_vals.shape[1])  # (num_rays, N_samples)

    # Distances between samples
    dists = z_vals[:, 1:] - z_vals[:, :-1]              # (num_rays, N_samples-1)
    z_max_dists = (z_max.unsqueeze(-1) - z_vals[:, -1:])# (num_rays, 1)
    dists = torch.cat([dists, z_max_dists], dim=-1)     # (num_rays, N_samples)

    # Ensure dists and density are non-negative and bounded
    dists = torch.clamp(dists, min=0.0)
    density = torch.clamp(density, min=0.0, max=1.0)

    # LOG SPACE
    free_energy = dists * density                       # (num_rays, N_samples)
    # Clamp free_energy to avoid overflow/underflow in exp
    free_energy = torch.clamp(free_energy, min=-50.0, max=50.0)

    alpha = 1.0 - torch.exp(-free_energy)               # probability occupied

    # add zero for CDF next step
    zeros = torch.zeros(dists.shape[0], 1, device=dists.device, dtype=dists.dtype)
    shifted_free_energy = torch.cat([zeros, free_energy], dim=-1)

    cum_fe = torch.cumsum(shifted_free_energy, dim=-1)
    cum_fe = torch.clamp(cum_fe, min=-50.0, max=50.0)

    transmittance = torch.exp(-cum_fe)
    # Clamp and sanitize transmittance
    transmittance = torch.clamp(transmittance, min=0.0, max=1.0)
    transmittance = torch.nan_to_num(transmittance, nan=0.0, posinf=0.0, neginf=0.0)

    fg_transmittance = transmittance[:, :-1]
    bg_weights = transmittance[:, -1]                   # (num_rays,)

    fg_weights = alpha * fg_transmittance               # (num_rays, N_samples)
    fg_weights = torch.nan_to_num(fg_weights, nan=0.0, posinf=0.0, neginf=0.0)
    bg_weights = torch.nan_to_num(bg_weights, nan=0.0, posinf=0.0, neginf=0.0)

    return fg_weights, bg_weights
