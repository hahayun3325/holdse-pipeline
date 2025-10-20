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
            create_graph=create_graph,
            retain_graph=True if i < num_dim - 1 else retain_graph,
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
    # ✅ CRITICAL: ALL lines below MUST be inside this if block!
    # ================================================================
    if time_code is not None:
        # Ensure time_code is at least 2D [B, D]
        if time_code.ndim == 1:
            time_code = time_code.unsqueeze(0)

            num_images = time_code.shape[0]
            num_samples = pnts_c.shape[0] // num_images

            # ✅ MUST BE INDENTED - inside the if block!
            time_code = (
                time_code[:, None, :]            # ← 4 spaces indent
                .repeat(1, num_samples, 1)       # ← 4 spaces indent
                .reshape(-1, time_code.shape[-1])# ← 4 spaces indent
            )                                     # ← 4 spaces indent
            feature_vectors = torch.cat([feature_vectors, time_code], dim=-1)  # ← 4 spaces indent

    # Rendering
    fg_rendering_output = rendering_network(
        pnts_c, normals, view_dirs, cond["pose"], feature_vectors
    )
    rgb_vals = fg_rendering_output[:, :3]
    return rgb_vals, normals


def sdf_func_with_deformer(deformer, sdf_fn, training, x, deform_info):
    """SDF function with deformer - handles optional tfs."""
    cond = deform_info["cond"]

    # ================================================================
    # ✅ FIX: Make tfs optional (ObjectServer doesn't provide it)
    # ================================================================
    tfs = deform_info.get("tfs", None)  # Use .get() instead of direct access

    if "verts" in deform_info:
        verts = deform_info["verts"]
    else:
        verts = None

    # ================================================================
    # ✅ FIX: Handle case when tfs is None
    # ================================================================
    if tfs is not None:
        # Original path: use deformer with tfs
        num_images = tfs.shape[0]
        x = x.view(num_images, -1, 3)
        x_c, outlier_mask = deformer.forward(
            x, tfs, return_weights=False, inverse=True, verts=verts
        )
    else:
        # No tfs available: skip deformation (use x directly as x_c)
        x_c = x
        # outlier_mask not needed

    # Continue with SDF computation
    output = sdf_fn(x_c, cond)
    sdf = output[:, :, 0:1]
    feature = output[:, :, 1:]
    # if not training:
    #     sdf[outlier_mask] = 4. # set a large SDF value for outlier points
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
    density = density_flat.reshape(
        -1, z_vals.shape[1]
    )  # (batch_size * num_pixels) x N_samples

    # included also the dist from the sphere intersection
    dists = z_vals[:, 1:] - z_vals[:, :-1]  # (num_rays, num_samples - 1)
    z_max_dists = (
        z_max.unsqueeze(-1) - z_vals[:, -1:]
    )  # (num_rays, 1), the last internval
    dists = torch.cat([dists, z_max_dists], -1)  # (num_rays, num_samples)

    # LOG SPACE
    free_energy = dists * density
    alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here

    # add zero for CDF next step
    shifted_free_energy = torch.cat(
        [torch.zeros(dists.shape[0], 1).cuda(), free_energy], dim=-1
    )  # add 0 for transperancy 1 at t_0
    transmittance = torch.exp(
        -torch.cumsum(shifted_free_energy, dim=-1)
    )  # probability of everything is empty up to now

    # fg transimittance: the first N-1 transittance
    fg_transmittance = transmittance[:, :-1]
    bg_weights = transmittance[
        :, -1
    ]  # factor to be multiplied with the bg volume rendering

    fg_weights = alpha * fg_transmittance  # probability of the ray hits something here
    return fg_weights, bg_weights
