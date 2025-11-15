import torch

import src.engine.volsdf_utils as volsdf_utils


def sort_tensor(tensor, indices):
    assert len(tensor.shape) == 3, "tensor must be 3D"
    assert len(indices.shape) == 2, "indices must be 2D"
    num_dim = tensor.shape[-1]
    expanded_indices = indices[:, :, None].repeat(1, 1, num_dim)

    # Sort tensor with expanded_indices
    tensor_sorted = torch.gather(tensor, 1, expanded_indices)

    return tensor_sorted


def integrate(colors, weights):
    assert len(colors.shape) == 3
    assert len(weights.shape) == 2
    rendered_color = torch.sum(colors * weights[:, :, None], dim=1)
    return rendered_color


def render_color(
    deformer,
    implicit_network,
    rendering_network,
    ray_dirs,
    cond,
    tfs,
    canonical_points,
    feature_vectors,
    is_training,
    num_samples,
    class_id,
    time_code,
):
    dirs = ray_dirs.unsqueeze(1).repeat(1, num_samples, 1)  ## view dir
    view = -dirs.reshape(-1, 3)
    canonical_points = canonical_points.reshape(-1, 3)

    # ✅ FIXED DEBUG: Only check what exists before render_fg_rgb
    print(f"[render_color] Before render_fg_rgb:")
    print(f"  canonical_points has_nan: {torch.isnan(canonical_points).any().item()}")
    print(f"  view has_nan: {torch.isnan(view).any().item()}")
    print(f"  feature_vectors has_nan: {torch.isnan(feature_vectors).any().item()}")

    fg_rgb, fg_normal = volsdf_utils.render_fg_rgb(
        deformer,
        implicit_network,
        rendering_network,
        canonical_points,
        view,
        cond,
        tfs,
        feature_vectors=feature_vectors,
        is_training=is_training,
        time_code=time_code,
    )

    print(f"[render_color] After render_fg_rgb:")
    print(f"  fg_rgb has_nan: {torch.isnan(fg_rgb).any().item()}")
    print(f"  fg_normal has_nan: {torch.isnan(fg_normal).any().item()}")

    fg_rgb = fg_rgb.reshape(-1, num_samples, 3)
    fg_normal = fg_normal.reshape(-1, num_samples, 3)

    MAX_CLASS = 4
    semantics = torch.zeros(fg_rgb.shape[0], num_samples, MAX_CLASS).to(fg_rgb.device)
    semantics[:, :, class_id] = 1.0
    return fg_rgb, fg_normal, semantics
