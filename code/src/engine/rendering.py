import torch
from loguru import logger

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
    view_points,          # view-space points for shading
    sdf_query_points,     # canonical points for SDF query
    feature_vectors,
    is_training,
    num_samples,
    class_id,
    time_code,
):
    dirs = ray_dirs.unsqueeze(1).repeat(1, num_samples, 1)  ## view dir
    view = -dirs.reshape(-1, 3)

    # ✅ FIX: Reshape each correctly (not canonical twice!)
    view_points_flat = view_points.reshape(-1, 3)
    sdf_query_flat = sdf_query_points.reshape(-1, 3)

    # DEBUG
    logger.info(f"[render_color] view_points range: [{view_points_flat.min():.3f}, {view_points_flat.max():.3f}]")
    logger.info(f"[render_color] sdf_query_points range: [{sdf_query_flat.min():.3f}, {sdf_query_flat.max():.3f}]")

    # ✅ Query SDF at sdf_query_points (canonical for object, view for hand)
    # but render_fg_rgb needs view_points for position-based calculations
    fg_rgb, fg_normal = volsdf_utils.render_fg_rgb(
        deformer,
        implicit_network,
        rendering_network,
        sdf_query_flat,      # SDF queried here (canonical for object, view for hand)
        view_points_flat,    # ✅ Positions for rendering (view space)
        view,                # view directions
        cond,
        tfs,
        feature_vectors=feature_vectors,
        is_training=is_training,
        time_code=time_code,
    )

    fg_rgb = fg_rgb.reshape(-1, num_samples, 3)
    fg_normal = fg_normal.reshape(-1, num_samples, 3)

    MAX_CLASS = 4
    semantics = torch.zeros(fg_rgb.shape[0], num_samples, MAX_CLASS).to(fg_rgb.device)
    semantics[:, :, class_id] = 1.0
    return fg_rgb, fg_normal, semantics
