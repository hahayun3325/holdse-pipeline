import torch
import torch.nn.functional as F


def normal_consistency_loss(sdf_grid):
    """Encourages smooth surfaces by penalizing SDF gradient variance."""
    # Compute gradients in 3D
    normals_x = sdf_grid[1:] - sdf_grid[:-1]
    normals_y = sdf_grid[:, 1:] - sdf_grid[:, :-1]
    normals_z = sdf_grid[:, :, 1:] - sdf_grid[:, :, :-1]

    # Return average variance
    return (normals_x.var() + normals_y.var() + normals_z.var()) / 3.0


def depth_smoothness_loss(depth, image):
    """Smooth depth where image is smooth (preserve edges)."""
    # Ensure 2D inputs [H, W] or [B, H, W]
    if depth.dim() == 2:
        depth = depth.unsqueeze(0)  # [H, W] -> [1, H, W]
    if image.dim() == 3 and image.shape[-1] == 3:
        image = image.permute(2, 0, 1).unsqueeze(0)  # [H, W, 3] -> [1, 3, H, W]
    elif image.dim() == 3:
        image = image.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]

    # Compute spatial gradients
    depth_grad_x = torch.abs(depth[:, :, 1:] - depth[:, :, :-1])
    depth_grad_y = torch.abs(depth[:, 1:, :] - depth[:, :-1, :])

    # Image gradients (for edge detection)
    if image.shape[1] == 3:  # RGB
        image_gray = image.mean(dim=1, keepdim=True)
    else:
        image_gray = image

    image_grad_x = torch.abs(image_gray[:, :, 1:] - image_gray[:, :, :-1])
    image_grad_y = torch.abs(image_gray[:, 1:, :] - image_gray[:, :-1, :])

    # Edge-aware weighting
    weight_x = torch.exp(-10 * image_grad_x)
    weight_y = torch.exp(-10 * image_grad_y)

    # Weighted smoothness
    loss_x = (depth_grad_x * weight_x).mean()
    loss_y = (depth_grad_y * weight_y).mean()

    return loss_x + loss_y
