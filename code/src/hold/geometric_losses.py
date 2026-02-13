import torch
import torch.nn.functional as F


def normal_consistency_loss(sdf_grid: torch.Tensor) -> torch.Tensor:
    """
    Encourage smooth SDF normals by penalizing gradient variation.

    Accepts:
      - [B, 1, D, H, W]  (preferred: volumetric SDF grid)
      - [B, D, H, W]
      - [D, H, W]
    Always treats the last three dims as spatial (D, H, W).
    """
    # Normalize shape to [B, 1, D, H, W]
    if sdf_grid.dim() == 5:
        # e.g., [B, 1, D, H, W] or [B, C, D, H, W]
        sdf = sdf_grid
    elif sdf_grid.dim() == 4:
        # assume [B, D, H, W] → add channel dim
        sdf = sdf_grid.unsqueeze(1)
    elif sdf_grid.dim() == 3:
        # [D, H, W] → add batch and channel dims
        sdf = sdf_grid.unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError(f"normal_consistency_loss: unsupported sdf_grid shape {sdf_grid.shape}")

    # Finite-difference gradients along D, H, W
    grad_z = sdf[:, :, 1:, :, :] - sdf[:, :, :-1, :, :]   # [B, 1, D-1, H,   W  ]
    grad_y = sdf[:, :, :, 1:, :] - sdf[:, :, :, :-1, :]   # [B, 1, D,   H-1, W  ]
    grad_x = sdf[:, :, :, :, 1:] - sdf[:, :, :, :, :-1]   # [B, 1, D,   H,   W-1]

    # Simple consistency: penalize gradient energy (smooth SDF → stable normals)
    loss = (
        grad_x.pow(2).mean() +
        grad_y.pow(2).mean() +
        grad_z.pow(2).mean()
    ) / 3.0

    return loss


def depth_smoothness_loss(depth, image):
    """Smooth depth where image is smooth (preserve edges)."""

    # -----------------------------
    # 1) Normalize depth to [B, 1, H_d, W_d]
    # -----------------------------
    if depth.dim() == 2:
        # [H, W] -> [1, 1, H, W]
        depth = depth.unsqueeze(0).unsqueeze(0)
    elif depth.dim() == 3:
        # [B, H, W] -> [B, 1, H, W]
        depth = depth.unsqueeze(1)
    elif depth.dim() == 4:
        # [B, C, H, W] or [B, 1, H, W] -> [B, 1, H, W]
        if depth.shape  != 1:
            depth = depth.mean(dim=1, keepdim=True)

    H_d, W_d = depth.shape[-2:]

    # -----------------------------
    # 2) Normalize image to [B, C, H, W]
    # -----------------------------
    if image.dim() == 2:
        # [H, W] -> [1, 1, H, W]
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.dim() == 3:
        if image.shape[-1] == 3:
            # [H, W, 3] -> [1, 3, H, W]
            image = image.permute(2, 0, 1).unsqueeze(0)
        else:
            # [C, H, W] -> [1, C, H, W]
            image = image.unsqueeze(0)
    elif image.dim() == 4 and image.shape[-1] in (1, 3):
        # [B, H, W, 3] or [B, H, W, 1] -> [B, C, H, W]
        image = image.permute(0, 3, 1, 2)

    # Match batch size if one of them is single-view
    if image.shape[0] != depth.shape[0]:
        if image.shape[0] == 1:
            image = image.expand(depth.shape[0], -1, -1, -1)
        elif depth.shape[0] == 1:
            depth = depth.expand(image.shape[0], -1, -1, -1)

    # -----------------------------
    # 3) Resize image to depth resolution
    # -----------------------------
    if image.shape[-2:] != (H_d, W_d):
        image = F.interpolate(
            image, size=(H_d, W_d), mode="bilinear", align_corners=False
        )

    # -----------------------------
    # 4) Compute spatial gradients
    # -----------------------------
    # depth: [B, 1, H_d, W_d] -> [B, H_d, W_d] for finite differences
    depth_2d = depth.squeeze(1)  # [B, H_d, W_d]
    depth_grad_x = torch.abs(depth_2d[:, :, 1:] - depth_2d[:, :, :-1])  # [B, H_d, W_d-1]
    depth_grad_y = torch.abs(depth_2d[:, 1:, :] - depth_2d[:, :-1, :])  # [B, H_d-1, W_d]

    # Image gradients (for edge detection)
    if image.shape  == 3:  # RGB
        image_gray = image.mean(dim=1, keepdim=True)  # [B, 1, H_d, W_d]
    else:
        image_gray = image  # assume [B, 1, H_d, W_d] or similar

    image_grad_x = torch.abs(
        image_gray[:, :, :, 1:] - image_gray[:, :, :, :-1]
    )  # [B, 1, H_d, W_d-1]
    image_grad_y = torch.abs(
        image_gray[:, :, 1:, :] - image_gray[:, :, :-1, :]
    )  # [B, 1, H_d-1, W_d]

    # -----------------------------
    # 5) Edge-aware weighting and loss
    # -----------------------------
    weight_x = torch.exp(-10 * image_grad_x)
    weight_y = torch.exp(-10 * image_grad_y)

    # Broadcast depth gradients to match weight shape
    loss_x = (depth_grad_x.unsqueeze(1) * weight_x).mean()
    loss_y = (depth_grad_y.unsqueeze(1) * weight_y).mean()

    return loss_x + loss_y
