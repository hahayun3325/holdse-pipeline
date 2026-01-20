import torch.nn as nn
import torch
import torch.nn.functional as F
import common.torch_utils as torch_utils
from src.model.mano.server import MANOServer

eps = 1e-6
l1_loss = nn.L1Loss(reduction="none")
l2_loss = nn.MSELoss(reduction="none")
from src.utils.const import SEGM_IDS


# L1 reconstruction loss for RGB values
def get_rgb_loss(rgb_values, rgb_gt, valid_pix, scores):
    """
    L1 reconstruction loss for RGB values with robust score broadcasting.
    Args:
        rgb_values: Predicted RGB, shape [N, 3] or [B*N, 3]
        rgb_gt:     Ground-truth RGB, same shape as rgb_values
        valid_pix:  Boolean or {0,1} mask over pixels, shape [N]
        scores:     Per-entity scores (e.g., per-frame or per-sample), any shape
    Returns:
        Scalar RGB loss.
    """
    # Clamp RGB to [0, 1] range (fix for loss > 1.0 bug)
    rgb_values = torch.clamp(rgb_values, 0, 1)

    # Base L1 loss with per-pixel validity mask
    # l1_loss is reduction="none", so this is [N, 3]
    rgb_loss = l1_loss(rgb_values, rgb_gt) * valid_pix[:, None]

    # Flatten scores to 1D for consistent handling
    scores_flat = scores.reshape(-1)  # shape [B]
    num_scores = scores_flat.shape[0]
    num_pix_total = rgb_loss.shape[0]

    if num_scores <= 0:
        # No scores: just return masked L1
        return (rgb_loss.sum() / (valid_pix.sum() + 1e-6))

    # Ideal case: each score applies to an equal block of pixels
    # i.e. N == num_scores * num_pix_per_score
    if num_pix_total % num_scores != 0:
        # Non-divisible: truncate to the largest multiple to avoid shape mismatch
        num_pix_per_score = num_pix_total // num_scores
        new_N = num_pix_per_score * num_scores
        # Only log once per run ideally, but for now warn each time
        print(
            f"[get_rgb_loss] ⚠️ Size mismatch: N={num_pix_total}, "
            f"scores={num_scores}. Truncating to N={new_N}."
        )
        rgb_loss = rgb_loss[:new_N]
        valid_pix = valid_pix[:new_N]
        num_pix_total = new_N

    # Now N is divisible by num_scores
    num_pix_per_score = num_pix_total // num_scores
    # Expand scores so each score is repeated over its pixel block
    scores_expanded = scores_flat[:, None].repeat(1, num_pix_per_score).view(-1, 1)  # [N, 1]

    # Apply scores
    rgb_loss = rgb_loss * scores_expanded  # [N, 3] * [N, 1]

    # Normalize by number of valid pixels
    rgb_loss = rgb_loss.sum() / (valid_pix.sum() + 1e-6)
    print(f"[get_rgb_loss DEBUG] rgb_loss.sum()={rgb_loss.sum().item():.6f}, valid_pix.sum()={valid_pix.sum().item():.6f}")
    return rgb_loss


# Eikonal loss introduced in IGR
def get_eikonal_loss(grad_theta):
    # ✅ Handle None case when grad_theta unavailable
    if grad_theta is None:
        # Return zero loss with no gradient
        return torch.tensor(0.0, requires_grad=False)

    eikonal_loss = ((grad_theta.norm(2, dim=-1) - 1) ** 2).mean()
    return eikonal_loss


def get_smoothness_loss(vertices):
    """
    Penalize rough/noisy object surfaces.
    Encourages smooth vertex positions (reduces variance).

    Args:
        vertices: [N, 3] tensor of vertex positions

    Returns:
        Scalar smoothness loss
    """
    if vertices is None:
        return torch.tensor(0.0, requires_grad=False)

    # Simple smoothness: minimize variance of positions
    # Lower variance = vertices are closer together = smoother surface
    smoothness = torch.var(vertices)
    return smoothness

# BCE loss for clear boundary
def get_bce_loss(acc_map, scores):
    bce_loss = (
        acc_map * (acc_map + eps).log() + (1 - acc_map) * (1 - acc_map + eps).log()
    )
    num_pix = bce_loss.shape[0] // scores.shape[0]
    scores = scores[:, None].repeat(1, num_pix).view(-1, 1)
    bce_loss = bce_loss * scores

    binary_loss = -1 * (bce_loss).mean() * 2
    return binary_loss


# Global opacity sparseness regularization
def get_opacity_sparse_loss(acc_map, index_off_surface, scores):
    """
    Compute opacity sparseness loss for off-surface regions.

    Args:
        acc_map: Accumulated opacity map [N]
        index_off_surface: Boolean mask for off-surface pixels [N]
        scores: Per-entity scores (any shape, will be flattened to 1D)

    Returns:
        opacity_sparse_loss: Scalar loss value
    """
    # Compute base sparseness loss
    opacity_sparse_loss = l1_loss(
        acc_map[index_off_surface],
        torch.zeros_like(acc_map[index_off_surface])
    )

    # ================================================================
    # FIX: Robustly flatten scores to 1D
    # ================================================================
    # Original shape for debugging
    original_shape = scores.shape

    # Flatten to 1D (handles any input shape)
    scores_flat = scores.reshape(-1)  # Most robust: reshape to 1D

    # Validate: ensure we have the right batch size
    num_pix = acc_map.shape[0] // scores_flat.shape[0]

    if acc_map.shape[0] % scores_flat.shape[0] != 0:
        # Mismatch: acc_map size not divisible by scores size
        # This is a warning condition but continue
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            f"[get_opacity_sparse_loss] Pixel/score mismatch: "
            f"acc_map={acc_map.shape[0]}, scores={scores_flat.shape[0]} (from {original_shape}). "
            f"Adjusting num_pix calculation."
        )
        # Use maximum possible num_pix
        num_pix = max(1, acc_map.shape[0] // scores_flat.shape[0])

    # Expand scores to match pixel count
    scores_expanded = scores_flat[:, None].repeat(1, num_pix).view(-1, 1)

    # Trim if necessary (in case of rounding issues)
    if scores_expanded.shape[0] > acc_map.shape[0]:
        scores_expanded = scores_expanded[:acc_map.shape[0]]

    # Select off-surface scores
    scores_selected = scores_expanded[index_off_surface]

    # Weight loss by scores
    opacity_sparse_loss = opacity_sparse_loss * scores_selected

    # Mean loss
    opacity_sparse_loss = opacity_sparse_loss.mean()

    return opacity_sparse_loss


def get_mask_loss(mask_prob, mask_gt, valid_pix):
    assert torch.all((mask_gt == 0) | (mask_gt == 1))
    # bce loss on mask
    mask_loss = F.binary_cross_entropy(
        mask_prob, mask_gt[:, None].float(), reduction="none"
    )
    mask_loss = mask_loss * valid_pix[:, None]
    mask_loss = mask_loss.sum() / (valid_pix.sum() + 1e-6)
    return mask_loss


def get_sem_loss(sem_pred, mask_gt, valid_pix, scores, edge_weight=0.0):
    """
    Semantic segmentation loss with optional edge weighting.

    Supports two formats:
    1. Full image: sem_pred shape (B, H, W, 4) or (B, 4, H, W)
    2. Sampled pixels: sem_pred shape (N, 4) where N = batch_size

    Args:
        sem_pred: Predicted segmentation
        mask_gt: Ground truth mask
        valid_pix: Valid pixel mask
        scores: Per-image weights
        edge_weight: Weight for edge-aware loss (default 0.0 = disabled)

    Returns:
        loss: Scalar semantic loss
    """
    # Detect format
    print(f"[DEBUG get_sem_loss] sem_pred shape: {sem_pred.shape}")
    print(f"[DEBUG get_sem_loss] mask_gt shape: {mask_gt.shape}")
    print(f"[DEBUG get_sem_loss] valid_pix shape: {valid_pix.shape}")

    is_sampled_pixels = (sem_pred.ndim == 2)  # (N, 4) format
    is_full_image = (sem_pred.ndim == 4)  # (B, H, W, 4) or (B, 4, H, W)

    if not (is_sampled_pixels or is_full_image):
        raise ValueError(f"Unsupported sem_pred shape: {sem_pred.shape}")

    # ================================================================
    # PROCESS GROUND TRUTH LABELS (same for both formats)
    # ================================================================
    semantic_gt = mask_gt.clone()

    # Existing segmentation band logic
    bnd_bg = semantic_gt < 25
    bnd_o = torch.logical_and(25 <= semantic_gt, semantic_gt < 100)
    bnd_r = torch.logical_and(100 <= semantic_gt, semantic_gt < 200)
    bnd_l = 200 <= semantic_gt

    # bandaid fix for aliasing
    semantic_gt[bnd_bg] = SEGM_IDS["bg"]
    semantic_gt[bnd_o] = SEGM_IDS["object"]
    semantic_gt[bnd_r] = SEGM_IDS["right"]
    semantic_gt[bnd_l] = SEGM_IDS["left"]

    # remap to 0,1,2,3
    semantic_gt[semantic_gt == SEGM_IDS["bg"]] = 0
    semantic_gt[semantic_gt == SEGM_IDS["object"]] = 1
    semantic_gt[semantic_gt == SEGM_IDS["right"]] = 2
    semantic_gt[semantic_gt == SEGM_IDS["left"]] = 3

    # ================================================================
    # CASE 1: SAMPLED PIXELS (N, 4)
    # ================================================================
    if is_sampled_pixels:
        print(f"[DEBUG] Using sampled pixel loss (no edge weighting)")

        # Flatten GT to match sampled format
        if semantic_gt.ndim > 1:
            semantic_gt_flat = semantic_gt.flatten()
        else:
            semantic_gt_flat = semantic_gt

        if valid_pix.ndim > 1:
            valid_pix_flat = valid_pix.flatten()
        else:
            valid_pix_flat = valid_pix

        # Standard cross-entropy loss
        sem_loss = F.cross_entropy(
            sem_pred,  # (N, 4) - logits
            semantic_gt_flat.long(),  # (N,) - class indices
            reduction="none",
        )

        # Apply valid pixel mask
        sem_loss = sem_loss * valid_pix_flat

        # Average over valid pixels
        sem_loss = sem_loss.sum() / (valid_pix_flat.sum() + 1e-6)

        # Note: edge_weight is ignored for sampled pixels (can't detect edges)
        if edge_weight > 0.0:
            print(f"[WARN] edge_weight={edge_weight} ignored for sampled pixel format")

        return sem_loss

    # ================================================================
    # CASE 2: FULL IMAGE (B, H, W, 4) or (B, 4, H, W)
    # ================================================================
    elif is_full_image:
        print(f"[DEBUG] Using full image loss (edge weighting={edge_weight})")

        # Normalize to (B, C, H, W) format
        if sem_pred.shape[1] == 4:
            # Already (B, 4, H, W)
            sem_pred_formatted = sem_pred
        elif sem_pred.shape[3] == 4:
            # (B, H, W, 4) - need permute
            sem_pred_formatted = sem_pred.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Unexpected full image sem_pred shape: {sem_pred.shape}")

        # Standard semantic loss
        sem_loss_standard = F.cross_entropy(
            sem_pred_formatted,  # (B, 4, H, W)
            semantic_gt.long(),  # (B, H, W)
            reduction="none",
        )

        # Apply valid pixel mask and image scores
        sem_loss_standard = sem_loss_standard * valid_pix
        sem_loss_standard = (sem_loss_standard.sum((1, 2)) / (valid_pix.sum((1, 2)) + 1e-6)) * scores
        sem_loss_standard = sem_loss_standard.mean()

        # ================================================================
        # EDGE-AWARE LOSS (only for full images)
        # ================================================================
        if edge_weight > 0.0:
            try:
                import kornia.filters

                # Detect edges in ground truth mask
                mask_for_edges = semantic_gt.float().unsqueeze(1)  # (B, 1, H, W)

                # Sobel edge detection
                edges = kornia.filters.sobel(mask_for_edges)  # (B, 1, H, W)
                edges = edges.squeeze(1)  # (B, H, W)

                # Threshold to binary edge mask
                edge_mask = (edges > 0.1).float()

                # Edge-weighted loss
                sem_loss_edge = F.cross_entropy(
                    sem_pred_formatted,
                    semantic_gt.long(),
                    reduction="none",
                )

                # Apply edge mask and valid pixels
                sem_loss_edge = sem_loss_edge * edge_mask * valid_pix

                # Normalize by edge pixel count
                edge_count = (edge_mask * valid_pix).sum((1, 2)) + 1e-6
                sem_loss_edge = (sem_loss_edge.sum((1, 2)) / edge_count) * scores
                sem_loss_edge = sem_loss_edge.mean()

                # Combine standard + edge-weighted
                total_loss = sem_loss_standard + edge_weight * sem_loss_edge

                return total_loss

            except ImportError:
                print("WARNING: kornia not installed, edge_weight ignored")
                return sem_loss_standard
            except Exception as e:
                print(f"WARNING: Edge detection failed: {e}, using standard loss")
                return sem_loss_standard
        else:
            return sem_loss_standard
def get_mano_cano_loss(pred_sdf, gt_sdf, limit, scores):
    pred_sdf = torch.clamp(pred_sdf, -limit, limit)
    gt_sdf = torch.clamp(gt_sdf, -limit, limit)
    mano_cano_loss = l1_loss(pred_sdf, gt_sdf)

    scores = scores[:, None]

    mano_cano_loss = mano_cano_loss * scores

    mano_cano_loss = mano_cano_loss.mean()
    return mano_cano_loss


# ============================================================================
# Phase 2: Simple SDS Loss Getter (Alternative)
# ============================================================================

def get_sds_loss(object_node, hand_pose, category, hold_loss_module, iteration):
    """
    Wrapper to compute SDS loss for Phase 2 integration.

    Args:
        object_node: HOLD's ObjectNode
        hand_pose: [B, 48] MANO parameters
        category: str - object category
        hold_loss_module: HOLDLoss instance from loss.py
        iteration: int - current training iteration

    Returns:
        sds_loss: scalar tensor
        sds_info: dict with auxiliary info
    """
    loss_sds, sds_info = hold_loss_module.compute_sds_loss(
        object_node=object_node,
        hand_pose=hand_pose,
        category=category,
        iteration=iteration
    )
    return loss_sds, sds_info


def get_joint_supervision_loss(model_outputs, batch, hand_node_name='right'):
    """Supervise predicted joints with ground truth MANO parameters.

    Args:
        model_outputs: Dict with predicted joint positions from model
        batch: Dict with GT MANO parameters
        hand_node_name: 'right' or 'left'

    Returns:
        Joint position loss (L2 distance in mm)
    """
    # Get predicted joints from model forward pass
    pred_key = f'j3d.{hand_node_name}'
    if pred_key not in model_outputs:
        return torch.tensor(0.0)

    pred_joints = model_outputs[pred_key]  # [B, 21, 3]

    # Check if GT MANO params exist in batch
    gt_pose_key = f'gt.{hand_node_name}.hand_pose'
    gt_trans_key = f'gt.{hand_node_name}.hand_trans'
    gt_shape_key = f'gt.{hand_node_name}.hand_shape'

    if not all(k in batch for k in [gt_pose_key, gt_trans_key, gt_shape_key]):
        return torch.tensor(0.0, device=pred_joints.device)

    # Get GT MANO parameters from batch
    gt_pose = batch[gt_pose_key]      # [B, 48]
    gt_trans = batch[gt_trans_key]    # [B, 3]
    gt_shape = batch[gt_shape_key]    # [B, 10] or [10]

    # Convert to torch tensors if needed
    if not isinstance(gt_pose, torch.Tensor):
        gt_pose = torch.from_numpy(gt_pose).to(pred_joints.device)
    if not isinstance(gt_trans, torch.Tensor):
        gt_trans = torch.from_numpy(gt_trans).to(pred_joints.device)
    if not isinstance(gt_shape, torch.Tensor):
        gt_shape = torch.from_numpy(gt_shape).to(pred_joints.device)

    # Ensure gt_shape is batched
    if gt_shape.ndim == 1:
        gt_shape = gt_shape.unsqueeze(0).expand(gt_pose.shape[0], -1)

    # Import MANO layer (assuming it's available in model_outputs context)
    # We need to compute GT joints by running GT params through MANO
    # For now, return a placeholder until we wire up MANO access

    # TODO: Get MANO layer from model and compute GT joints
    # gt_joints = mano_layer(gt_pose, gt_shape, gt_trans)
    # loss = torch.nn.functional.mse_loss(pred_joints, gt_joints)
    # return loss * 1000.0  # Scale to mm

    return torch.tensor(0.0, device=pred_joints.device)

def get_joint_supervision_loss_v2(model_outputs, batch, hand_node_name='right'):
    """Supervise predicted joints with pre-computed GT joints.

    Args:
        model_outputs: Dict with predicted joint positions
        batch: Dict with pre-computed GT joint positions
        hand_node_name: 'right' or 'left'

    Returns:
        Joint position loss (L2 distance in mm)
    """
    # Get predicted joints
    pred_key = f'j3d.{hand_node_name}'
    if pred_key not in model_outputs:
        return torch.tensor(0.0)

    pred_joints = model_outputs[pred_key]  # [B, 21, 3]

    # Get GT joints (pre-computed in training loop)
    gt_key = f'gt.j3d.{hand_node_name}'
    if gt_key not in batch:
        return torch.tensor(0.0, device=pred_joints.device)

    gt_joints = batch[gt_key]  # [B, 21, 3]

    # Ensure same shape
    if pred_joints.shape != gt_joints.shape:
        return torch.tensor(0.0, device=pred_joints.device)

    # L2 loss on joint positions (in meters)
    loss = torch.nn.functional.mse_loss(pred_joints, gt_joints)

    # Scale to mm for interpretability
    return loss * 1000.0