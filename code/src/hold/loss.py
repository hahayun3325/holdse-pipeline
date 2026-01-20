import torch
from torch import nn
from PIL import Image

import src.hold.loss_terms as loss_terms

# ====================================================================
# PHASE 4: Contact Refinement Imports
# ====================================================================
try:
    from src.model.ghop.contact_refinement import GHOPContactRefinement
    GHOP_CONTACT_AVAILABLE = True
except ImportError:
    GHOP_CONTACT_AVAILABLE = False

from loguru import logger
# ====================================================================


class Loss(nn.Module):
    """Standard HOLD loss module for reconstruction.

    Computes:
    - RGB reconstruction loss
    - Semantic segmentation loss
    - Eikonal regularization
    - MANO canonical SDF loss
    - Opacity sparsity loss
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.milestone = 30000
        self.im_w = None
        self.im_h = None

        # DEBUG: Print config structure
        print(f"[DEBUG INIT] args type: {type(args)}")
        print(f"[DEBUG INIT] args has __dict__: {hasattr(args, '__dict__')}")

        if hasattr(args, '__dict__'):
            print(f"[DEBUG INIT] args keys: {list(vars(args).keys())[:20]}")  # First 20 keys

        # Check for w_mask_edge at top level
        print(f"[DEBUG INIT] hasattr(args, 'w_mask_edge'): {hasattr(args, 'w_mask_edge')}")

        # Check for nested loss config (SAFE)
        has_loss_attr = hasattr(args, 'loss')
        print(f"[DEBUG INIT] hasattr(args, 'loss'): {has_loss_attr}")

        if has_loss_attr:
            try:
                loss_keys = list(vars(args.loss).keys()) if hasattr(args.loss, '__dict__') else []
                print(f"[DEBUG INIT] args.loss keys: {loss_keys[:20]}")
                print(f"[DEBUG INIT] hasattr(args.loss, 'w_mask_edge'): {hasattr(args.loss, 'w_mask_edge')}")
            except Exception as e:
                print(f"[DEBUG INIT] Error accessing args.loss: {e}")

        # Extract edge weight from config (SAFE)
        try:
            if hasattr(args, 'w_mask_edge'):
                self.edge_weight = args.w_mask_edge
                print(f"[DEBUG INIT] Found w_mask_edge (flat): {self.edge_weight}")
            elif hasattr(args, 'loss') and hasattr(args.loss, 'w_mask_edge'):
                self.edge_weight = args.loss.w_mask_edge
                print(f"[DEBUG INIT] Found w_mask_edge (nested): {self.edge_weight}")
            else:
                self.edge_weight = 0.1
                print(f"[WARN] w_mask_edge not found in config, using default: {self.edge_weight}")
        except Exception as e:
            print(f"[ERROR] Failed to read w_mask_edge: {e}")
            self.edge_weight = 0.1

    def forward(self, batch, model_outputs):
        """Compute standard HOLD losses.

        Args:
            batch: Input batch with GT data
            model_outputs: Model predictions

        Returns:
            loss_dict: Dictionary with loss components and total loss
        """
        device = batch["idx"].device

        # Equal scoring for now
        idx = batch["idx"]
        # Ensure idx is 1D for image_scores shape
        if idx.dim() > 1:
            idx = idx.squeeze()
        if idx.dim() == 0:
            idx = idx.unsqueeze(0)

        image_scores = torch.ones(idx.shape[0]).float().to(device)

        # ✅ ADD HERE (after line 95):
        # Extract training progress for weight scheduling
        progress = min(self.milestone, model_outputs.get("step", 0))

        # Get image dimensions
        if self.im_w is None:
            if "im_path" in batch:
                # HOLD dataset: use image path
                im = Image.open(batch["im_path"][0][0])
                self.im_w = im.size[0]
                self.im_h = im.size[1]
            elif "rgb" in batch:
                # GHOP dataset: use rgb field shape
                rgb_shape = batch["rgb"].shape
                if len(rgb_shape) == 4:  # [B, H, W, 3]
                    self.im_h = rgb_shape[1]
                    self.im_w = rgb_shape[2]
                elif len(rgb_shape) == 3:  # [B, N, 3]
                    # Assume square image
                    import math
                    num_pixels = rgb_shape[1]
                    side = int(math.sqrt(num_pixels))
                    self.im_w = side
                    self.im_h = side
                else:
                    # Fallback
                    self.im_w = 640
                    self.im_h = 480
            else:
                # Fallback to default
                self.im_w = 640
                self.im_h = 480

        # ================================================================
        # ✅ FIX: Extract ground truth from correct fields
        # ================================================================
        # RGB: try multiple field names
        if "gt.rgb" in batch:
            rgb_gt = batch["gt.rgb"].view(-1, 3).cuda()
        elif "rgb" in batch:
            rgb_gt = batch["rgb"].view(-1, 3).cuda()
        else:
            # No RGB ground truth - skip RGB loss
            rgb_gt = None

        # Mask: try multiple field names
        if "gt.mask" in batch:
            mask_gt = batch["gt.mask"].view(-1)
        elif "mask" in batch:
            mask_gt = batch["mask"].view(-1)
        else:
            # No mask ground truth - create dummy
            if rgb_gt is not None:
                mask_gt = torch.ones(rgb_gt.shape[0], device=device)
            else:
                mask_gt = torch.ones(self.im_w * self.im_h, device=device)

        # ✅ FIX: Ensure mask_gt is long for one-hot encoding
        mask_gt = mask_gt.long()

        valid_pix = torch.ones_like(mask_gt).float()

        # ================================================================
        # ✅ FIX: Compute losses only if ground truth exists
        # ================================================================
        loss_dict = {}

        # RGB reconstruction loss
        if rgb_gt is not None and "rgb" in model_outputs:
            nan_filter = ~torch.any(model_outputs["rgb"].isnan(), dim=1)

            # ✅ NEW DEBUG LOGGING
            print(f"[RGB LOSS DEBUG] Before nan_filter: rgb_pred shape={model_outputs['rgb'].shape}, rgb_gt shape={rgb_gt.shape}")
            print(f"[RGB LOSS DEBUG] nan_filter sum={nan_filter.sum().item()} out of {nan_filter.shape[0]}")
            print(f"[RGB LOSS DEBUG] valid_pix sum={valid_pix[nan_filter].sum().item()}")

            rgb_loss = loss_terms.get_rgb_loss(
                model_outputs["rgb"][nan_filter],
                rgb_gt[nan_filter],
                valid_pix[nan_filter],
                image_scores,
            )
            loss_dict["loss/rgb"] = rgb_loss * 1.0
            print(f"[DEBUG RGB LOSS] rgb_loss value: {rgb_loss.item():.6f}")

        # Semantic segmentation loss
        if "semantics" in model_outputs:
            sem_pred = model_outputs["semantics"]  # [N, 4] where N = num pixels

            # ✅ FIX: Flatten mask_gt to match sem_pred
            if "gt.mask" in batch:
                mask_gt_sem = batch["gt.mask"].view(-1).long()  # Flatten [2, 128] → [256]
            elif "mask" in batch:
                mask_gt_sem = batch["mask"].view(-1).long()
            else:
                # Fallback: create dummy mask
                mask_gt_sem = torch.zeros(sem_pred.shape[0], device=device).long()

            # ✅ VALIDATION: Ensure shapes match
            assert mask_gt_sem.shape[0] == sem_pred.shape[0], \
                f"Semantic loss shape mismatch: mask {mask_gt_sem.shape} vs pred {sem_pred.shape}"

            sem_loss = loss_terms.get_sem_loss(
                sem_pred,
                mask_gt_sem,           # ← Now [256] matches sem_pred
                valid_pix[:sem_pred.shape[0]],  # Match length
                image_scores,
                edge_weight=self.edge_weight,
            )

            # Standard semantic weight scheduling
            w_sem = torch.linspace(1.1, 0.1, self.milestone + 1)[progress]

            # Apply weight and store
            loss_dict["loss/sem"] = sem_loss * w_sem

            # ✅ LOG: Confirm semantic loss is being computed
            print(f"[DEBUG SEM LOSS] Value: {sem_loss.item():.6f}, Weight: {w_sem:.6f}, Weighted: {(sem_loss * w_sem).item():.6f}")

        # Opacity sparse loss
        opacity_sparse_loss = 0.0
        for key in model_outputs.search("index_off_surface").keys():
            node_id = key.split(".")[0]
            opacity_sparse_loss += loss_terms.get_opacity_sparse_loss(
                model_outputs[f"{node_id}.mask_prob"],
                model_outputs[f"{node_id}.index_off_surface"],
                image_scores,
            )

        # Add opacity sparse loss if we have it
        if opacity_sparse_loss > 0:
            progress = min(self.milestone, model_outputs.get("step", 0))
            w_sparse = torch.linspace(0.0, 1.0, self.milestone + 1)[progress]
            loss_dict["loss/opacity_sparse"] = opacity_sparse_loss * w_sparse

        # ================================================================
        # ✅ FIX: Eikonal loss with None handling
        # ================================================================
        eikonal_loss = 0.0
        for key in model_outputs.search("grad_theta").keys():
            # ✅ FIX: Skip if grad_theta is None (happens when server not initialized)
            if model_outputs[key] is not None:
                eikonal_loss += loss_terms.get_eikonal_loss(model_outputs[key])
            # Optionally log first occurrence
            elif not hasattr(self, '_logged_grad_theta_none'):
                logger.debug(f"Skipping eikonal loss for {key} - grad_theta is None (server not ready)")
                self._logged_grad_theta_none = True

        # Eikonal loss with threshold
        low_bnd_eikonal = 0.0008
        eikonal_loss *= 0.00001
        if eikonal_loss > low_bnd_eikonal:
            loss_dict["loss/eikonal"] = eikonal_loss

        # MANO canonical SDF loss
        mano_cano_loss = 0.0
        for key in model_outputs.search("pts2mano_sdf_cano").keys():
            node_id = key.split(".")[0]
            gt_sdf = model_outputs[f"{node_id}.pts2mano_sdf_cano"].detach()
            pred_sdf = model_outputs[f"{node_id}.pred_sdf"]
            mano_cano_loss += loss_terms.get_mano_cano_loss(
                pred_sdf, gt_sdf, limit=0.01, scores=image_scores
            )

        # MANO cano loss
        if mano_cano_loss > 0:
            loss_dict["loss/mano_cano"] = mano_cano_loss * 5.0

        # Total loss
        if len(loss_dict) > 0:
            loss_dict["loss"] = sum([loss_dict[k] for k in loss_dict.keys()])
        else:
            loss_dict["loss"] = torch.tensor(0.0, device=device, requires_grad=True)

        return loss_dict


# ========================================================================
# PHASE 4: Contact Refinement Loss Module
# ========================================================================
# Implements Stage 2 contact-based refinement using GHOP contact utilities.
# Computes penetration (repulsion) and attraction losses to enforce
# physically plausible hand-object contact.
# ========================================================================

class Phase4ContactLoss(nn.Module):
    """Stage 2 contact refinement for Phase 4.

    This loss module computes contact-aware losses for hand-object interaction:
    1. Penetration loss: Repels hand vertices that penetrate object surface
    2. Attraction loss: Draws contact-prone vertices toward object surface
    3. Damping loss: Regularizes vertex movement for temporal stability

    Args:
        contact_thresh (float): Distance threshold for contact detection (meters)
        collision_thresh (float): Penetration threshold for collision penalty (meters)
        contact_zones (str): Contact region mode ('zones', 'all', 'adaptive')
        w_penetration (float): Weight for penetration loss component
        w_attraction (float): Weight for attraction loss component
        w_damping (float): Weight for damping regularization
    """

    def __init__(
        self,
        contact_thresh=0.01,
        collision_thresh=0.005,
        contact_zones='zones',
        w_penetration=100.0,
        w_attraction=10.0,
        w_damping=0.1
    ):
        super().__init__()

        if not GHOP_CONTACT_AVAILABLE:
            raise ImportError(
                "[Phase 4] GHOPContactRefinement not available. "
                "Please ensure src.model.ghop.contact_refinement is installed."
            )

        # Initialize GHOP contact refiner
        # NOTE: Pass weight_damp=0 since we handle damping separately
        self.contact_refiner = GHOPContactRefinement(
            contact_thresh=contact_thresh,
            collision_thresh=collision_thresh,
            contact_zones=contact_zones
        )

        # Store loss weights
        self.w_penetration = w_penetration
        self.w_attraction = w_attraction
        self.w_damping = w_damping

        # Thresholds for logging
        self.contact_thresh = contact_thresh
        self.collision_thresh = collision_thresh

        # Cache for previous hand vertices (for damping loss)
        self.prev_hand_verts = None

        logger.info(
            f"[Phase4ContactLoss] Initialized with:\n"
            f"  Contact threshold: {contact_thresh}m\n"
            f"  Collision threshold: {collision_thresh}m\n"
            f"  Penetration weight: {w_penetration}\n"
            f"  Attraction weight: {w_attraction}\n"
            f"  Damping weight: {w_damping}"
        )

    def forward(self, hand_verts, hand_faces, obj_verts, obj_faces):
        """Compute contact loss with GHOP utilities.

        Args:
            hand_verts: [B, 778, 3] or [778, 3] hand vertex positions
            hand_faces: [1538, 3] hand triangle face indices
            obj_verts: [B, V, 3] or [V, 3] object vertex positions
            obj_faces: [F, 3] object triangle face indices

        Returns:
            contact_loss (torch.Tensor): Scalar total contact loss
            contact_info (dict): Dictionary with loss components and metrics
        """
        # Ensure batch dimension exists
        if hand_verts.dim() == 2:
            hand_verts = hand_verts.unsqueeze(0)
        if obj_verts.dim() == 2:
            obj_verts = obj_verts.unsqueeze(0)

        # Call GHOP contact refiner
        # NOTE: We pass weight_pen and weight_miss, but NOT weight_damp
        contact_loss, contact_metrics = self.contact_refiner(
            hand_verts=hand_verts,
            hand_faces=hand_faces,
            obj_verts=obj_verts,
            obj_faces=obj_faces,
            weight_pen=self.w_penetration,
            weight_miss=self.w_attraction,
            weight_damp=0.0  # Disable internal damping - we compute it separately
        )

        # Compute damping loss separately for better control
        damping_loss = torch.tensor(0.0, device=hand_verts.device)
        if self.prev_hand_verts is not None:
            if self.prev_hand_verts.shape == hand_verts.shape:
                damping_loss = self._compute_damping_loss(
                    hand_verts,
                    self.prev_hand_verts
                )
                # Add to total loss
                contact_loss = contact_loss + damping_loss * self.w_damping
                contact_metrics['damping'] = damping_loss.item()
            else:
                logger.warning(
                    f"[Phase4ContactLoss] Shape mismatch: "
                    f"prev {self.prev_hand_verts.shape} vs curr {hand_verts.shape}"
                )

        # Update cache for next iteration
        self.prev_hand_verts = hand_verts.detach().clone()

        # Construct detailed info dictionary
        # NOTE: Do NOT call .item() on contact_loss here - breaks gradient
        contact_info = {
            'total': contact_loss.item() if not contact_loss.requires_grad else contact_loss.detach().item(),
            'penetration': contact_metrics.get('penetration', 0.0),
            'attraction': contact_metrics.get('attraction', 0.0),
            'dist_mean': contact_metrics.get('dist_mean', 0.0),
            'num_contacts': contact_metrics.get('num_contacts', 0),
            'num_penetrations': contact_metrics.get('num_penetrations', 0),
            'damping': contact_metrics.get('damping', 0.0),
        }

        return contact_loss, contact_info

    def _compute_damping_loss(self, curr_verts, prev_verts):
        """Compute damping loss to limit vertex displacement.

        Args:
            curr_verts: [B, N, 3] current vertex positions
            prev_verts: [B, N, 3] previous vertex positions

        Returns:
            damping_loss (torch.Tensor): Scalar damping loss
        """
        # L2 distance between current and previous vertices
        displacement = torch.norm(curr_verts - prev_verts, dim=-1)  # [B, N]

        # Average displacement across batch and vertices
        damping_loss = displacement.mean()

        return damping_loss

    def reset_cache(self):
        """Reset cached previous vertices (call at epoch boundaries)."""
        self.prev_hand_verts = None
        logger.debug("[Phase4ContactLoss] Cache reset")


# ========================================================================
# END PHASE 4: Contact Refinement Loss Module
# ========================================================================


# ========================================================================
# PHASE 4: Unified HOLD Loss with Contact Refinement (OPTIONAL)
# ========================================================================
# NOTE: This class is OPTIONAL. The recommended approach is to use the
# standard Loss class and add Phase 4 contact losses directly in
# HOLD.training_step() for better modularity.
# ========================================================================

class HOLDLoss(nn.Module):
    """Unified loss module that combines standard HOLD losses with Phase 4 contact.

    **IMPORTANT**: This class is provided for compatibility but is NOT the
    recommended approach. Phase 4 contact losses should be computed directly
    in HOLD.training_step() for better separation of concerns.

    This wrapper orchestrates:
    1. Standard HOLD losses (RGB, semantic, eikonal, etc.) via `Loss` class
    2. Phase 3 GHOP SDS losses (handled in hold.py)
    3. Phase 4 contact refinement losses

    Args:
        args: Argument namespace with training configuration
        phase4_config (dict): Phase 4 configuration dictionary
    """

    def __init__(self, args, phase4_config=None):
        super().__init__()

        # Initialize base HOLD loss
        self.base_loss = Loss(args)
        self.args = args

        # Initialize Phase 4 contact loss if enabled
        self.phase4_enabled = False
        if phase4_config is not None and phase4_config.get('enabled', False):
            if not GHOP_CONTACT_AVAILABLE:
                logger.error(
                    "[HOLDLoss] Phase 4 contact loss requested but "
                    "GHOPContactRefinement not available. Disabling Phase 4."
                )
                self.contact_loss_module = None
            else:
                self.contact_loss_module = Phase4ContactLoss(
                    contact_thresh=phase4_config.get('contact_thresh', 0.01),
                    collision_thresh=phase4_config.get('collision_thresh', 0.005),
                    contact_zones=phase4_config.get('contact_zones', 'zones'),
                    w_penetration=phase4_config.get('w_penetration', 100.0),
                    w_attraction=phase4_config.get('w_attraction', 10.0),
                    w_damping=phase4_config.get('w_damping', 0.1)
                )
                self.phase4_enabled = True
                self.contact_start_iter = phase4_config.get('contact_start_iter', 500)
                self.contact_warmup_iters = phase4_config.get('contact_warmup_iters', 100)
                self.w_contact = phase4_config.get('w_contact', 10.0)
                logger.info("[HOLDLoss] Phase 4 contact loss enabled")
        else:
            self.contact_loss_module = None
            logger.info("[HOLDLoss] Phase 4 disabled")

    def forward(
        self,
        batch,
        model_outputs,
        current_step=None,
        hand_verts=None,
        hand_faces=None,
        obj_verts=None,
        obj_faces=None
    ):
        """Compute unified HOLD + Phase 4 losses.

        Args:
            batch: Input batch
            model_outputs: Model output dictionary
            current_step: Current training iteration
            hand_verts: [B, 778, 3] hand vertices (for Phase 4)
            hand_faces: [1538, 3] hand faces (for Phase 4)
            obj_verts: [B, V, 3] or List[[V, 3]] object vertices (for Phase 4)
            obj_faces: [F, 3] or List[[F, 3]] object faces (for Phase 4)

        Returns:
            loss_dict: Dictionary with all loss components
        """
        # Compute base HOLD losses
        loss_dict = self.base_loss(batch, model_outputs)

        # Add Phase 4 contact loss if enabled and past start iteration
        if (
            self.phase4_enabled
            and self.contact_loss_module is not None
            and current_step is not None
            and current_step >= self.contact_start_iter
        ):
            # Check if mesh inputs are provided
            if (
                hand_verts is not None
                and hand_faces is not None
                and obj_verts is not None
                and obj_faces is not None
            ):
                try:
                    # Handle list-based object meshes (batch of variable-size meshes)
                    if isinstance(obj_verts, list):
                        # Process each batch element separately
                        total_contact_loss = 0.0
                        contact_info_accum = {
                            'penetration': 0.0,
                            'attraction': 0.0,
                            'dist_mean': 0.0,
                            'num_contacts': 0,
                            'num_penetrations': 0
                        }

                        batch_size = hand_verts.shape[0]
                        num_valid = 0

                        for b in range(batch_size):
                            h_verts = hand_verts[b:b+1]  # [1, 778, 3]
                            o_verts = obj_verts[b]
                            o_faces = obj_faces[b]

                            # Skip empty meshes
                            if o_verts.shape[0] == 0:
                                continue

                            # Compute contact loss for this sample
                            contact_loss_b, contact_info_b = self.contact_loss_module(
                                hand_verts=h_verts,
                                hand_faces=hand_faces,
                                obj_verts=o_verts,
                                obj_faces=o_faces
                            )

                            total_contact_loss += contact_loss_b
                            num_valid += 1

                            # Accumulate metrics
                            for key in contact_info_accum:
                                contact_info_accum[key] += contact_info_b.get(key, 0.0)

                        # Average over valid samples
                        if num_valid > 0:
                            contact_loss = total_contact_loss / num_valid
                            for key in contact_info_accum:
                                contact_info_accum[key] /= num_valid
                            contact_info = contact_info_accum
                            contact_info['total'] = contact_loss.item()
                        else:
                            logger.warning("[HOLDLoss] No valid object meshes for contact loss")
                            return loss_dict
                    else:
                        # Tensor-based meshes (uniform batch)
                        contact_loss, contact_info = self.contact_loss_module(
                            hand_verts=hand_verts,
                            hand_faces=hand_faces,
                            obj_verts=obj_verts,
                            obj_faces=obj_faces
                        )

                    # Progressive warmup schedule
                    contact_progress = min(
                        (current_step - self.contact_start_iter) / self.contact_warmup_iters,
                        1.0
                    )
                    contact_weight = self.w_contact * contact_progress

                    # Add to loss dictionary
                    weighted_contact = contact_loss * contact_weight
                    loss_dict['loss/contact'] = weighted_contact
                    loss_dict['loss'] = loss_dict['loss'] + weighted_contact

                    # Add contact metrics
                    loss_dict['contact/penetration'] = contact_info.get('penetration', 0.0)
                    loss_dict['contact/attraction'] = contact_info.get('attraction', 0.0)
                    loss_dict['contact/dist_mean'] = contact_info.get('dist_mean', 0.0)
                    loss_dict['contact/weight'] = contact_weight
                    loss_dict['contact/num_contacts'] = contact_info.get('num_contacts', 0)
                    loss_dict['contact/num_penetrations'] = contact_info.get('num_penetrations', 0)

                    logger.debug(
                        f"[HOLDLoss - Step {current_step}] Contact loss: {weighted_contact.item():.4f}, "
                        f"weight: {contact_weight:.2f}"
                    )

                except Exception as e:
                    logger.error(f"[HOLDLoss] Contact loss computation failed: {e}")
                    import traceback
                    traceback.print_exc()

        return loss_dict

    def reset_cache(self):
        """Reset Phase 4 cache at epoch boundaries."""
        if self.phase4_enabled and self.contact_loss_module is not None:
            self.contact_loss_module.reset_cache()


# ========================================================================
# END PHASE 4: Unified HOLD Loss
# ========================================================================