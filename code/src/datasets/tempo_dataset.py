import random
import numpy as np

from src.datasets.image_dataset import ImageDataset
import sys

sys.path = [".."] + sys.path

import common.ld_utils as ld_utils


class TempoDataset(ImageDataset):
    """
    Load a temporal window per __getitem__ by sampling a (start, end) index pair.

    NOW PROVIDES:
    - hA_n: Next frame hand pose (for Phase 5 temporal consistency)
    - c2w_n: Next frame camera
    - frame_idx: Current frame index
    - sequence_id: Video sequence identifier
    """

    def __init__(self, args):
        super().__init__(args)
        self.offset = args.offset
        print(f"[TempoDataset] Using Temporal Dataset with offset: {self.offset}")

        total_frames = len(self.img_paths)
        # Explicitly compute candidate indices
        start_idx = np.arange(total_frames - self.offset)
        end_idx = start_idx + self.offset

        # (Optional) validate that all end_idx are in range
        valid_mask = (end_idx >= 0) & (end_idx < total_frames)
        if not np.all(valid_mask):
            invalid = np.where(~valid_mask)[0]
            print(f"[TempoDataset] ⚠️ Found {len(invalid)} invalid pairs; dropping them.")
            start_idx = start_idx[valid_mask]
            end_idx = end_idx[valid_mask]

        self.pairs = np.stack((start_idx, end_idx), axis=1)
        self.sample_idx = 0

        print(f"[TempoDataset] Total frames: {total_frames}")
        print(f"[TempoDataset] Num pairs: {len(self.pairs)}")
        if len(self.pairs) > 0:
            print(f"[TempoDataset] First pair: {self.pairs[0]}")
            print(f"[TempoDataset] Last pair:  {self.pairs[-1]}")

        # Log Phase 5 support
        print(f"[TempoDataset] ✅ Phase 5 temporal consistency ENABLED")
        print(f"[TempoDataset]    Will provide hA_n and c2w_n fields")

    def __getitem__(self, idx):
        start_idx, end_idx = random.choice(self.pairs)
        start_idx = int(start_idx)
        end_idx = int(end_idx)

        # ================================================================
        # Load current frame (t) and next frame (t+offset)
        # ================================================================
        current_frame = super().__getitem__(start_idx)
        next_frame = super().__getitem__(end_idx)

        # ================================================================
        # PRESERVE ORIGINAL STACKING for batch dimension compatibility
        # ================================================================
        # Stack frames to create batch dimension (original behavior)
        data = ld_utils.stack_dl(
            ld_utils.ld2dl([current_frame, next_frame]),
            dim=0,
            verbose=False
        )

        # ================================================================
        # ADD TEMPORAL FIELDS for Phase 5
        # ================================================================
        # Extract hand pose from next frame (already in data[1] from stacking)
        # But Phase 5 needs explicit hA_n field
        if 'right.params' in next_frame:
            hand_next = next_frame['right.params']
        elif 'gt.right.hand_pose' in next_frame:
            hand_next = next_frame['gt.right.hand_pose']
        else:
            # Fallback: extract from stacked data
            if 'right.params' in data:
                # data['right.params'] is stacked, shape (2, ...)
                hand_next = data['right.params'][1]  # Take second frame
            else:
                raise KeyError(
                    f"Cannot find hand pose. "
                    f"Available keys: {list(next_frame.keys())}"
                )

        # Handle dimension: if hand_next is batched (1, D), squeeze it
        if hand_next.dim() == 2 and hand_next.shape[0] == 1:
            hand_next = hand_next.squeeze(0)

        # Slice to 45 DOF if needed
        if hand_next.shape[-1] > 45:
            hand_next = hand_next[..., :45]
            if not hasattr(self, '_logged_slice'):
                print(f"[TempoDataset] Hand pose: {hand_next.shape[-1]+3} → 45 DOF")
                self._logged_slice = True

        # Add batch dimension if missing (Phase 5 expects batched)
        if hand_next.dim() == 1:
            hand_next = hand_next.unsqueeze(0)  # (D,) → (1, D)

        # Extract camera from next frame
        if 'c2w' in next_frame:
            camera_next = next_frame['c2w']
        elif 'extrinsics' in next_frame:
            camera_next = next_frame['extrinsics']
        elif 'extrinsics' in data:
            camera_next = data['extrinsics'][1]  # Take second frame from stack
        else:
            raise KeyError(
                f"Cannot find camera. "
                f"Available keys: {list(next_frame.keys())}"
            )

        # Add batch dimension if missing
        if camera_next.dim() == 2:  # (4, 4) → (1, 4, 4)
            camera_next = camera_next.unsqueeze(0)

        # ================================================================
        # Add temporal fields to stacked data
        # ================================================================
        data['hA_n'] = hand_next           # Next frame hand pose (1, 45)
        data['c2w_n'] = camera_next        # Next frame camera (1, 4, 4)
        data['frame_idx'] = start_idx      # Current frame index
        data['sequence_id'] = self.args.case  # Sequence identifier

        self.sample_idx += 1
        return data  # ✅ Return stacked data WITH temporal fields

    def __len__(self):
        return self.args.tempo_len
