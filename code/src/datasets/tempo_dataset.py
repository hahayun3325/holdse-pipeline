import random
import numpy as np

from src.datasets.image_dataset import ImageDataset
import sys

sys.path = [".."] + sys.path

import common.ld_utils as ld_utils


class TempoDataset(ImageDataset):
    """
    Load a temporal window per __getitem__ by sampling a (start, end) index pair.
    """

    def __init__(self, args):
        super().__init__(args)
        self.offset = args.offset
        print(f"Using Temporal Dataset with offset: {self.offset}")

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
    def __getitem__(self, idx):
        start_idx, end_idx = random.choice(self.pairs)
        start_idx = int(start_idx)
        end_idx = int(end_idx)

        left = super().__getitem__(start_idx)
        right = super().__getitem__(end_idx)
        data = ld_utils.stack_dl(ld_utils.ld2dl([left, right]), dim=0, verbose=False)
        self.sample_idx += 1
        return data

    def __len__(self):
        return self.args.tempo_len
