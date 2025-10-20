"""GHOP HOI4D dataset adapter for HOLDSE temporal consistency testing."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from loguru import logger

class GHOPHOIDataset(Dataset):
    """
    GHOP HOI4D video dataset for temporal consistency.
    
    Provides frame pairs (t, t+1) with:
    - Hand poses: hA (current), hA_n (next)
    - Camera poses: c2w (current), c2w_n (next)
    - Sequence metadata: sequence_id, frame_idx
    
    Args:
        data_dir: Path to GHOP sequence (e.g., ~/Projects/ghop/data/HOI4D_clip/Bottle_1)
        split: 'train' or 'val'
        downsample: Image downsample factor (default 1.0)
        args: Additional arguments for compatibility
    """
    
    def __init__(self, data_dir, split='train', downsample=1.0, args=None, load_images=False):
        self.data_dir = Path(data_dir).expanduser()
        self.split = split
        self.load_images = load_images

        logger.info(f"[GHOPHOIDataset] Loading: {self.data_dir}")
        
        # Validate dataset structure
        self._validate_dataset()
        
        # Load camera data
        camera_file = self.data_dir / 'cameras_hoi.npz'
        camera_dict = np.load(camera_file)
        
        # Extract camera-to-world transforms
        if 'cTw' in camera_dict:
            cTw = torch.from_numpy(camera_dict['cTw']).float()
            self.c2w = self._invert_transform(cTw)
            logger.info(f"  Camera transforms: inverted cTw to c2w")
        elif 'wTc' in camera_dict:
            self.c2w = torch.from_numpy(camera_dict['wTc']).float()
            logger.info(f"  Camera transforms: using wTc directly")
        else:
            raise ValueError("Camera poses not found in cameras_hoi.npz")
        
        # Extract intrinsics (optional)
        if 'K_pix' in camera_dict:
            self.intrinsics = torch.from_numpy(camera_dict['K_pix']).float()
            logger.info(f"  Intrinsics: loaded from K_pix")
        else:
            # Default intrinsics
            N = len(self.c2w)
            self.intrinsics = torch.eye(4).unsqueeze(0).repeat(N, 1, 1)
            logger.warning(f"  Intrinsics: using default (identity)")
        
        # Load hand data
        hands_file = self.data_dir / 'hands.npz'
        hands_dict = np.load(hands_file)
        
        # Hand articulation parameters
        if 'hA' in hands_dict:
            hA = hands_dict['hA']
            if hA.ndim == 3:
                hA = hA.squeeze(1)
            self.hA = torch.from_numpy(hA).float()
        else:
            raise ValueError("Hand poses 'hA' not found in hands.npz")

        # Hand shape parameters (beta)
        if 'beta' in hands_dict:
            beta = hands_dict['beta']
            if beta.ndim == 2:
                beta = beta.mean(axis=0)
            self.beta = torch.from_numpy(beta).float()
            logger.info(f"  Beta: loaded from hands.npz, shape={self.beta.shape}")
        else:
            # Only use default if beta not in file
            self.beta = torch.zeros(10)
            logger.warning(f"  Beta: not found in hands.npz, using default (zeros)")

        # Load object category
        text_file = self.data_dir / 'text.txt'
        if text_file.exists():
            with open(text_file, 'r') as f:
                self.category = f.read().strip().lower()
        else:
            self.category = self.data_dir.name.split('_')[0].lower()
        
        # ================================================================
        # Setup frame indices (MUST BE BEFORE hand_trans)
        # ================================================================
        self.n_frames = len(self.c2w)  # ← DEFINE n_frames HERE
        split_ratio = 0.8
        split_idx = int(self.n_frames * split_ratio)
        
        if split == 'train':
            self.frame_indices = list(range(0, split_idx - 1))
        else:  # val
            self.frame_indices = list(range(split_idx, self.n_frames - 1))
        
        logger.info(f"  Total frames: {self.n_frames}")
        logger.info(f"  {split} frame pairs: {len(self.frame_indices)}")
        logger.info(f"  Category: {self.category}")

        # ================================================================
        # Extract hand translation (AFTER n_frames is defined)
        # ================================================================
        if 'hand_trans' in hands_dict or 'hand_translation' in hands_dict:
            trans_key = 'hand_trans' if 'hand_trans' in hands_dict else 'hand_translation'
            hand_trans = hands_dict[trans_key]
            if hand_trans.ndim == 3:
                hand_trans = hand_trans.squeeze(1)
            self.hand_trans = torch.from_numpy(hand_trans).float()
            logger.info(f"  Hand translation: loaded from {trans_key}, shape={self.hand_trans.shape}")
        else:
            # Default: zero translation (NOW self.n_frames exists)
            self.hand_trans = torch.zeros(self.n_frames, 3)
            logger.warning(f"  Hand translation: not found, using default (zeros)")

        # ================================================================
        # HOLD compatibility attributes
        # ================================================================
        # Image paths (for save_misc)
        self.image_dir = self.data_dir / 'image'
        if self.image_dir.exists():
            self.image_files = sorted(list(self.image_dir.glob('*.png')))
            self.img_paths = [str(p) for p in self.image_files]
            logger.info(f"  Images found: {len(self.image_files)}")
        else:
            logger.warning(f"  Image directory not found: {self.image_dir}")
            self.image_files = []
            self.img_paths = []

        # Intrinsics and extrinsics aliases (for save_misc)
        # These allow HOLD code to access camera data uniformly
        self.intrinsics_all = self.intrinsics  # [N, 4, 4]

        # Compute extrinsics (world-to-camera) from c2w
        self.extrinsics = torch.inverse(self.c2w)  # [N, 4, 4]
        self.extrinsics_all = self.extrinsics

        # Scene scale (for save_misc)
        self.scale = 1.0

        logger.info(f"  ✅ HOLD compatibility attributes added")
        logger.info(f"    - img_paths: {len(self.img_paths)} images")
        logger.info(f"    - intrinsics_all: {self.intrinsics_all.shape}")
        logger.info(f"    - extrinsics_all: {self.extrinsics_all.shape}")
        logger.info(f"    - scale: {self.scale}")

    def _validate_dataset(self):
        """Validate GHOP dataset structure."""
        required_files = ['cameras_hoi.npz', 'hands.npz']
        
        for file in required_files:
            filepath = self.data_dir / file
            if not filepath.exists():
                raise FileNotFoundError(
                    f"Missing required GHOP file: {file}\n"
                    f"Expected in: {self.data_dir}\n"
                    f"Make sure GHOP HOI4D dataset is properly extracted."
                )
        
        # Check image directory (optional but recommended)
        image_dir = self.data_dir / 'image'
        if not image_dir.exists():
            logger.warning(f"Image directory not found: {image_dir}")
        else:
            num_images = len(list(image_dir.glob('*.png')))
            logger.info(f"  Images found: {num_images}")
    
    def _invert_transform(self, T):
        """
        Invert 4x4 transformation matrices.
        
        Args:
            T: [N, 4, 4] transforms
        
        Returns:
            T_inv: [N, 4, 4] inverted transforms
        """
        R = T[:, :3, :3]  # [N, 3, 3]
        t = T[:, :3, 3:]  # [N, 3, 1]
        
        R_inv = R.transpose(1, 2)  # [N, 3, 3]
        t_inv = -R_inv @ t  # [N, 3, 1]
        
        T_inv = torch.zeros_like(T)
        T_inv[:, :3, :3] = R_inv
        T_inv[:, :3, 3:4] = t_inv
        T_inv[:, 3, 3] = 1.0
        
        return T_inv
    
    def __len__(self):
        """Return number of frame pairs."""
        return len(self.frame_indices)
    
    def __getitem__(self, idx):
        """
        Get frame pair (t, t+1) for temporal consistency.

        Returns:
            sample: Dict with current and next frame data
        """
        # Get frame indices
        frame_idx = self.frame_indices[idx]
        frame_idx_n = frame_idx + 1
        
        # Build sample dict
        sample = {
            # Identity & Metadata
            'idx': torch.tensor([frame_idx]),
            'sequence_id': self.data_dir.name,
            'frame_idx': torch.tensor([frame_idx]),
            'category': self.category,
            'text': f'a hand grasping a {self.category}',
            
            # Current Frame
            'c2w': self.c2w[frame_idx],              # [4, 4]
            'intrinsics': self.intrinsics[frame_idx], # [4, 4]
            'hA': self.hA[frame_idx],                 # [45]
            'right.betas': self.beta,                 # [10]
            'th_betas': self.beta,                    # [10] alias
            
            # Next Frame (CRITICAL FOR PHASE 5)
            'c2w_n': self.c2w[frame_idx_n],           # [4, 4]
            'intrinsics_n': self.intrinsics[frame_idx_n], # [4, 4]
            'hA_n': self.hA[frame_idx_n],             # [45] CRITICAL!

            # ================================================================
            # FIX: Add img_size and pixel_per_batch for validation/inference
            # ================================================================
            'img_size': [480, 640],  # [H, W] - Python list for HOLD compatibility
            'pixel_per_batch': 8192,  # ADD THIS LINE - Standard rendering batch size
        }
        
        return sample


def create_ghop_dataset(data_dir, split='train', **kwargs):
    """Factory function for compatibility with HOLD's dataset creation."""
    return GHOPHOIDataset(data_dir, split=split, **kwargs)
