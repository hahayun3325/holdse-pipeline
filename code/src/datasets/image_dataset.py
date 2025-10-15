import glob
import os

import cv2
import numpy as np
import torch
from loguru import logger


from torch.utils.data import Dataset
from src.datasets.utils import load_image, load_mask
from src.datasets.utils import reform_dict, weighted_sampling, load_K_Rt_from_P
import json
from pathlib import Path
from loguru import logger

class ImageDataset(Dataset):
    def setup_poses(self, data):
        entities = data["entities"]
        out = {}
        for name, val in entities.items():
            reform_fn = reform_dict[name.split("_")[0]]
            out[name] = reform_fn(self.scale, val)

        self.params = out

    def __init__(self, args):
        """
        Initialize dataset with text prompt support.

        MODIFICATIONS:
        1. Call _load_text_metadata() after setup methods
        2. Store text_metadata and category
        3. Log metadata loading status
        """
        self.root = os.path.join("./data", args.case, "build")
        self.args = args
        data = np.load(os.path.join(self.root, "data.npy"), allow_pickle=True).item()

        self.setup_images()
        self.setup_masks()
        self.setup_cameras(data)
        self.setup_poses(data)

        self.debug_dump(args)

        self.num_sample = self.args.num_sample
        self.sampling_strategy = "weighted"

        # ============================================================
        # REPLACE OLD CATEGORY INFERENCE WITH NEW TEXT METADATA SYSTEM
        # ============================================================
        # OLD CODE (DELETE):
        # self.object_category = self._infer_object_category(args)
        # logger.info(f"Using object category: {self.object_category}")

        # NEW CODE (ADD):
        self.text_metadata = self._load_text_metadata(args.case)
        self.category = self.text_metadata.get('category', 'Object')

        logger.info(f"[ImageDataset] Text prompt metadata loaded")
        logger.info(f"  Case: {args.case}")
        logger.info(f"  Category: '{self.category}'")
        logger.info(f"  Method: {self.text_metadata.get('method', 'unknown')}")
        logger.info(f"  Confidence: {self.text_metadata.get('confidence', 0.0):.2f}")

        # Backward compatibility: keep object_category attribute
        self.object_category = self.category

    def debug_dump(self, args):
        if args.debug:
            out = {}
            out["intrinsics_all"] = self.intrinsics_all
            out["extrinsics_all"] = self.extrinsics_all
            out["scale_mats"] = self.scale_mats
            out["world_mats"] = self.world_mats
            out["img_paths"] = self.img_paths
            out["mask_paths"] = self.mask_paths
            out["img_size"] = self.img_size
            out["n_images"] = self.n_images
            out["params"] = self.params
            out["scale"] = self.scale

            out_p = os.path.join(args.log_dir, "dataset_info.pth")
            torch.save(out, out_p)
            print(f"Saved dataset info to {out_p}")

    def _load_text_metadata(self, case_name):
        """
        Load text prompt metadata with 3-tier priority fallback.

        Priority:
        4. text_metadata.json (richest, supports LLM future)
        5. text.txt (simple GHOP format)
        6. Auto-detect from directory name

        Args:
            case_name: Dataset case name (e.g., 'hold_mug1_itw')

        Returns:
            Dictionary with prompt metadata
        """
        build_path = Path(self.root)

        # ============================================================
        # Priority 1: Load from metadata.json (future-proof)
        # ============================================================
        metadata_file = build_path / "text_metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                logger.debug(f"[ImageDataset] Loaded from {metadata_file.name}")
                logger.debug(f"  Version: {metadata.get('version', 'unknown')}")
                logger.debug(f"  Method: {metadata.get('method', 'unknown')}")

                # Validate required fields
                if 'category' not in metadata:
                    logger.warning("metadata.json missing 'category', using fallback")
                    raise ValueError("Missing category")

                return metadata

            except Exception as e:
                logger.warning(f"[ImageDataset] Failed to load metadata.json: {e}")
                # Continue to next priority

        # ============================================================
        # Priority 2: Load from text.txt (GHOP format)
        # ============================================================
        text_file = build_path / "text.txt"
        if text_file.exists():
            try:
                with open(text_file, 'r') as f:
                    category = f.read().strip()

                # Validate non-empty
                if not category:
                    raise ValueError("text.txt is empty")

                # Reconstruct minimal metadata
                metadata = {
                    'version': '1.0',
                    'method': 'text_file',
                    'category': category,
                    'simple_prompt': category,
                    'ghop_prompt': f"an image of a hand grasping a {category}",
                    'source': 'text.txt',
                    'confidence': 0.7
                }

                logger.debug(f"[ImageDataset] Loaded from text.txt: '{category}'")
                return metadata

            except Exception as e:
                logger.warning(f"[ImageDataset] Failed to load text.txt: {e}")
                # Continue to next priority

        # ============================================================
        # Priority 3: Auto-detect from directory name (fallback)
        # ============================================================
        logger.warning(f"[ImageDataset] No text files found, auto-detecting category")

        # Parse directory name: hold_mug1_itw -> mug
        parts = case_name.split('_')

        if len(parts) >= 2:
            # Extract second part: hold_mug1_itw -> mug1
            category_raw = parts[1]

            # Remove trailing digits: mug1 -> mug
            category = ''.join([c for c in category_raw if not c.isdigit()])

            # Capitalize: mug -> Mug
            category = category.capitalize()

            logger.info(f"[ImageDataset] Auto-detected: '{category}' from '{case_name}'")
        else:
            # Fallback to generic
            category = 'Object'
            logger.warning(f"[ImageDataset] Cannot parse '{case_name}', using 'Object'")

        # Construct minimal metadata
        metadata = {
            'version': '1.0',
            'method': 'auto_detect',
            'category': category,
            'simple_prompt': category,
            'ghop_prompt': f"an image of a hand grasping a {category}",
            'source': 'directory_name',
            'confidence': 0.5
        }

        return metadata

    def get_text_prompt(self, format='ghop'):
        """
        Get text prompt in specified format.

        Args:
            format: One of 'simple', 'ghop', 'detailed'

        Returns:
            Text prompt string

        Examples:
            >>> dataset.get_text_prompt('simple')
            'Mug'

            >>> dataset.get_text_prompt('ghop')
            'an image of a hand grasping a Mug'

            >>> dataset.get_text_prompt('detailed')
            'A hand firmly grasps a ceramic coffee mug by its handle'
        """
        if format == 'simple':
            return self.text_metadata.get('simple_prompt', self.category)

        elif format == 'ghop':
            # GHOP format: "an image of a hand grasping a {category}"
            return self.text_metadata.get('ghop_prompt',
                                          f"an image of a hand grasping a {self.category}")

        elif format == 'detailed':
            # If LLM-generated detailed prompts exist, use them
            detailed_prompts = self.text_metadata.get('detailed_prompts', [])
            if detailed_prompts and len(detailed_prompts) > 0:
                return detailed_prompts[0]  # Use first variant
            else:
                # Fallback to GHOP format
                return self.text_metadata.get('ghop_prompt',
                                              f"a hand grasping a {self.category.lower()}")

        else:
            # Unknown format: return simple category
            logger.warning(f"Unknown format '{format}', returning simple prompt")
            return self.category

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        """
        Get a single data sample with text prompts.

        MODIFICATIONS:
        7. Add text prompt fields to batch
        8. Include metadata for debugging
        9. Maintain backward compatibility
        """
        img = load_image(self.img_paths[idx])
        mask = load_mask(self.mask_paths[idx], img.shape)

        img_size = self.img_size
        uv = np.mgrid[: img_size[0], : img_size[1]].astype(np.int32)
        uv = np.flip(uv, axis=0).copy().transpose(1, 2, 0).astype(np.float32)
        entity_keys = self.params.keys()
        params = {key + ".params": self.params[key][idx] for key in entity_keys}

        if self.num_sample > 0:
            hand_types = [key for key in entity_keys if "right" in key or "left" in key]
            num_sample = self.num_sample // len(hand_types)

            uv_list = []
            mask_list = []
            img_list = []
            for hand_type in hand_types:
                samples = weighted_sampling(
                    {"rgb": img, "uv": uv, "obj_mask": mask},
                    img_size,
                    num_sample,
                    hand_type,
                )[0]
                uv_list.append(samples["uv"])
                mask_list.append(samples["obj_mask"])
                img_list.append(samples["rgb"])

            uv = np.concatenate(uv_list, axis=0)
            mask = np.concatenate(mask_list, axis=0)
            img = np.concatenate(img_list, axis=0)

        # ============================================================
        # ENHANCE: Add text prompt fields to batch
        # ============================================================
        batch = {
            "uv": uv.reshape(-1, 2).astype(np.float32),
            "intrinsics": self.intrinsics_all[idx],
            "extrinsics": self.extrinsics_all[idx],
            "im_path": self.img_paths[idx],
            "idx": idx,
            "gt.rgb": img.reshape(-1, 3).astype(np.float32),
            "gt.mask": mask.reshape(-1).astype(np.int64),
            "img_size": self.img_size,
            "total_pixels": self.total_pixels,

            # ========================================================
            # ENHANCED TEXT PROMPT FIELDS
            # ========================================================
            "category": self.category,  # Simple category name
            "object_category": self.category,  # Backward compatibility

            # Text prompts in different formats
            "text_prompt": self.get_text_prompt('ghop'),  # Default GHOP format
            "text_prompt_simple": self.get_text_prompt('simple'),  # Just category
            "text_prompt_detailed": self.get_text_prompt('detailed'),  # LLM-generated

            # Full metadata for debugging/logging
            "text_metadata": self.text_metadata,
        }
        batch.update(params)
        return batch

    def setup_images(self):
        img_dir = os.path.join(self.root, "image")
        self.img_paths = sorted(glob.glob(f"{img_dir}/*.png"))
        assert len(self.img_paths) > 0
        self.img_size = cv2.imread(self.img_paths[0]).shape[:2]
        self.total_pixels = np.prod(self.img_size)
        self.n_images = len(self.img_paths)

    def setup_masks(self):
        mask_dir = os.path.join(self.root, "mask")
        self.mask_paths = sorted(glob.glob(f"{mask_dir}/*.png"))
        if len(self.mask_paths) == 0:
            logger.warning("No mask found, using fake mask")
            self.mask_paths = [None] * self.n_images
        else:
            assert len(self.mask_paths) == self.n_images

    def setup_cameras(self, data):
        camera_dict = data["cameras"]
        self.scale_mats, self.world_mats = [], []
        self.intrinsics_all, self.extrinsics_all = [], []

        for idx in range(self.n_images):
            scale_mat = camera_dict[f"scale_mat_{idx}"].astype(np.float32)
            world_mat = camera_dict[f"world_mat_{idx}"].astype(np.float32)
            self.scale_mats.append(scale_mat)
            self.world_mats.append(world_mat)

            # Compute camera parameters
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, extrinsics = load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.extrinsics_all.append(torch.from_numpy(extrinsics).float())
        self.scale = 1 / self.scale_mats[0][0, 0]
        assert len(self.intrinsics_all) == len(self.extrinsics_all)
