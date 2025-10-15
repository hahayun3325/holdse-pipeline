"""
LLM-based text prompt generation (Phase 2).
Placeholder for future implementation.
"""

import json
import base64
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from PIL import Image


class LLMPromptGenerator:
    """
    Generate text prompts using vision LLM models.

    Supported models:
    - GPT-4 Vision
    - Claude 3 Vision
    - LLaVA
    - BLIP-2
    """

    def __init__(self, model_name="gpt-4-vision", api_key=None):
        """
        Initialize LLM prompt generator.

        Args:
            model_name: LLM model to use
            api_key: API key for cloud models
        """
        self.model_name = model_name
        self.api_key = api_key

        # TODO: Initialize model client
        # self.client = self._init_model_client()

    def select_representative_frames(self, image_dir: Path,
                                     num_frames: int = 3) -> List[Path]:
        """
        Select representative frames from dataset.

        Strategy:
        - Frame 0: Initial contact
        - Frame N/2: Mid interaction
        - Frame N-1: Final pose

        Args:
            image_dir: Directory containing images
            num_frames: Number of frames to select (default: 3)

        Returns:
            List of image paths
        """
        images = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))

        if len(images) == 0:
            raise ValueError(f"No images found in {image_dir}")

        # Select evenly spaced frames
        indices = np.linspace(0, len(images) - 1, num_frames, dtype=int)
        selected = [images[i] for i in indices]

        return selected

    def encode_image_base64(self, image_path: Path) -> str:
        """Encode image to base64 for API transmission."""
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        return image_data

    def generate_prompt_metadata(self, dataset_path: Path) -> Dict:
        """
        Generate text prompt metadata using LLM vision model.

        Args:
            dataset_path: Path to dataset directory

        Returns:
            Dictionary with rich prompt metadata
        """
        # TODO: Implement LLM API call
        # This is a placeholder showing the expected workflow

        print(f"\n[LLM Generator] Processing {dataset_path.name}...")

        # Step 1: Select representative frames
        image_dir = dataset_path / "build" / "image"
        if not image_dir.exists():
            raise ValueError(f"Image directory not found: {image_dir}")

        frames = self.select_representative_frames(image_dir, num_frames=3)
        print(f"  Selected {len(frames)} frames for analysis")

        # Step 2: Encode images
        encoded_images = [self.encode_image_base64(f) for f in frames]

        # Step 3: Call LLM API
        # response = self._call_llm_api(encoded_images)

        # Placeholder: Return mock response
        metadata = {
            'version': '2.0',  # Version 2.0 indicates LLM-generated
            'method': 'llm_vision',
            'model': self.model_name,
            'created_at': '2025-10-14T15:30:00',

            # Core fields (compatible with Phase 1)
            'category': 'Mug',  # TODO: From LLM response
            'simple_prompt': 'Mug',
            'ghop_prompt': 'an image of a hand grasping a Mug',

            # Rich LLM-generated fields (Phase 2 additions)
            'object_description': 'A ceramic coffee mug with handle',
            'interaction_type': 'power_grasp',
            'hand_pose_description': 'fingers wrapped around handle',
            'detailed_prompts': [
                'A hand firmly grasps a ceramic coffee mug by its handle',
                'Person holding a coffee mug in a natural drinking position',
                'Hand wrapped around the handle of a ceramic mug'
            ],

            # Confidence and alternatives
            'confidence': 0.95,
            'alternatives': ['Cup', 'Container'],

            # Analysis metadata
            'analyzed_frames': [str(f.name) for f in frames],
            'source': 'llm_analysis'
        }

        print(f"  ✓ Category: {metadata['category']} (confidence: {metadata['confidence']:.2f})")

        return metadata

    def _call_llm_api(self, encoded_images: List[str]) -> Dict:
        """
        Call LLM API with images and prompt.

        TODO: Implement actual API calls for:
        - OpenAI GPT-4 Vision
        - Anthropic Claude 3 Vision
        - Local models (LLaVA, BLIP-2)
        """
        raise NotImplementedError("LLM API integration not yet implemented")


def update_dataset_with_llm(data_root="./data",
                            model_name="gpt-4-vision",
                            api_key=None,
                            update_existing=False):
    """
    Update all datasets with LLM-generated prompts.

    Args:
        data_root: Root directory containing datasets
        model_name: LLM model to use
        api_key: API key for cloud models
        update_existing: Whether to re-analyze existing prompts
    """
    generator = LLMPromptGenerator(model_name, api_key)
    data_root = Path(data_root)

    datasets = [d for d in data_root.iterdir()
                if d.is_dir() and d.name.startswith('hold_')]

    print(f"\n{'=' * 70}")
    print(f"LLM-Based Prompt Generation")
    print(f"Model: {model_name}")
    print(f"Datasets: {len(datasets)}")
    print('=' * 70)

    for dataset_path in datasets:
        try:
            # Check if already has LLM metadata
            metadata_file = dataset_path / "build" / "text_metadata.json"
            if metadata_file.exists() and not update_existing:
                with open(metadata_file, 'r') as f:
                    existing = json.load(f)

                if existing.get('method') == 'llm_vision':
                    print(f"⊙ {dataset_path.name} (already has LLM metadata)")
                    continue

            # Generate LLM metadata
            metadata = generator.generate_prompt_metadata(dataset_path)

            # Save metadata
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Update text.txt for backward compatibility
            text_file = dataset_path / "build" / "text.txt"
            with open(text_file, 'w') as f:
                f.write(metadata['simple_prompt'])

            print(f"✓ {dataset_path.name}")

        except Exception as e:
            print(f"✗ Error: {dataset_path.name}: {e}")

    print('=' * 70)


if __name__ == "__main__":
    print("LLM-based prompt generation (Phase 2)")
    print("This feature requires API keys and is not yet fully implemented.")
    print("\nFor now, use manual prompt generation:")
    print("  python create_text_prompts.py --batch --manual")