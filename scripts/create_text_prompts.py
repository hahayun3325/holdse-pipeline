"""
Future-proof text prompt creation system.
Supports both manual (Phase 1) and LLM-based (Phase 2) generation.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import argparse

# Category mapping (can be extended)
CATEGORY_MAPPING = {
    'mug': {'category': 'Mug', 'aliases': ['cup', 'coffee_mug']},
    'bottle': {'category': 'Bottle', 'aliases': ['water_bottle', 'plastic_bottle']},
    'bowl': {'category': 'Bowl', 'aliases': ['dish', 'container']},
    'can': {'category': 'Can', 'aliases': ['soda_can', 'tin_can']},
    'box': {'category': 'Box', 'aliases': ['package', 'container']},
    'phone': {'category': 'Phone', 'aliases': ['smartphone', 'mobile']},
    'scissors': {'category': 'Scissors', 'aliases': ['shears', 'cutter']},
    'stapler': {'category': 'Stapler', 'aliases': ['office_stapler']},
}


class TextPromptManager:
    """
    Manages text prompt creation with support for manual and LLM-based methods.
    """

    def __init__(self, data_root="./data"):
        self.data_root = Path(data_root)
        self.category_mapping = CATEGORY_MAPPING

    def create_manual_prompts(self, dataset_path: Path, category: str = None) -> Dict:
        """
        Create text prompts manually (Phase 1).

        Args:
            dataset_path: Path to dataset directory
            category: Override category (if None, auto-detect)

        Returns:
            Dictionary with prompt metadata
        """
        # Auto-detect category if not provided
        if category is None:
            category = self._detect_category(dataset_path.name)

        # Normalize category
        category = category.capitalize()

        # Create prompt metadata
        metadata = {
            'version': '1.0',
            'method': 'manual',
            'created_at': datetime.now().isoformat(),
            'category': category,
            'simple_prompt': category,
            'ghop_prompt': f"an image of a hand grasping a {category}",
            'detailed_prompts': [
                f"a hand grasping a {category.lower()}",
                f"a hand holding a {category.lower()}",
                f"hand-{category.lower()} interaction"
            ],
            'source': 'directory_name',
            'confidence': 0.8  # Manual detection confidence
        }

        return metadata

    def _detect_category(self, dirname: str) -> str:
        """Detect category from directory name."""
        parts = dirname.split('_')
        if len(parts) >= 2:
            category_raw = parts[1]
            category = ''.join([c for c in category_raw if not c.isdigit()]).lower()

            # Check if in mapping
            if category in self.category_mapping:
                return self.category_mapping[category]['category']
            return category.capitalize()

        return 'Object'

    def save_text_prompts(self, dataset_path: Path, metadata: Dict):
        """
        Save text prompts in multiple formats.

        Creates:
        - text.txt: Simple GHOP format
        - metadata.json: Rich format for future LLM updates
        """
        build_path = dataset_path / "build"

        if not build_path.exists():
            print(f"Warning: {build_path} does not exist")
            return False

        # 1. Create text.txt (GHOP format)
        text_file = build_path / "text.txt"
        with open(text_file, 'w') as f:
            f.write(metadata['simple_prompt'])

        # 2. Create metadata.json (extended format)
        metadata_file = build_path / "text_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        return True

    def load_text_prompts(self, dataset_path: Path) -> Dict:
        """
        Load text prompts with priority fallback.

        Priority:
        1. metadata.json (richest information)
        2. text.txt (simple category)
        3. Auto-detect from dirname
        """
        build_path = dataset_path / "build"

        # Priority 1: Load from metadata.json
        metadata_file = build_path / "text_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            print(f"[Load] Metadata from {metadata_file.name} (method: {metadata.get('method', 'unknown')})")
            return metadata

        # Priority 2: Load from text.txt
        text_file = build_path / "text.txt"
        if text_file.exists():
            with open(text_file, 'r') as f:
                category = f.read().strip()

            # Reconstruct metadata from simple format
            metadata = {
                'version': '1.0',
                'method': 'text_file',
                'category': category,
                'simple_prompt': category,
                'ghop_prompt': f"an image of a hand grasping a {category}",
                'source': 'text.txt',
                'confidence': 0.7
            }
            print(f"[Load] Category from text.txt: '{category}'")
            return metadata

        # Priority 3: Auto-detect
        metadata = self.create_manual_prompts(dataset_path)
        print(f"[Load] Auto-detected category: '{metadata['category']}'")
        return metadata


def create_manual_prompts_batch(data_root="./data",
                                save_metadata=True,
                                overwrite=False):
    """
    Create manual text prompts for all HOLD datasets.

    Args:
        data_root: Root directory containing datasets
        save_metadata: Whether to save metadata.json (future-proof)
        overwrite: Whether to overwrite existing text.txt files
    """
    manager = TextPromptManager(data_root)
    data_root = Path(data_root)

    datasets = [d for d in data_root.iterdir()
                if d.is_dir() and d.name.startswith('hold_')]

    if len(datasets) == 0:
        print(f"No HOLD datasets found in {data_root}")
        return

    print(f"\n{'=' * 70}")
    print(f"Creating text prompts for {len(datasets)} datasets")
    print(f"Save metadata: {save_metadata}")
    print('=' * 70)

    results = []
    for dataset_path in sorted(datasets):
        try:
            # Check if already exists
            text_file = dataset_path / "build" / "text.txt"
            if text_file.exists() and not overwrite:
                with open(text_file, 'r') as f:
                    existing_cat = f.read().strip()
                print(f"⊙ {dataset_path.name:30s} → {existing_cat:15s} (existing)")
                results.append((dataset_path.name, existing_cat, "⊙"))
                continue

            # Create prompts
            metadata = manager.create_manual_prompts(dataset_path)

            # Save
            if manager.save_text_prompts(dataset_path, metadata):
                category = metadata['category']
                print(f"✓ {dataset_path.name:30s} → {category:15s}")
                results.append((dataset_path.name, category, "✓"))
            else:
                results.append((dataset_path.name, None, "✗"))

        except Exception as e:
            print(f"✗ Error: {dataset_path.name}: {e}")
            results.append((dataset_path.name, None, "✗"))

    # Summary
    print(f"\n{'=' * 70}")
    print("Summary:")
    print('=' * 70)

    category_counts = {}
    for name, cat, status in results:
        if cat:
            category_counts[cat] = category_counts.get(cat, 0) + 1

    print("\nCategory distribution:")
    for cat in sorted(category_counts.keys()):
        count = category_counts[cat]
        print(f"  {cat:15s}: {count:2d} datasets")

    new_count = sum(1 for _, _, s in results if s == "✓")
    existing_count = sum(1 for _, _, s in results if s == "⊙")
    print(f"\n✓ Created {new_count} new prompts")
    print(f"⊙ Kept {existing_count} existing prompts")
    print('=' * 70)


# ============================================================
# LLM-BASED GENERATION (Phase 2 - Future Implementation)
# ============================================================

def create_llm_prompts_batch(data_root="./data",
                             llm_model="gpt-4-vision",
                             update_existing=False):
    """
    Create text prompts using LLM vision model (Phase 2 - Future).

    This is a placeholder for future LLM integration.
    When implemented, it will:
    1. Load representative images from dataset
    2. Send to LLM with structured prompt
    3. Parse LLM response for category + interaction
    4. Save rich metadata with confidence scores

    Args:
        data_root: Root directory containing datasets
        llm_model: LLM model to use (e.g., gpt-4-vision, claude-vision)
        update_existing: Whether to update existing manual prompts
    """
    print(f"\n{'=' * 70}")
    print("LLM-BASED PROMPT GENERATION (Phase 2)")
    print('=' * 70)
    print("\n⚠️  This feature is not yet implemented.")
    print("\nPlanned functionality:")
    print("  1. Analyze images with vision LLM")
    print("  2. Recognize object categories")
    print("  3. Describe hand-object interactions")
    print("  4. Generate rich, detailed prompts")
    print("  5. Save with confidence scores")
    print("\nFor now, use manual prompt generation:")
    print("  python create_text_prompts.py --batch --manual")
    print('=' * 70)

    # TODO: Implement LLM-based generation
    # See Phase 2 implementation plan below


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create text prompts for HOLD datasets (manual or LLM-based)"
    )

    # Mode selection
    parser.add_argument("--manual", action="store_true",
                        help="Use manual prompt generation (Phase 1 - default)")
    parser.add_argument("--llm", action="store_true",
                        help="Use LLM-based generation (Phase 2 - future)")

    # Dataset selection
    parser.add_argument("--data_path", type=str,
                        help="Single dataset path")
    parser.add_argument("--batch", action="store_true",
                        help="Process all datasets")
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Root directory for batch processing")

    # Options
    parser.add_argument("--category", type=str,
                        help="Override category name (manual mode)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing prompts")
    parser.add_argument("--no-metadata", action="store_false", dest="save_metadata",
                        help="Don't save metadata.json (only text.txt)")

    args = parser.parse_args()

    # Default to manual mode
    if not args.manual and not args.llm:
        args.manual = True

    if args.llm:
        # Phase 2: LLM-based generation
        if args.batch:
            create_llm_prompts_batch(args.data_root, update_existing=args.overwrite)
        else:
            print("Error: LLM mode currently only supports --batch")
            print("For single dataset, use manual mode")

    elif args.manual:
        # Phase 1: Manual generation
        if args.batch:
            create_manual_prompts_batch(
                args.data_root,
                save_metadata=args.save_metadata,
                overwrite=args.overwrite
            )
        elif args.data_path:
            manager = TextPromptManager()
            metadata = manager.create_manual_prompts(
                Path(args.data_path),
                category=args.category
            )
            manager.save_text_prompts(Path(args.data_path), metadata)
        else:
            print("Error: Specify --data_path or --batch")