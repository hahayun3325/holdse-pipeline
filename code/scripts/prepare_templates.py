"""
Preprocess YCB/ShapeNet meshes into normalized templates for Chamfer loss.
Supports fuzzy matching for descriptive category names.
"""
import os
import torch
import trimesh
import numpy as np
from pathlib import Path

def normalize_mesh(vertices):
    """Normalize mesh to unit sphere centered at origin."""
    vertices = vertices - vertices.mean(axis=0)
    max_dist = np.abs(vertices).max()
    if max_dist > 0:
        vertices = vertices / max_dist
    return vertices

def process_ycb_templates(ycb_dir, output_dir):
    """
    Process YCB models into normalized PyTorch tensors.

    Args:
        ycb_dir: Path to PyBullet YCB objects directory
        output_dir: Output directory for processed templates
    """
    os.makedirs(output_dir, exist_ok=True)

    # ============================================================
    # Descriptive Category → YCB Mapping (from your metadata)
    # ============================================================
    # Format: 'descriptive_name': ('YcbDirectoryName', 'template_name')

    category_to_ycb = {
        # Your actual categories from metadata
        'cheez-it box': ('YcbCrackerBox', 'cracker_box'),
        'cheez-it cracker box': ('YcbCrackerBox', 'cracker_box'),
        'cracker box': ('YcbCrackerBox', 'cracker_box'),
        'crackerbox': ('YcbCrackerBox', 'cracker_box'),

        'banana': ('YcbBanana', 'banana'),

        'power drill': ('YcbPowerDrill', 'power_drill'),
        'drill': ('YcbPowerDrill', 'power_drill'),

        "french's mustard bottle": ('YcbMustardBottle', 'mustard_bottle'),
        'mustard bottle': ('YcbMustardBottle', 'mustard_bottle'),
        'mustard': ('YcbMustardBottle', 'mustard_bottle'),

        'game/media box': ('YcbGelatinBox', 'gelatin_box'),
        'game/product box': ('YcbGelatinBox', 'gelatin_box'),
        'game box': ('YcbGelatinBox', 'gelatin_box'),
        'product box': ('YcbGelatinBox', 'gelatin_box'),
        'small rectangular object': ('YcbGelatinBox', 'gelatin_box'),
        'package box': ('YcbGelatinBox', 'gelatin_box'),

        # Additional objects available in PyBullet (for future use)
        'chips can': ('YcbChipsCan', 'chips_can'),
        'pringles': ('YcbChipsCan', 'chips_can'),

        'tomato soup can': ('YcbTomatoSoupCan', 'tomato_soup_can'),
        'soup can': ('YcbTomatoSoupCan', 'tomato_soup_can'),

        'scissors': ('YcbScissors', 'scissors'),
        'hammer': ('YcbHammer', 'hammer'),
        'foam brick': ('YcbFoamBrick', 'foam_brick'),
        'pear': ('YcbPear', 'pear'),
        'strawberry': ('YcbStrawberry', 'strawberry'),
        'tennis ball': ('YcbTennisBall', 'tennis_ball'),

        # Fallback mappings for objects NOT in PyBullet (will skip gracefully)
        'domino sugar box': ('YcbSugarBox', 'sugar_box'),  # Not available
        'sugar box': ('YcbSugarBox', 'sugar_box'),  # Not available
        'red mug': ('YcbMug', 'mug'),  # Not available
        'red speckled mug': ('YcbMug', 'mug'),  # Not available
        'mug': ('YcbMug', 'mug'),  # Not available
        'cleaning product bottle': ('YcbBleachCleanser', 'bleach_cleanser'),  # Not available
        'soft scrub bottle': ('YcbBleachCleanser', 'bleach_cleanser'),  # Not available
        'bleach': ('YcbBleachCleanser', 'bleach_cleanser'),  # Not available
    }

    processed_count = 0
    skipped_count = 0

    for category_name, (ycb_dir_name, template_name) in category_to_ycb.items():
        mesh_dir = os.path.join(ycb_dir, ycb_dir_name)

        # Updated mesh search patterns (handles PyBullet naming)
        mesh_candidates = [
            'textured_simple_reoriented.obj',  # PyBullet primary format
            'textured_reoriented.obj',         # PyBullet alternative
            'textured.obj',                     # Original YCB format
            'textured_simple.obj',             # Simplified YCB
            'model.obj',                        # Generic fallback
            'nontextured.ply',                 # PLY format
        ]

        mesh_path = None
        for candidate in mesh_candidates:
            full_path = os.path.join(mesh_dir, candidate)
            if os.path.exists(full_path):
                mesh_path = full_path
                break

        if mesh_path is None:
            # Only warn for unique template names (avoid duplicate warnings)
            if skipped_count == 0 or template_name not in [t for c, (d, t) in list(category_to_ycb.items())[:processed_count+skipped_count]]:
                print(f"⚠️  Skipping '{template_name}': No mesh in {mesh_dir}")
            skipped_count += 1
            continue

        try:
            # Load mesh
            mesh = trimesh.load(mesh_path, force='mesh')

            # Extract vertices
            vertices = np.array(mesh.vertices, dtype=np.float32)

            # Normalize to unit sphere
            vertices_norm = normalize_mesh(vertices)

            # Convert to PyTorch
            vertices_torch = torch.from_numpy(vertices_norm)

            # Save as .pt file (use template name, not category)
            output_path = os.path.join(output_dir, f"{template_name}.pt")

            # Only save if not already processed (avoid duplicates)
            if not os.path.exists(output_path):
                torch.save(vertices_torch, output_path)
                print(f"✅ {category_name:30s} → {template_name:20s}: {vertices_torch.shape[0]:6d} vertices")
                processed_count += 1
            else:
                print(f"⏭️  {category_name:30s} → {template_name:20s}: Already exists, skipping")

        except Exception as e:
            print(f"❌ Failed '{category_name}': {e}")

    print(f"\n{'='*70}")
    print(f"✅ Processed: {processed_count} unique templates")
    print(f"⚠️  Skipped:   {skipped_count} entries (missing YCB models)")
    print(f"{'='*70}")

    # List created templates
    if processed_count > 0:
        print("\nCreated templates:")
        for pt_file in sorted(Path(output_dir).glob("*.pt")):
            template = torch.load(pt_file)
            print(f"  - {pt_file.name:25s} ({template.shape[0]:6d} vertices)")

    return processed_count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ycb_dir',
        type=str,
        default='/home/fredcui/Projects/holdse/downloads/ycb/pybullet-object-models/pybullet_object_models/ycb_objects',
        help='Path to PyBullet YCB objects directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/templates',
        help='Output directory for processed templates'
    )
    args = parser.parse_args()

    print("=" * 70)
    print("TEMPLATE PREPROCESSING: YCB Objects for HOLD")
    print("=" * 70)
    print(f"Input:  {args.ycb_dir}")
    print(f"Output: {args.output_dir}")
    print("=" * 70)
    print()

    process_ycb_templates(args.ycb_dir, args.output_dir)

'''
python scripts/prepare_templates.py --ycb_dir /path/to/ycb --output_dir data/templates
'''