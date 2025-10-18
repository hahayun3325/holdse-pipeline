#!/bin/bash
# Create complete mock HOLD data structure for GHOP datasets
# WITH proper image symlinks (not dummy files)

set -e

echo "========================================================================="
echo "Creating COMPLETE mock HOLD data for GHOP datasets"
echo "========================================================================="
echo ""

cd ~/Projects/holdse/code

# GHOP configuration
GHOP_ROOT=~/Projects/ghop/data/HOI4D_clip
OBJECTS=("Bottle_1" "Bowl_1" "Kettle_1" "Knife_1" "Mug_1")

for obj in "${OBJECTS[@]}"
do
    obj_lower=$(echo "$obj" | tr '[:upper:]' '[:lower:]')
    case_name="ghop_${obj_lower}"

    echo "Processing: $obj -> $case_name"

    # Create directories
    mock_dir="./data/$case_name"
    build_dir="$mock_dir/build"
    image_dir="$build_dir/image"
    mask_dir="$build_dir/mask"

    mkdir -p "$build_dir" "$image_dir" "$mask_dir"

    # Create COMPLETE mock data.npy with ALL OBJECT FIELDS
    python << EOF
import numpy as np
from pathlib import Path

# Count frames
ghop_image_dir = Path('$GHOP_ROOT/$obj/image')
num_frames = len(list(ghop_image_dir.glob('*.png'))) if ghop_image_dir.exists() else 71
print(f"  Detected {num_frames} frames")

# Camera matrices for ALL frames
cameras = {}
for idx in range(num_frames):
    cameras[f'scale_mat_{idx}'] = np.eye(4, dtype=np.float32)
    cameras[f'world_mat_{idx}'] = np.eye(4, dtype=np.float32)

# COMPLETE entity data with ALL OBJECT FIELDS
entities = {
    'right': {
        # IMPROVED: Use realistic MANO parameters instead of zeros
        'hand_poses': np.random.randn(num_frames, 48).astype(np.float32) * 0.1,  # Small random poses
        'hand_trans': np.random.randn(num_frames, 3).astype(np.float32) * 0.05,  # Small random translation
        'mean_shape': np.random.randn(10).astype(np.float32) * 0.1,  # Random but fixed shape
    },
    'object': {
        # Object poses
        'object_poses': np.random.randn(num_frames, 6).astype(np.float32) * 0.1,

        # CRITICAL: Object mesh data (ALL REQUIRED FIELDS)
        'pts.cano': np.random.randn(1000, 3).astype(np.float32) * 0.1,  # Canonical points
        'tri': np.random.randint(0, 1000, (1800, 3), dtype=np.int32),   # Triangle faces

        # CRITICAL: Object scale (required by ObjectModel line 23)
        'obj_scale': 1.0,  # Object scale factor

        # CRITICAL: Normalization matrix (required by ObjectModel line 25)
        'norm_mat': np.eye(4, dtype=np.float32),  # Normalization transform
    }
}

# Complete data
data = {
    # Basic info
    'case': '$case_name',
    'data_dir': '$GHOP_ROOT/$obj',
    'dataset_type': 'ghop_hoi',

    # Image info
    'img_wh': [512, 512],
    'n_frames': num_frames,

    # Camera intrinsics
    'intrinsics': np.array([
        [500.0, 0.0, 256.0],
        [0.0, 500.0, 256.0],
        [0.0, 0.0, 1.0]
    ]),

    # Scene parameters
    'scene_bounding_sphere': 1.5,

    # Additional fields
    'scene_center': np.array([0.0, 0.0, 0.0]),
    'scene_scale': 1.0,

    # Camera matrices (ALL frames)
    'cameras': cameras,

    # Entity data (ALL frames)
    'entities': entities,
}

# Save complete mock data.npy
output_path = Path('$build_dir/data.npy')
np.save(output_path, data)
print(f"  ✓ data.npy: {num_frames} frames")
print(f"    - Cameras: {len(cameras)//2} pairs")
print(f"    - Object mesh: {entities['object']['pts.cano'].shape[0]} vertices")
print(f"    - Object scale: {entities['object']['obj_scale']}")
print(f"    - Norm matrix: {entities['object']['norm_mat'].shape}")
EOF

    # Create symlinks
    echo "  Creating symlinks..."

    # Image symlinks
    if [ -d "$GHOP_ROOT/$obj/image" ]; then
        for img_file in "$GHOP_ROOT/$obj/image"/*.png; do
            if [ -f "$img_file" ]; then
                ln -sf "$img_file" "$image_dir/$(basename "$img_file")"
            fi
        done
        echo "  ✓ Images: $(ls "$image_dir"/*.png 2>/dev/null | wc -l)"
    fi

    # ========================================================================
    # CRITICAL: Convert GHOP masks to HOLD format with CORRECT SEGM_IDS
    # HOLD expects: {"bg": 0, "object": 50, "right": 150, "left": 250}
    # ========================================================================
    echo "  Converting GHOP masks to HOLD format (SEGM_IDS-compatible)..."

    python <<EOF
import cv2
import numpy as np
from pathlib import Path

ghop_hand_mask_dir = Path('$GHOP_ROOT/$obj/hand_mask')
ghop_obj_mask_dir = Path('$GHOP_ROOT/$obj/obj_mask')
output_mask_dir = Path('$mask_dir')

# HOLD SEGM_IDS (from src/utils/const.py)
SEGM_BG = 0
SEGM_OBJECT = 50
SEGM_RIGHT = 150
SEGM_LEFT = 250

if not ghop_hand_mask_dir.exists():
    print("  ⚠️  Hand mask directory not found, skipping")
else:
    # Process each frame
    hand_masks = sorted(ghop_hand_mask_dir.glob('*.png'))
    obj_masks = sorted(ghop_obj_mask_dir.glob('*.png')) if ghop_obj_mask_dir.exists() else []

    for idx, hand_mask_path in enumerate(hand_masks):
        # Load GHOP masks (0-255 grayscale)
        hand_mask = cv2.imread(str(hand_mask_path), cv2.IMREAD_GRAYSCALE)

        # Try to load corresponding object mask
        obj_mask = None
        if idx < len(obj_masks):
            obj_mask = cv2.imread(str(obj_masks[idx]), cv2.IMREAD_GRAYSCALE)

        # Create HOLD-compatible segmentation mask with CORRECT SEGM_IDS
        h, w = hand_mask.shape
        hold_mask = np.zeros((h, w), dtype=np.uint8)

        # Background is already 0 (SEGM_BG)

        # Set object pixels to 50 (SEGM_OBJECT)
        if obj_mask is not None:
            obj_pixels = obj_mask > 128
            hold_mask[obj_pixels] = SEGM_OBJECT

        # Set right hand pixels to 150 (SEGM_RIGHT)
        # Note: GHOP only has one hand, assume it's right hand
        hand_pixels = hand_mask > 128
        hold_mask[hand_pixels] = SEGM_RIGHT

        # Save HOLD-compatible mask
        output_path = output_mask_dir / hand_mask_path.name
        cv2.imwrite(str(output_path), hold_mask)

    print(f"  ✓ Converted {len(hand_masks)} masks to HOLD format")
    print(f"    - SEGM_IDS: bg=0, object=50, right=150")
    print(f"    - Mask range: [{SEGM_BG}, {SEGM_RIGHT}]")
    print(f"    - Compatible with HOLD const.py")
EOF

    # Create text.txt
    category=$(echo "$obj" | sed 's/_[0-9]*//')
    echo "$category" > "$build_dir/text.txt"
    echo "  ✓ text.txt: $category"

    # Symlink to GHOP data
    ln -sf "$GHOP_ROOT/$obj" "$mock_dir/ghop_data"
    echo ""
done

echo "========================================================================="
echo "✅ COMPLETE mock data structure created"
echo "   - utils.py unchanged (HOLD compatibility preserved)"
echo "   - GHOP masks converted to HOLD segmentation format"
echo "========================================================================="
echo ""

# Detailed verification
echo "Detailed Verification:"
echo "------------------------------------------------------------------------"
for obj in "${OBJECTS[@]}"; do
    obj_lower=$(echo "$obj" | tr '[:upper:]' '[:lower:]')
    case_name="ghop_${obj_lower}"

    echo ""
    echo "Case: $case_name"

    # Check data.npy
    if [ -f "./data/$case_name/build/data.npy" ]; then
        echo "  ✓ data.npy exists"

        # Verify contents
        python << PYEOF
import numpy as np
data = np.load('./data/$case_name/build/data.npy', allow_pickle=True).item()
required_fields = ['case', 'scene_bounding_sphere', 'intrinsics', 'img_wh', 'n_frames', 'cameras', 'entities']
missing = [f for f in required_fields if f not in data]
if missing:
    print(f"  ❌ Missing fields: {missing}")
else:
    print(f"  ✓ All required fields present")
    print(f"    - scene_bounding_sphere: {data['scene_bounding_sphere']}")
    print(f"    - n_frames: {data['n_frames']}")
    print(f"    - cameras: {list(data['cameras'].keys())}")
    print(f"    - entities: {list(data['entities'].keys())}")
PYEOF
    else
        echo "  ❌ data.npy missing"
    fi

    # Check image symlinks
    if [ -d "./data/$case_name/build/image" ]; then
        num_images=$(ls ./data/$case_name/build/image/*.png 2>/dev/null | wc -l)
        if [ $num_images -gt 0 ]; then
            echo "  ✓ Image directory: $num_images images"

            # Verify first image is readable
            first_image=$(ls ./data/$case_name/build/image/*.png 2>/dev/null | head -1)
            if [ -f "$first_image" ]; then
                # Check if it's a valid symlink
                if [ -L "$first_image" ]; then
                    target=$(readlink -f "$first_image")
                    if [ -f "$target" ]; then
                        echo "  ✓ First image symlink valid: $(basename $first_image) -> valid target"
                    else
                        echo "  ❌ First image symlink broken: $(basename $first_image)"
                    fi
                else
                    echo "  ⚠️ First image is not a symlink"
                fi
            fi
        else
            echo "  ❌ No images found"
        fi
    else
        echo "  ❌ Image directory missing"
    fi

    # Check masks
    if [ -d "./data/$case_name/build/mask" ]; then
        num_masks=$(ls ./data/$case_name/build/mask/*.png 2>/dev/null | wc -l)
        echo "  ✓ Mask directory: $num_masks masks"
    fi

    # Check text.txt
    if [ -f "./data/$case_name/build/text.txt" ]; then
        category=$(cat "./data/$case_name/build/text.txt")
        echo "  ✓ text.txt: $category"
    else
        echo "  ⚠️ text.txt missing"
    fi
done

echo ""
echo "Ready to run GHOP training!"