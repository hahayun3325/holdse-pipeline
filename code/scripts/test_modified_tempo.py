import sys
sys.path.insert(0, '.')

from src.datasets.tempo_dataset import TempoDataset
import argparse

# Create mock args matching ImageDataset's expectations
args = argparse.Namespace(
    # Dataset paths
    data_dir='data/hold_MC1_ho3d/build',
    case='hold_MC1_ho3d',
    seq_name='hold_MC1_ho3d',
    
    # Temporal settings
    offset=1,          # Consecutive frames
    tempo_len=100,
    
    # Debug and logging (REQUIRED by ImageDataset)
    debug=False,       # ✅ ADD: Prevents debug dump
    verbose=False,     # ✅ ADD: Reduces logging
    
    # Training settings (may be checked by ImageDataset)
    split='train',
    img_size=512,
    
    # Object/hand configuration (if needed)
    setup_with_object=True,
    setup_with_hand=True,
)

print("=" * 70)
print("TESTING MODIFIED TempoDataset")
print("=" * 70)

# Create dataset
try:
    dataset = TempoDataset(args)
    print(f"\n✅ Dataset created successfully")
    print(f"   Total samples: {len(dataset)}")
except Exception as e:
    print(f"\n❌ Dataset creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Load first sample
try:
    sample = dataset[0]
    print(f"\n✅ Sample loaded successfully")
    print(f"   Sample keys ({len(sample.keys())} total):")
    
    # Group keys by category for readability
    spatial_keys = [k for k in sample.keys() if k in ['uv', 'intrinsics', 'extrinsics', 'c2w']]
    gt_keys = [k for k in sample.keys() if k.startswith('gt.')]
    hand_keys = [k for k in sample.keys() if 'hand' in k.lower() or 'right' in k.lower()]
    temporal_keys = [k for k in sample.keys() if k in ['hA_n', 'c2w_n', 'frame_idx', 'sequence_id']]
    other_keys = [k for k in sample.keys() if k not in spatial_keys + gt_keys + hand_keys + temporal_keys]
    
    print(f"   - Spatial: {spatial_keys}")
    print(f"   - Ground truth: {gt_keys}")
    print(f"   - Hand/params: {hand_keys}")
    print(f"   - Temporal (NEW): {temporal_keys}")
    if other_keys:
        print(f"   - Other: {other_keys}")
        
except Exception as e:
    print(f"\n❌ Sample loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check temporal fields
print("\n" + "=" * 70)
print("TEMPORAL FIELD VALIDATION")
print("=" * 70)

required_fields = {
    'hA_n': 'Next frame hand pose',
    'c2w_n': 'Next frame camera',
    'frame_idx': 'Current frame index',
    'sequence_id': 'Sequence identifier'
}

all_present = True
for field, description in required_fields.items():
    present = field in sample
    status = '✅' if present else '❌'
    print(f"{status} {field:15} - {description}")
    
    if present and field in ['hA_n', 'c2w_n']:
        print(f"   Shape: {sample[field].shape}")
        print(f"   Type:  {type(sample[field])}")
        
        # Validate dimensions
        if field == 'hA_n':
            expected_dof = sample[field].shape[-1]
            if expected_dof == 45:
                print(f"   ✅ Hand DOF correct (45)")
            elif expected_dof == 48:
                print(f"   ⚠️  Hand DOF: 48 (should have been sliced to 45!)")
            else:
                print(f"   ⚠️  Hand DOF: {expected_dof} (expected 45)")
        
        if field == 'c2w_n':
            if sample[field].shape == (4, 4):
                print(f"   ✅ Camera matrix shape correct (4x4)")
            else:
                print(f"   ⚠️  Unexpected camera shape (expected 4x4)")
    
    elif present and field == 'frame_idx':
        print(f"   Value: {sample[field]}")
    elif present and field == 'sequence_id':
        print(f"   Value: {sample[field]}")
    
    all_present = all_present and present

print("\n" + "=" * 70)
if all_present:
    print("✅ ALL TEMPORAL FIELDS PRESENT")
    print("✅ Phase 5 temporal consistency will ACTIVATE")
    print("\nNext step: Re-run training with modified TempoDataset")
else:
    print("❌ MISSING TEMPORAL FIELDS")
    print("❌ Phase 5 will still SKIP")
    print("\nDebug: Check which fields are missing above")
print("=" * 70)
