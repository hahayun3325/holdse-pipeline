import sys
import os
sys.path.insert(0, 'code')

from src.datasets.ghop_hoi_dataset import GHOPHOIDataset

data_dir = "data/hold_MC1_ho3d/ghop_data"

if not os.path.exists(data_dir):
    print(f"❌ GHOP data directory not found: {data_dir}")
    print(f"This is why --use_ghop was disabled.")
    sys.exit(1)

print(f"✅ GHOP data directory exists: {data_dir}")

# Check for required files
required_files = ['cameras_hoi.npz', 'hands.npz']
for f in required_files:
    path = os.path.join(data_dir, f)
    if os.path.exists(path):
        print(f"✅ {f} exists")
    else:
        print(f"❌ {f} missing")

# Try to instantiate dataset
try:
    class Args:
        data_dir = data_dir
        split = 'train'
        
    dataset = GHOPHOIDataset(
        data_dir=data_dir,
        split='train',
        args=Args()
    )
    
    print(f"\n✅ GHOPHOIDataset loaded successfully")
    print(f"   Total frames: {dataset.n_frames}")
    print(f"   Frame pairs: {len(dataset)}")
    
    # Test loading one sample
    sample = dataset[0]
    print(f"\n✅ Sample loaded successfully")
    print(f"   Keys: {list(sample.keys())}")
    
    if 'hA_n' in sample and 'c2w_n' in sample:
        print(f"   ✅ Has temporal fields (hA_n, c2w_n)")
        print(f"   ✅ Phase 5 will work with this dataset")
    else:
        print(f"   ❌ Missing temporal fields")
        print(f"   ❌ Phase 5 won't work")
        
except Exception as e:
    print(f"\n❌ Error loading GHOPHOIDataset: {e}")
    import traceback
    traceback.print_exc()
    print(f"\nThis is likely why --use_ghop was disabled.")
