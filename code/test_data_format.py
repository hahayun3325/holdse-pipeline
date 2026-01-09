import numpy as np
import torch
import sys
sys.path.append('./src')

# Test 1: Load existing data.npy
data_path = "data/hold_MC1_ho3d/build/data.npy"
try:
    entities = np.load(data_path, allow_pickle=True).item()["entities"]
    print("✅ Loaded existing data.npy")
    print(f"Keys: {entities.keys()}")
    print(f"Right hand keys: {entities['right'].keys()}")
    print(f"Shapes:")
    for key, val in entities['right'].items():
        if isinstance(val, np.ndarray):
            print(f"  {key}: {val.shape}")
except Exception as e:
    print(f"❌ Failed to load: {e}")

# Test 2: Create synthetic data (mask sampling output format)
try:
    synthetic_entities = {
        "right": {
            "hand_poses": np.random.randn(144, 48).astype(np.float32),  # N frames, 48D
            "hand_trans": np.random.randn(144, 3).astype(np.float32),   # N frames, 3D
            "mean_shape": np.random.randn(10).astype(np.float32)        # 10D beta
        }
    }
    
    # Save and reload
    np.save("test_data.npy", {"entities": synthetic_entities})
    reloaded = np.load("test_data.npy", allow_pickle=True).item()["entities"]
    print("✅ Synthetic data format valid")
    
except Exception as e:
    print(f"❌ Synthetic data failed: {e}")
