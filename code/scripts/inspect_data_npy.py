import numpy as np
import sys

data_path = "data/hold_MC1_ho3d/build/data.npy"
data = np.load(data_path, allow_pickle=True).item()

print("=" * 70)
print("DATA.NPY STRUCTURE ANALYSIS")
print("=" * 70)

# Check sequence info
if 'seq_name' in data:
    print(f"Sequence name: {data['seq_name']}")

# Check frames
if 'entities' in data and 'right' in data['entities']:
    hand_data = data['entities']['right']
    if 'hand_poses' in hand_data:
        n_frames = hand_data['hand_poses'].shape[0]
        print(f"Number of frames: {n_frames}")
        print(f"Hand pose shape: {hand_data['hand_poses'].shape}")

# Check cameras
camera_keys = [k for k in data.get('cameras', {}).keys() if 'worldmat' in k]
print(f"Number of camera matrices: {len(camera_keys)}")

if len(camera_keys) > 0:
    print(f"First camera: {camera_keys[0]}")
    print(f"Last camera: {camera_keys[-1]}")

# Check if data supports temporal access
print("\n" + "=" * 70)
print("TEMPORAL SUPPORT ANALYSIS")
print("=" * 70)

if n_frames > 1:
    print(f"✅ Multiple frames available: {n_frames}")
    print(f"✅ Data SUPPORTS temporal consistency")
    print(f"❌ But TempoDataset doesn't create hA_n/c2w_n fields")
    print(f"\nRECOMMENDATION:")
    print(f"  Data has {n_frames} sequential frames.")
    print(f"  Need to either:")
    print(f"    1. Convert to GHOP format (cameras_hoi.npz, hands.npz)")
    print(f"    2. Modify TempoDataset to extract temporal pairs from data.npy")
    print(f"    3. Create VideoSequenceDataset for data.npy format")
else:
    print(f"❌ Only single frame")
    print(f"❌ Data does NOT support temporal consistency")

print("=" * 70)
