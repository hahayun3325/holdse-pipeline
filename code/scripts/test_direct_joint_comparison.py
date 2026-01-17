# test_joint_subset.py
import torch
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')

# Load your prediction
pred = torch.load('logs/evaluation_results/MC1_3phases_e30_infer_official_nframes100_predictions_20260115_171441.pkl')

# Load official HOLD prediction for comparison
official_pred = torch.load('/home/fredcui/Projects/hold-master/code/logs/cb20a1702/checkpoints/last.ckpt.predictions.pkl')

print("HOLDSE joints shape:", pred['j3d_c.right'].shape)
print("Official HOLD joints shape:", official_pred['j3d_c.right'].shape)

# Compare frame 0 joint positions
frame_idx = 0
holdse_joints = pred['j3d_c.right'][frame_idx]  # [16, 3]
official_joints = official_pred['j3d_c.right'][frame_idx]  # [16, 3]

print("\nJoint-by-joint comparison (frame 0):")
print(f"{'Idx':<4} {'HOLDSE (x,y,z)':<30} {'Official (x,y,z)':<30} {'Distance':<10}")
print("="*80)

for i in range(16):
    h_joint = holdse_joints[i]
    o_joint = official_joints[i]
    dist = torch.norm(h_joint - o_joint).item()
    print(f"{i:<4} ({h_joint[0]:6.3f}, {h_joint[1]:6.3f}, {h_joint[2]:6.3f})  "
          f"({o_joint[0]:6.3f}, {o_joint[1]:6.3f}, {o_joint[2]:6.3f})  {dist:6.2f}mm")

# Check if joint 0 (wrist) matches
wrist_dist = torch.norm(holdse_joints[0] - official_joints[0]).item()
print(f"\nWrist (joint 0) distance: {wrist_dist:.2f}mm")
if wrist_dist > 50:
    print("⚠️  WARNING: Wrist position differs significantly!")
    print("This suggests coordinate frame mismatch or different joint ordering")