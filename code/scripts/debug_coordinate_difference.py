import torch
import numpy as np
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')

# '''
# Check 1: Compare normalize_shift application
# '''
# # Load dataset to check normalize_shift
# dataset = np.load('data/hold_MC1_ho3d/build/data.npy', allow_pickle=True).item()
# print("normalize_shift from dataset:", dataset['normalize_shift'])
#
# # Check misc file
# misc = np.load('/home/fredcui/Projects/hold/code/logs/cb20a1702/misc/000080000.npy', allow_pickle=True).item()
# print("scale from misc:", misc['scale'])
#
# # Load both predictions
# hold_style = torch.load('logs/evaluation_results/MC1_official_hold_style_predictions_20260109_205919.pkl')
# your_style = torch.load('logs/evaluation_results/MC1_stage1_e200_official_nframes1_predictions_20260108_065728.pkl')
#
# # Check if one has normalize_shift applied, the other doesn't
# # Expected normalize_shift: [-0.0085, -0.014, 0.426] (42.6cm in Z)
# shift = np.array([-0.0085238, -0.01372686, 0.42570806])
#
# # Test: Does your_style + shift ≈ hold_style?
# your_corrected = your_style['j3d_c.right'][0, 0].numpy() + shift
# print(f"\nYour style + shift: {your_corrected}")
# print(f"HOLD style: {hold_style['j3d_c.right'][0, 0].numpy()}")
# print(f"Difference after correction: {np.linalg.norm(your_corrected - hold_style['j3d_c.right'][0, 0].numpy()) * 1000:.2f} mm")
#
# # Test: Does hold_style - shift ≈ your_style?
# hold_corrected = hold_style['j3d_c.right'][0, 0].numpy() - shift
# print(f"\nHOLD style - shift: {hold_corrected}")
# print(f"Your style: {your_style['j3d_c.right'][0, 0].numpy()}")
# print(f"Difference after correction: {np.linalg.norm(hold_corrected - your_style['j3d_c.right'][0, 0].numpy()) * 1000:.2f} mm")
#
# '''
# Check 2: Compare scale factors
# '''
# # Check if scales match
# print("\n=== SCALE COMPARISON ===")
# misc = np.load('/home/fredcui/Projects/hold/code/logs/cb20a1702/misc/000080000.npy', allow_pickle=True).item()
# print(f"Misc scale: {misc['scale']}")
#
# dataset = np.load('data/hold_MC1_ho3d/build/data.npy', allow_pickle=True).item()
# print(f"Dataset scene_bounding_sphere: {dataset.get('scene_bounding_sphere', 'NOT FOUND')}")
#
# # Check if one uses scale, the other uses 1/scale
# your_wrist_mag = torch.norm(your_style['j3d_c.right'][0, 0])
# hold_wrist_mag = torch.norm(hold_style['j3d_c.right'][0, 0])
# print(f"\nYour wrist magnitude: {your_wrist_mag:.4f}")
# print(f"HOLD wrist magnitude: {hold_wrist_mag:.4f}")
# print(f"Ratio: {hold_wrist_mag / your_wrist_mag:.4f}")

# '''
# Check 3: Verify map_deform2eval implementations match
# '''
# # Compare the two map_deform2eval functions
# import inspect
#
# # Your version
# from scripts.extract_predictions import map_deform2eval as your_map
# # HOLD version
# from scripts.extract_from_checkpoint_hold_style import map_deform2eval as hold_map
#
# print("=== YOUR map_deform2eval ===")
# print(inspect.getsource(your_map))
#
# print("\n=== HOLD map_deform2eval ===")
# print(inspect.getsource(hold_map))

'''
Check 4: Extraction Results after Fix
'''

import torch

# Your new extraction with fixed scale
fixed = torch.load('logs/evaluation_results/MC1_stage1_e200_official_nframes100_Test_predictions_20260109_220004.pkl')

# HOLD-style extraction
hold = torch.load('logs/evaluation_results/MC1_official_hold_style_predictions_20260109_205919.pkl')

# Compare
print("Fixed extraction (frame 0 wrist):", fixed['j3d_c.right'][0, 0])
print("HOLD extraction (frame 0 wrist):", hold['j3d_c.right'][0, 0])

diff = ((fixed['j3d_c.right'][0, 0] - hold['j3d_c.right'][0, 0])**2).sum().sqrt()
print(f"Difference: {diff * 1000:.2f} mm")  # Should be < 10mm after fix