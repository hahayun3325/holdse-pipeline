import numpy as np
import cv2

# Load one sample from dataset
from src.datasets.ho3d import HO3DDataset

dataset = HO3DDataset(split='test', seq_name='SM1')
sample = dataset[0]

print("Dataset sample keys:", sample.keys())
print("RGB shape:", sample['rgb'].shape)
print("Mask shape:", sample.get('mask', 'NOT FOUND'))
print("Object mask shape:", sample.get('object_mask', 'NOT FOUND'))
print("Hand mask shape:", sample.get('hand_mask', 'NOT FOUND'))

# Check if masks are binary or probability
if 'object_mask' in sample:
    obj_mask = sample['object_mask']
    print(f"Object mask range: [{obj_mask.min()}, {obj_mask.max()}]")
    print(f"Object mask mean: {obj_mask.mean():.3f}")
    print(f"Expected: [0, 1] range, mean ~0.1-0.3 (object covers 10-30% of image)")