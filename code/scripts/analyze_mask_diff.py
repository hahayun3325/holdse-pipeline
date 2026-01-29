import numpy as np
import cv2
import glob

# Load HOLDSE masks
holdse_masks = sorted(glob.glob("logs/640c1f867/test/visuals/object.mask_prob/*.png"))
hold_masks = sorted(glob.glob("logs/official_hold/test/visuals/object.mask_prob/*.png"))

differences = []
for hs, ho in zip(holdse_masks[:10], hold_masks[:10]):
    hs_mask = cv2.imread(hs, cv2.IMREAD_GRAYSCALE) / 255.0
    ho_mask = cv2.imread(ho, cv2.IMREAD_GRAYSCALE) / 255.0

    # Compute difference
    diff = np.abs(hs_mask - ho_mask).mean()
    differences.append(diff)

    # Compute IoU
    intersection = (hs_mask * ho_mask).sum()
    union = ((hs_mask + ho_mask) > 0).sum()
    iou = intersection / (union + 1e-6)

    print(f"Frame {hs.split('/')[-1]}: Diff={diff:.3f}, IoU={iou:.3f}")

print(f"\nMean mask difference: {np.mean(differences):.3f}")
print(f"Expected: <0.05 (good), >0.2 (broken)")