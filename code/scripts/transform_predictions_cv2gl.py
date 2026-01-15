#!/usr/bin/env python3
"""Transform predictions from OpenCV to OpenGL coordinate system."""

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common import transforms as tf

def transform_cv2gl_simple(coords):
    """Simple cv→gl: flip Y and Z."""
    # coords: [N, ..., 3]
    coords_gl = coords.clone()
    coords_gl[..., 1] *= -1  # Flip Y
    coords_gl[..., 2] *= -1  # Flip Z
    return coords_gl

def main():
    pred_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else pred_path.replace('.pkl', '_gl.pkl')
    
    print(f"Loading: {pred_path}")
    pred = torch.load(pred_path)
    
    # Transform all 3D coordinates
    for key in ['v3d_c.right', 'j3d_c.right', 'v3d_c.object', 
                'root.right', 'root.object']:
        if key in pred:
            print(f"Transforming {key}...")
            pred[key] = transform_cv2gl_simple(pred[key])
    
    # Root-aligned coordinates also need transformation
    for key in ['j3d_ra.right', 'v3d_ra.object', 'v3d_right.object']:
        if key in pred:
            print(f"Transforming {key}...")
            pred[key] = transform_cv2gl_simple(pred[key])
    
    torch.save(pred, output_path)
    print(f"\n✓ Saved to: {output_path}")
    
if __name__ == '__main__':
    main()
