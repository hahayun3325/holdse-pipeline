#!/usr/bin/env python3
"""
Extract complete architecture with layer dimensions.
"""

import re
import json

print("="*70)
print("COMPLETE ARCHITECTURE EXTRACTION")
print("="*70)

architecture = {
    "ImplicitNet": {},
    "RenderingNet": {},
    "LaplaceDensity": {},
    "ErrorBoundSampler": {},
}

# 1. Extract ImplicitNet dimensions
print("\n1. IMPLICIT NETWORK ARCHITECTURE")
print("-" * 70)

with open('src/networks/shape_net.py', 'r') as f:
    content = f.read()
    
    # Find __init__ method
    init_match = re.search(r'class ImplicitNet.*?def __init__\(self, opt.*?\):(.*?)(?=\n    def )', content, re.DOTALL)
    if init_match:
        init_code = init_match.group(1)
        
        # Extract dims
        dims_match = re.search(r'dims = \[(.*?)\]', init_code)
        if dims_match:
            print(f"Found dims line: {dims_match.group(0)}")
            architecture["ImplicitNet"]["dims_expression"] = dims_match.group(0)
        
        # Extract skip_in
        skip_match = re.search(r'self.skip_in = (.*)', init_code)
        if skip_match:
            print(f"Found skip_in: {skip_match.group(1)}")
            architecture["ImplicitNet"]["skip_in"] = skip_match.group(1)
        
        # Extract multires
        multires_match = re.search(r'opt.multires.*?(\d+)', init_code)
        if multires_match:
            print(f"Multires referenced: {multires_match.group(0)}")
        
        # Count layers
        num_layers_match = re.search(r'self.num_layers = len\(dims\)', init_code)
        if num_layers_match:
            print("Number of layers: len(dims)")

# 2. Extract RenderingNet dimensions
print("\n2. RENDERING NETWORK ARCHITECTURE")
print("-" * 70)

with open('src/networks/texture_net.py', 'r') as f:
    content = f.read()
    
    init_match = re.search(r'class RenderingNet.*?def __init__\(self, opt.*?\):(.*?)(?=\n    def )', content, re.DOTALL)
    if init_match:
        init_code = init_match.group(1)
        
        # Extract dims
        dims_match = re.search(r'dims = \[(.*?)\]', init_code)
        if dims_match:
            print(f"Found dims line: {dims_match.group(0)}")
            architecture["RenderingNet"]["dims_expression"] = dims_match.group(0)
        
        # Extract multires_view
        multires_match = re.search(r'opt.multires_view.*?(\d+)', init_code)
        if multires_match:
            print(f"Multires_view referenced: {multires_match.group(0)}")

# 3. Print summary
print("\n3. ARCHITECTURE SUMMARY")
print("-" * 70)
print(json.dumps(architecture, indent=2))

print("\n4. TYPICAL CONFIGURATION (from debug output)")
print("-" * 70)
print("""
From previous inspection debug output:

ImplicitNet:
  - d_in: 3 (xyz coordinates)
  - multires: 6 → 39 dims after encoding (3 + 3*2*6)
  - dims: [39, 256, 256, 256, 256, 256, 256, 256, 257]
  - skip_in: [4] (skip connection at layer 4)
  - d_out: 1 (SDF)
  - feature_vector_size: 256
  - cond: 'pose' (45 dims for MANO)
  - Total output: 257 = SDF (1) + features (256)

RenderingNet:
  - d_in: 3 (view direction) OR encoded if multires_view > 0
  - feature_vector_size: 256 (from ImplicitNet)
  - multires_view: 4 → 27 dims (3 + 3*2*4)
  - pose embedding: 8 dims (if mode='pose')
  - Total input: 256 (features) + 27 (view) + 8 (pose) = 291
  - dims: [291, 256, 256, 256, 3]
  - d_out: 3 (RGB)

LaplaceDensity:
  - Parameters: beta (learnable)
  - Formula: density = (1/beta) * exp(-|SDF| / beta)

ErrorBoundSampler:
  - N_samples: 64 (typical)
  - Strategy: Stratified sampling
  - Range: [near, far] where far = 2 * sdf_bounding_sphere
""")

print("="*70)
