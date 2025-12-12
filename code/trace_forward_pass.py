#!/usr/bin/env python3
"""
Trace the forward pass for Stage 1 (base HOLD model).
"""

print("="*70)
print("STAGE 1 FORWARD PASS ANALYSIS")
print("="*70)

# Read the HOLD model forward method
with open('src/hold/hold.py', 'r') as f:
    lines = f.readlines()

# Find the forward method
in_forward = False
forward_lines = []
indent_level = 0

for i, line in enumerate(lines):
    if 'def forward(self' in line:
        in_forward = True
        indent_level = len(line) - len(line.lstrip())
    
    if in_forward:
        forward_lines.append((i+1, line))
        
        # Stop at next method definition at same indent level
        if line.strip().startswith('def ') and len(line) - len(line.lstrip()) == indent_level and len(forward_lines) > 5:
            break

print("\n1. FORWARD METHOD SIGNATURE:")
print("-" * 50)
for line_num, line in forward_lines[:10]:
    print(f"Line {line_num}: {line.rstrip()}")

print("\n2. KEY OPERATIONS IN FORWARD PASS:")
print("-" * 50)
keywords = ['render', 'ray', 'sample', 'network', 'implicit', 'volume', 'density', 'color']
for line_num, line in forward_lines:
    if any(kw in line.lower() for kw in keywords):
        print(f"Line {line_num}: {line.strip()[:80]}")

print("\n3. STAGE 1 FORWARD PASS FLOW:")
print("-" * 50)
print("""
Input: Batch with images, camera parameters, MANO parameters
  ↓
1. Ray Generation
   - Sample UV coordinates from image
   - Convert to 3D rays using camera intrinsics/extrinsics
   ↓
2. Ray Sampling (Volume Rendering)
   - Sample points along each ray
   - Query implicit networks at sample points
   ↓
3. Implicit Network Evaluation
   For each node (hand, object, background):
     - Query SDF network → density values
     - Query rendering network → RGB values
   ↓
4. Volume Rendering Integration
   - Integrate density along rays → alpha values
   - Integrate RGB * alpha → final pixel colors
   ↓
5. Loss Computation
   - RGB loss: Predicted vs GT pixel colors
   - Mask loss: Predicted vs GT silhouettes
   - Eikonal loss: SDF gradient regularization
   ↓
Output: Rendered images + loss values
""")
print("="*70)
