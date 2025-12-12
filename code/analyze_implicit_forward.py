#!/usr/bin/env python3
"""
Extract and analyze ImplicitNet forward pass.
"""

print("="*70)
print("IMPLICIT NETWORK FORWARD PASS ANALYSIS")
print("="*70)

with open('src/networks/shape_net.py', 'r') as f:
    content = f.read()

# Find ImplicitNet class
import re
class_match = re.search(r'class ImplicitNet\(.*?\):(.*?)(?=\nclass |\Z)', content, re.DOTALL)

if class_match:
    class_content = class_match.group(1)
    
    # Find forward method
    forward_match = re.search(r'def forward\(self.*?\):(.*?)(?=\n    def |\Z)', class_content, re.DOTALL)
    
    if forward_match:
        print("\n1. FORWARD METHOD IMPLEMENTATION:")
        print("-" * 70)
        forward_code = forward_match.group(0)
        
        # Print with line numbers
        for i, line in enumerate(forward_code.split('\n'), 1):
            print(f"{i:3d}: {line}")
        
        print("\n2. KEY OPERATIONS:")
        print("-" * 70)
        
        # Extract key operations
        operations = []
        for line in forward_code.split('\n'):
            if 'embedder' in line.lower():
                operations.append(('POSITIONAL_ENCODING', line.strip()))
            elif 'lin' in line and '=' in line:
                operations.append(('LINEAR_LAYER', line.strip()))
            elif 'skip' in line.lower():
                operations.append(('SKIP_CONNECTION', line.strip()))
            elif 'softplus\|relu\|activation' in line.lower():
                operations.append(('ACTIVATION', line.strip()))
            elif 'return' in line:
                operations.append(('OUTPUT', line.strip()))
        
        for op_type, op_line in operations:
            print(f"  [{op_type:20s}] {op_line}")
        
        print("\n3. INPUT/OUTPUT ANALYSIS:")
        print("-" * 70)
        print("Expected inputs:")
        print("  - pts: (B, N, 3) 3D point coordinates")
        print("  - cond: (B, pose_dim) Conditioning vector (MANO pose)")
        print("")
        print("Expected outputs:")
        print("  - sdf: (B, N, 1) Signed distance values")
        print("  - features: (B, N, 256) Feature vectors for rendering")
        
    else:
        print("Forward method not found!")
else:
    print("ImplicitNet class not found!")

print("="*70)
