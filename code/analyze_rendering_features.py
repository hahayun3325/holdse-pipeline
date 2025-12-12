#!/usr/bin/env python3
"""
Analyze how RenderingNet combines features with view directions.
"""

print("="*70)
print("RENDERING NETWORK FEATURE COMBINATION ANALYSIS")
print("="*70)

# Read RenderingNet implementation
with open('src/networks/texture_net.py', 'r') as f:
    lines = f.readlines()

# Find forward method
in_forward = False
forward_lines = []
indent_level = 0

for i, line in enumerate(lines):
    if 'def forward(' in line and 'RenderingNet' in ''.join(lines[max(0,i-20):i]):
        in_forward = True
        indent_level = len(line) - len(line.lstrip())
        print("\n1. FORWARD METHOD SIGNATURE:")
        print("-" * 50)
    
    if in_forward:
        forward_lines.append((i+1, line.rstrip()))
        
        # Stop at next method or class
        if line.strip().startswith('def ') and len(forward_lines) > 5:
            curr_indent = len(line) - len(line.lstrip())
            if curr_indent == indent_level:
                break

# Print forward method
for line_num, line in forward_lines[:60]:
    print(f"{line_num:4d}: {line}")

print("\n2. INPUT DIMENSIONS ANALYSIS:")
print("-" * 50)
print("""
From __init__ debug output:
  opt.d_in: Base input dimension
  + feature_vector_size: Features from ImplicitNet
  + multires_view encoding: View direction positional encoding
  + pose embedding (if mode='pose'): MANO pose features
  
  Total d_in = sum of all above
""")

print("\n3. FEATURE FLOW:")
print("-" * 50)
print("""
ImplicitNet forward() → features (256-dim) + SDF (1-dim)
                         ↓
        Concatenate: [features, view_dirs, pose_embed]
                         ↓
              RenderingNet forward()
                         ↓
                   RGB output (3-dim)
""")

print("\n4. KEY QUESTIONS TO ANSWER:")
print("-" * 50)
print("- Where does feature_vector_size come from?")
print("- How are view directions encoded?")
print("- What is the pose embedding dimension?")
print("- Are features concatenated or added?")

print("="*70)
