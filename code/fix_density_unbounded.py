#!/usr/bin/env python3
"""Fix unbounded density values"""

import re

print("=" * 80)
print("FINDING WHERE DENSITY SHOULD BE CLAMPED")
print("=" * 80)
print()

files_to_check = [
    'src/engine/density.py',
    'src/networks/rendering_network.py',
    'src/model/mano/deformer.py',
    'src/hold/hold.py'
]

for filepath in files_to_check:
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        print(f"\nChecking {filepath}...")
        print("-" * 80)
        
        for i, line in enumerate(lines, 1):
            if 'density' in line.lower() and '=' in line:
                # Found a density assignment
                if 'sigmoid' in line or 'clamp' in line or 'relu' in line:
                    print(f"✓ Line {i}: SAFE (has bounding)")
                    print(f"  {line.strip()}")
                else:
                    print(f"✗ Line {i}: POTENTIALLY UNSAFE (no bounding)")
                    print(f"  {line.strip()}")
                    
                    # Show context
                    print(f"  Context:")
                    for j in range(max(0, i-3), min(len(lines), i+3)):
                        print(f"    {j+1}: {lines[j].rstrip()}")
                    print()
    
    except FileNotFoundError:
        pass

