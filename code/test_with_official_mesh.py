# test_with_official_mesh.py
import numpy as np
import shutil
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')

# Load HOLDSE misc
holdse_misc = np.load('logs/9a27219db/misc/000056000.npy', allow_pickle=True).item()

# Load official misc
official_misc = np.load('/home/fredcui/Projects/hold-master/code/logs/cb20a1702/misc/000080000.npy', allow_pickle=True).item()

# Create hybrid: HOLDSE params + official canonical mesh
hybrid_misc = holdse_misc.copy()
hybrid_misc['mesh_c_o'] = official_misc['mesh_c_o']  # Use official mesh
hybrid_misc['obj_scale'] = official_misc['obj_scale']  # Use official scale

# Remove HOLDSE-specific keys if they exist
hybrid_misc.pop('object_cano', None)
hybrid_misc.pop('object.obj_scale', None)

# Save
np.save('logs/9a27219db/misc/000056000_hybrid.npy', hybrid_misc)

# Copy to official HOLD for evaluation
shutil.copy('logs/9a27219db/misc/000056000_hybrid.npy',
            '/home/fredcui/Projects/hold-master/code/logs/holdse_test/misc/000056000.npy')

print("✅ Created hybrid: HOLDSE hand + official object mesh")
print("   Evaluate to see if object metrics improve")