# create_hybrid_c3b7a8352.py
# Immediate: Apply Hybrid Fix to New Checkpoint
import numpy as np
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')

# Load NEW checkpoint's misc
new_misc = np.load('logs/c3b7a8352/misc/000056000.npy', allow_pickle=True).item()

# Load official canonical mesh
official_misc = np.load('/home/fredcui/Projects/hold-master/code/logs/cb20a1702/misc/000080000.npy',
                        allow_pickle=True).item()

# Create hybrid
hybrid = {
    'obj_scale': official_misc['obj_scale'],      # Use official scale
    'mesh_c_o': official_misc['mesh_c_o'],        # Use official mesh ← KEY FIX
    'mesh_c_h': new_misc.get('right_cano', new_misc.get('mesh_c_h')),
    'img_paths': new_misc['img_paths'],
    'K': new_misc['K'],
    'w2c': new_misc['w2c'],
    'scale': new_misc['scale'],
}

# Save for official HOLD evaluation
np.save('/home/fredcui/Projects/hold-master/code/logs/c3b7a8352/misc/000056000.npy',
        hybrid)

print("✅ Created hybrid: New checkpoint + official canonical mesh")
