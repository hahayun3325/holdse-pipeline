import numpy as np
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')

misc = np.load('logs/9a27219db/misc/000056000.npy', allow_pickle=True).item()

# Create compatible version
misc_fixed = {
    'obj_scale': misc.get('object.obj_scale', misc.get('obj_scale')),
    'mesh_c_o': misc.get('object_cano', misc.get('mesh_c_o')),
    'mesh_c_h': misc.get('right_cano', misc.get('mesh_c_h')),
    'img_paths': misc['img_paths'],
    'K': misc['K'],
    'w2c': misc['w2c'],
    'scale': misc['scale'],
}

print(f"obj_scale value: {misc_fixed['obj_scale']}")
np.save('logs/9a27219db/misc/000056000_fixed.npy', misc_fixed)