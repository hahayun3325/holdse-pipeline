import trimesh
import numpy as np
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')
# Load meshes from misc files
official_misc = np.load('/home/fredcui/Projects/hold-master/code/logs/cb20a1702/misc/000080000.npy', allow_pickle=True).item()
holdse_misc = np.load('logs/9a27219db/misc/000056000.npy', allow_pickle=True).item()

official_mesh = official_misc['mesh_c_o']
holdse_mesh = holdse_misc['object_cano']

print("="*70)
print("CANONICAL OBJECT MESH COMPARISON")
print("="*70)

# Basic stats
print(f"\nOfficial HOLD:")
print(f"  Vertices: {len(official_mesh.vertices)}")
print(f"  Faces: {len(official_mesh.faces)}")
print(f"  Bounds: {official_mesh.bounds}")
print(f"  Volume: {official_mesh.volume:.6f}")
print(f"  Centroid: {official_mesh.centroid}")

print(f"\nHOLDSE:")
print(f"  Vertices: {len(holdse_mesh.vertices)}")
print(f"  Faces: {len(holdse_mesh.faces)}")
print(f"  Bounds: {holdse_mesh.bounds}")
print(f"  Volume: {holdse_mesh.volume:.6f}")
print(f"  Centroid: {holdse_mesh.centroid}")

# Critical check: Are they in the same coordinate frame?
official_size = official_mesh.bounds[1] - official_mesh.bounds[0]
holdse_size = holdse_mesh.bounds[1] - holdse_mesh.bounds[0]

print(f"\nBounding box size:")
print(f"  Official: {official_size}")
print(f"  HOLDSE: {holdse_size}")
print(f"  Ratio: {holdse_size / official_size}")

# Check if they overlap at all
def boxes_overlap(bounds1, bounds2):
    return (bounds1[0] < bounds2[1]).all() and (bounds1[1] > bounds2[0]).all()

overlap = boxes_overlap(official_mesh.bounds, holdse_mesh.bounds)
print(f"\n{'✅' if overlap else '❌'} Bounding boxes overlap: {overlap}")

# Chamfer distance between canonical meshes
official_pts = official_mesh.sample(10000)
holdse_pts = holdse_mesh.sample(10000)

from scipy.spatial import cKDTree
tree_official = cKDTree(official_pts)
tree_holdse = cKDTree(holdse_pts)

dist_h2o, _ = tree_official.query(holdse_pts)
dist_o2h, _ = tree_holdse.query(official_pts)

cd_canonical = (dist_h2o.mean() + dist_o2h.mean()) / 2
print(f"\n❗ Chamfer distance (canonical space): {cd_canonical:.4f}")

if cd_canonical > 0.1:
    print("   ⚠️  SEVERE: Canonical meshes are completely different shapes!")
elif cd_canonical > 0.01:
    print("   ⚠️  WARNING: Canonical meshes differ significantly")
else:
    print("   ✅ Canonical meshes are similar")