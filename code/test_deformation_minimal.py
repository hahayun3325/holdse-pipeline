import torch
from src.model.mano.server import MANOServer
from src.model.mano.deformer import MANODeformer

# Setup
betas = torch.zeros(1, 10)
server = MANOServer(betas=betas, is_rhand=True)
deformer = MANODeformer(max_dist=2.0, K=15, betas=betas, is_rhand=True)

# Get posed mesh (world space)
scene_scale = torch.tensor([4.6947])
transl = torch.tensor([[0.0, 0.5, 0.0]])
pose = torch.randn(1, 48) * 0.1  # Small random pose

output = server(scene_scale, transl, pose, betas)
verts_world = output['verts']
tfs = output['tfs']

print(f"World verts range: [{verts_world.min():.4f}, {verts_world.max():.4f}]")

# Test inverse deformation
# Query points near finger tips in world space
test_points = verts_world[0, :5].unsqueeze(0)  # First 5 vertices
print(f"Test points (world): {test_points[0, :3]}")

# Deform to canonical
canonical_points, outliers = deformer(test_points, tfs, inverse=True, verts=verts_world)
print(f"Canonical points: {canonical_points[0, :3]}")
print(f"Expected: ~[-0.1, 0.1]")
print(f"Outliers: {outliers.sum()}/{outliers.numel()}")

# Verify round-trip
world_points_back, _ = deformer(canonical_points, tfs, inverse=False, verts=verts_world)
error = (world_points_back - test_points).abs().mean()
print(f"Round-trip error: {error:.6f} (should be < 0.001)")
