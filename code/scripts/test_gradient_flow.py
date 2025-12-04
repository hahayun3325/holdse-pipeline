import torch
import sys
import os
# Add the code directory to path
sys.path.insert(0, os.path.abspath('/home/fredcui/Projects/holdse/code'))

# Try to find 'common' directory
if os.path.exists('/home/fredcui/Projects/holdse/code/src/common'):
    sys.path.insert(0, '/home/fredcui/Projects/holdse/code/src')
elif os.path.exists('/home/fredcui/Projects/holdse/code/common'):
    pass # Already in path
else:
    print("⚠️ Warning: Could not find 'common' directory")
from src.model.mano.params import MANOParams
from src.model.mano.server import MANOServer

print("="*60)
print("TESTING MANO GRADIENT FLOW")
print("="*60)

# Create params
params = MANOParams(
    2,  # Just 2 frames
    {
        "betas": 10,
        "global_orient": 3,
        "transl": 3,
        "pose": 45,
    },
    "right",
)
params.load_params("hold_MC1_ho3d")

# Check params
print("\n1. Parameter requires_grad:")
for name in ['pose', 'global_orient', 'transl']:
    param = getattr(params, name).weight
    print(f"   {name}: {param.requires_grad}")
    if not param.requires_grad:
        print(f"   ❌ ERROR: {name} has requires_grad=False!")
        sys.exit(1)

print("   ✅ All params have requires_grad=True")

# Create server
betas_init = params.betas.weight[0].detach()
server = MANOServer(betas=betas_init, is_rhand=True)

# Test forward pass
print("\n2. Testing forward pass:")

scene_scale = torch.ones(1, 1, 1).cuda()
transl = params.transl.weight[0:1].cuda()  # [1, 3]
pose = params.pose.weight[0:1].cuda()  # [1, 45]
orient = params.global_orient.weight[0:1].cuda()  # [1, 3]
betas = params.betas.weight.cuda()  # [1, 10]

full_pose = torch.cat([orient, pose], dim=1)  # [1, 48]

print(f"   Inputs require_grad:")
print(f"     transl: {transl.requires_grad}")
print(f"     full_pose: {full_pose.requires_grad}")

# Forward
output = server.forward(scene_scale, transl, full_pose, betas)

print(f"\n   Outputs require_grad:")
print(f"     verts: {output['verts'].requires_grad}")
print(f"     jnts: {output['jnts'].requires_grad}")

if not output['verts'].requires_grad:
    print(f"   ❌ ERROR: Server output has no gradients!")
    sys.exit(1)

# Test backward
print("\n3. Testing backward pass:")

loss = output['verts'].sum()
loss.backward()

print(f"   Gradients computed:")
print(f"     transl.grad: {transl.grad is not None}")
print(f"     pose.grad: {pose.grad is not None}")

if transl.grad is not None:
    print(f"     transl.grad mean: {transl.grad.abs().mean():.6f}")
if pose.grad is not None:
    print(f"     pose.grad mean: {pose.grad.abs().mean():.6f}")

if transl.grad is None or pose.grad is None:
    print(f"\n❌ GRADIENTS NOT COMPUTED!")
    print(f"   Check for .detach() or torch.no_grad() in code")
    sys.exit(1)

print("\n" + "="*60)
print("✅ SUCCESS! Gradients flow to MANO parameters!")
print("="*60)