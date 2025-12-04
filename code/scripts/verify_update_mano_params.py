import torch

# Fixed comparison script (handles Long types)
# ckpt1 = torch.load('logs/7ed8871fe/checkpoints/last.ckpt')  # 1 epoch
ckpt1 = torch.load('logs/d839b2738/checkpoints/last.ckpt')  # 70 epoch
ckpt70 = torch.load('/home/fredcui/Projects/hold/code/logs/cb20a1702/checkpoints/last.ckpt')  # Official Checkpoint

print("=" * 60)
print("VERIFYING MANO PARAMETER UPDATES")
print("=" * 60)

# Focus on MANO parameters
mano_keys = [
    'model.nodes.right.params.pose.weight',
    'model.nodes.right.params.global_orient.weight',
    'model.nodes.right.params.transl.weight',
    'model.nodes.right.params.betas.weight',
]

for key in mano_keys:
    param1 = ckpt1['state_dict'][key].float()  # Convert to float
    param70 = ckpt70['state_dict'][key].float()

    diff = torch.abs(param1 - param70).mean().item()
    rel_diff = diff / (torch.abs(param1).mean().item() + 1e-10)

    print(f"\n{key.split('.')[-2]}:")
    print(f"  Absolute change: {diff:.6f}")
    print(f"  Relative change: {rel_diff:.2%}")

    if diff > 1e-4:
        print(f"  ✅ UPDATED (changed significantly)")
    else:
        print(f"  ❌ FROZEN (no change)")

print("\n" + "=" * 60)

'''
Step 1: Verify MANO Parameters Actually Updated
'''