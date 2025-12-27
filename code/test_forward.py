import torch
from src.model.ghop.diffusion import GHOP3DUNetWrapper

wrapper = GHOP3DUNetWrapper(
    unet_ckpt_path='/home/fredcui/Projects/ghop/output/joint_3dprior/mix_data/checkpoints/last.ckpt',
    device='cpu'
)

# Test forward pass
batch_size = 1
latent = torch.randn(batch_size, 3, 6, 6, 6)
timestep = torch.tensor([100])
context = torch.randn(batch_size, 1, 768)  # Text embedding

print("Testing forward pass...")
try:
    output = wrapper(latent, timestep, context)
    print(f"✅ Forward pass successful!")
    print(f"   Input shape: {latent.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Expected: {(batch_size, 3, 6, 6, 6)}")
    
    if output.shape == (batch_size, 3, 6, 6, 6):
        print("✅ Output shape matches expected!")
    else:
        print("❌ Output shape mismatch!")
except Exception as e:
    print(f"❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
