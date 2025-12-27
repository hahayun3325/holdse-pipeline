import torch
from src.model.ghop.diffusion import GHOP3DUNetWrapper

# Clear GPU memory
torch.cuda.empty_cache()

wrapper = GHOP3DUNetWrapper(
    unet_ckpt_path='/home/fredcui/Projects/ghop/output/joint_3dprior/mix_data/checkpoints/last.ckpt',
    device='cuda'
)

# Test forward pass on GPU
batch_size = 2
latent = torch.randn(batch_size, 3, 6, 6, 6, device='cuda')
timestep = torch.tensor([100, 200], device='cuda')
context = torch.randn(batch_size, 1, 768, device='cuda')

print("Before forward pass:")
print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

output = wrapper(latent, timestep, context)

print("After forward pass:")
print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
print(f"  Peak: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

# Expected: ~500 MB if 100% loaded, ~1200 MB if 77% loaded
