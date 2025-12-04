# Save as: scripts/extract_rendering_dims.py
import torch


def extract_rendering_dims(ckpt_path, name="Checkpoint"):
    """Extract d_in for all rendering networks."""

    ckpt = torch.load(ckpt_path, map_location='cpu')

    print(f"\n{'=' * 80}")
    print(f"{name} - Rendering Network Input Dimensions")
    print('=' * 80)

    # Rendering networks to check
    networks = [
        ('Right Hand', 'model.nodes.right.rendering_network.lin0.weight'),
        ('Object', 'model.nodes.object.rendering_network.lin0.weight'),
        ('Background', 'model.background.bg_rendering_network.lin0.weight'),
    ]

    dims = {}
    for network_name, key in networks:
        if key in ckpt['state_dict']:
            weight = ckpt['state_dict'][key]
            d_in = weight.shape[1]  # Input dimension
            d_out = weight.shape[0]  # Output dimension
            dims[network_name] = d_in
            print(f"{network_name:<20} d_in={d_in:<6} d_out={d_out:<6} shape={weight.shape}")
        else:
            print(f"{network_name:<20} NOT FOUND")

    return dims


# Compare both
official_dims = extract_rendering_dims('/home/fredcui/Projects/hold/code/logs/cb20a1702/checkpoints/last.ckpt',
                                       'OFFICIAL')
stage1_dims = extract_rendering_dims('logs/140dc5c18/checkpoints/last.ckpt', 'STAGE 1')

print(f"\n{'=' * 80}")
print("DIMENSION DIFFERENCES")
print('=' * 80)
for key in official_dims:
    if key in stage1_dims:
        diff = official_dims[key] - stage1_dims[key]
        ratio = official_dims[key] / stage1_dims[key]
        print(f"{key:<20} Official: {official_dims[key]:<6} Stage 1: {stage1_dims[key]:<6} "
              f"Diff: {diff:<6} Ratio: {ratio:.2f}x")