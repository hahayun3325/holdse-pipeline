# Save as: scripts/reverse_engineer_config.py
import torch


def reverse_engineer_config(ckpt_path, name="Checkpoint"):
    """Attempt to infer configuration from checkpoint."""

    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt['state_dict']

    print(f"\n{'=' * 80}")
    print(f"{name} - Inferred Configuration")
    print('=' * 80)

    config = {}

    # Implicit network
    impl_keys = [k for k in sd.keys() if 'implicit_network.lin0.weight' in k and 'right' in k]
    if impl_keys:
        key = impl_keys[0]
        config['implicit_d_in'] = sd[key].shape[1]
        config['implicit_d_out'] = sd[key].shape[0]

        # Count layers
        n_layers = len([k for k in sd.keys() if 'implicit_network.lin' in k and 'right' in k and '.weight' in k])
        config['implicit_n_layers'] = n_layers

    # Rendering network
    rnet_keys = [k for k in sd.keys() if 'rendering_network.lin0.weight' in k and 'right' in k]
    if rnet_keys:
        key = rnet_keys[0]
        config['rendering_d_in'] = sd[key].shape[1]
        config['rendering_d_out'] = sd[key].shape[0]

        n_layers = len([k for k in sd.keys() if 'rendering_network.lin' in k and 'right' in k and '.weight' in k])
        config['rendering_n_layers'] = n_layers

    # Frame latent
    frame_key = 'model.nodes.object.frame_latent_encoder.weight'
    if frame_key in sd:
        config['n_frames'] = sd[frame_key].shape[0]
        config['latent_dim'] = sd[frame_key].shape[1]

    # Print
    print("\nmodel:")
    print("  implicit_network:")
    print(f"    d_in: {config.get('implicit_d_in', 'N/A')}")
    print(f"    dims: [...{config.get('implicit_n_layers', 'N/A')} layers...]")

    print("\n  rendering_network:")
    print(f"    d_in: {config.get('rendering_d_in', 'N/A')}")
    print(f"    dims: [...{config.get('rendering_n_layers', 'N/A')} layers...]")

    print("\n  frame_encoding:")
    print(f"    n_frames: {config.get('n_frames', 'N/A')}")
    print(f"    latent_dim: {config.get('latent_dim', 'N/A')}")

    return config


# Compare configs
official_config = reverse_engineer_config('/home/fredcui/Projects/hold/code/logs/cb20a1702/checkpoints/last.ckpt',
                                          'OFFICIAL')
stage1_config = reverse_engineer_config('logs/140dc5c18/checkpoints/last.ckpt', 'STAGE 1')