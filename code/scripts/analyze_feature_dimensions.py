# Save as: scripts/analyze_feature_dimensions.py
import torch


def analyze_features(ckpt_path, name="Checkpoint"):
    """Reverse-engineer feature composition from dimensions."""

    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt['state_dict']

    print(f"\n{'=' * 80}")
    print(f"{name} - Feature Vector Analysis")
    print('=' * 80)

    # Get rendering network input dimension
    rnet_key = 'model.nodes.right.rendering_network.lin0.weight'
    if rnet_key not in sd:
        print("❌ Rendering network not found")
        return

    d_in = sd[rnet_key].shape[1]
    print(f"Total d_in: {d_in}")

    # Get implicit network output (geometric features)
    impl_keys = [k for k in sd.keys() if 'implicit_network' in k and 'right' in k]
    if impl_keys:
        # Look for output layer
        out_key = [k for k in impl_keys if 'out.weight' in k or 'lin_out.weight' in k]
        if out_key:
            feature_dim = sd[out_key[0]].shape[0]
            print(f"\nGeometric features: {feature_dim}")

    # Try to identify components
    print(f"\nEstimated breakdown:")
    print(f"  xyz coordinates:    3")

    # Check if there's a positional encoding module
    pe_keys = [k for k in sd.keys() if 'embedder' in k.lower() or 'encoding' in k.lower()]
    if pe_keys:
        print(f"  Positional encoding: Found ({len(pe_keys)} params)")

    # Frame latent code dimension
    frame_enc_keys = [k for k in sd.keys() if 'frame_latent' in k]
    if frame_enc_keys:
        for k in frame_enc_keys:
            if 'weight' in k:
                dim = sd[k].shape[1] if len(sd[k].shape) > 1 else sd[k].shape[0]
                print(f"  Frame encoding:     {dim}")

    # Remaining dimensions
    remaining = d_in - 3
    print(f"  Other features:     {remaining}")
    print(f"\n  Total:              {d_in}")


# Analyze both
analyze_features('/home/fredcui/Projects/hold/code/logs/cb20a1702/checkpoints/last.ckpt', 'OFFICIAL')
analyze_features('logs/140dc5c18/checkpoints/last.ckpt', 'STAGE 1')