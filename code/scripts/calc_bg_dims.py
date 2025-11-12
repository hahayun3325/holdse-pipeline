# save as: calc_bg_dims.py
"""
Background Rendering Network Dimension Calculator

This script calculates the correct d_in value for bg_rendering_network
based on the actual inputs it will receive.
"""


def calculate_bg_rendering_dims(
        view_dirs_base_dim=3,
        feature_vectors_dim=256,
        frame_latent_dim=32,
        multires_view=4,
        has_frame_latent=True
):
    """
    Calculate required d_in for background rendering network.

    Background rendering receives:
    - view_dirs: base 3D, gets encoded if multires_view > 0
    - feature_vectors: typically 256D
    - frame_latent_code: typically 32D (optional)
    """

    print("=" * 70)
    print("BACKGROUND RENDERING NETWORK DIMENSION CALCULATOR")
    print("=" * 70)

    # Calculate view_dirs dimension after encoding
    if multires_view > 0:
        # Positional encoding: base + 2 * multires * base
        view_dirs_encoded_dim = view_dirs_base_dim + 2 * multires_view * view_dirs_base_dim
        print(f"\nView Direction Encoding:")
        print(f"  Base dimension: {view_dirs_base_dim}D")
        print(f"  multires_view: {multires_view}")
        print(
            f"  Encoded dimension: {view_dirs_base_dim} + 2*{multires_view}*{view_dirs_base_dim} = {view_dirs_encoded_dim}D")
    else:
        view_dirs_encoded_dim = view_dirs_base_dim
        print(f"\nView Direction: {view_dirs_base_dim}D (no encoding)")

    # Calculate total input dimension
    total_input_dim = view_dirs_encoded_dim + feature_vectors_dim
    if has_frame_latent:
        total_input_dim += frame_latent_dim

    print(f"\nTotal Input Dimension Calculation:")
    print(f"  view_dirs (encoded): {view_dirs_encoded_dim}D")
    print(f"  feature_vectors: {feature_vectors_dim}D")
    if has_frame_latent:
        print(f"  frame_latent_code: {frame_latent_dim}D")
    print(f"  " + "=" * 50)
    print(f"  TOTAL: {total_input_dim}D")

    # Calculate what d_in should be in config
    # The __init__ will add: (input_ch - 3) + dim_frame_encoding
    # So we need to reverse-engineer the base d_in
    multires_overhead = view_dirs_encoded_dim - view_dirs_base_dim

    if has_frame_latent:
        required_d_in = view_dirs_base_dim + feature_vectors_dim + frame_latent_dim
    else:
        required_d_in = view_dirs_base_dim + feature_vectors_dim

    print(f"\nConfiguration Required:")
    print(f"  bg_rendering_network:")
    print(f"    d_in: {required_d_in}  # Base dimension before adjustments")
    print(f"    multires_view: {multires_view}")
    print(f"    dim_frame_encoding: {frame_latent_dim}")
    print(f"    feature_vector_size: {feature_vectors_dim}")

    print(f"\nInitialization Calculation (in __init__):")
    print(f"  dims[0] starts at: {required_d_in}")
    if multires_view > 0:
        print(f"  + multires adjustment: {multires_overhead} (from {view_dirs_encoded_dim} - {view_dirs_base_dim})")
        print(f"  dims[0] after multires: {required_d_in + multires_overhead}")
    if has_frame_latent:
        print(f"  + dim_frame_encoding: {frame_latent_dim}")
        print(f"  dims[0] final: {required_d_in + multires_overhead + frame_latent_dim}")

    expected_final_dim = required_d_in + multires_overhead + (frame_latent_dim if has_frame_latent else 0)

    print(f"\n✅ Expected lin0 input dimension: {expected_final_dim}D")
    print(f"   (This should match the actual rendering_input dimension)")
    print("=" * 70)

    return required_d_in, expected_final_dim


if __name__ == "__main__":
    # Your current configuration
    print("\n\nCURRENT CONFIGURATION:")
    required_d_in, expected_final = calculate_bg_rendering_dims(
        view_dirs_base_dim=3,
        feature_vectors_dim=256,
        frame_latent_dim=32,
        multires_view=4,
        has_frame_latent=True
    )

    print(f"\n\nSUMMARY:")
    print(f"  Set in config: bg_rendering_network.d_in = {required_d_in}")
    print(f"  Network will expect: {expected_final}D input")
