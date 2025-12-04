# Save as: scripts/compare_network_topology.py
import torch


def get_network_topology(ckpt_path, network_prefix):
    """Extract layer structure for a specific network."""

    ckpt = torch.load(ckpt_path, map_location='cpu')

    layers = []
    for key in sorted(ckpt['state_dict'].keys()):
        if network_prefix in key and '.weight' in key and 'lin' in key:
            shape = ckpt['state_dict'][key].shape
            layer_name = key.split('.')[-2]
            layers.append((layer_name, shape))

    return layers


networks = [
    ('Right Hand Rendering', 'model.nodes.right.rendering_network'),
    ('Object Rendering', 'model.nodes.object.rendering_network'),
    ('Background Rendering', 'model.background.bg_rendering_network'),
]

for network_name, prefix in networks:
    print(f"\n{'=' * 80}")
    print(f"{network_name}")
    print('=' * 80)

    official_topo = get_network_topology('/home/fredcui/Projects/hold/code/logs/cb20a1702/checkpoints/last.ckpt',
                                         prefix)
    stage1_topo = get_network_topology('logs/140dc5c18/checkpoints/last.ckpt', prefix)

    print(f"\n{'Layer':<15} {'Official Shape':<25} {'Stage 1 Shape':<25} {'Match'}")
    print("-" * 80)

    max_len = max(len(official_topo), len(stage1_topo))
    for i in range(max_len):
        off_layer = official_topo[i] if i < len(official_topo) else ('N/A', 'N/A')
        s1_layer = stage1_topo[i] if i < len(stage1_topo) else ('N/A', 'N/A')

        match = "✅" if off_layer[1] == s1_layer[1] else "❌"

        print(f"{off_layer[0]:<15} {str(off_layer[1]):<25} {str(s1_layer[1]):<25} {match}")