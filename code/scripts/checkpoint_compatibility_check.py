# Save as: scripts/checkpoint_compatibility_check.py
import torch


def check_compatibility(official_path, stage1_path):
    """Quick compatibility check."""

    official = torch.load(official_path, map_location='cpu')
    stage1 = torch.load(stage1_path, map_location='cpu')

    print("=" * 80)
    print("CHECKPOINT COMPATIBILITY CHECK")
    print("=" * 80)

    # Critical rendering network dimensions
    critical_keys = [
        'model.nodes.right.rendering_network.lin0.weight',
        'model.nodes.object.rendering_network.lin0.weight',
        'model.background.bg_rendering_network.lin0.weight',
    ]

    compatible = True
    issues = []

    for key in critical_keys:
        off_shape = official['state_dict'].get(key, torch.tensor([])).shape
        s1_shape = stage1['state_dict'].get(key, torch.tensor([])).shape

        network = key.split('.')[2] if len(key.split('.')) > 2 else 'unknown'

        if off_shape != s1_shape:
            compatible = False
            if len(off_shape) > 1 and len(s1_shape) > 1:
                diff = off_shape[1] - s1_shape[1]
                issues.append(f"  ❌ {network}: {off_shape[1]} vs {s1_shape[1]} (diff: {diff})")
            else:
                issues.append(f"  ❌ {network}: Shape mismatch")
        else:
            print(f"  ✅ {network}: Compatible")

    if issues:
        print("\nINCOMPATIBLE LAYERS:")
        for issue in issues:
            print(issue)

        print("\n" + "=" * 80)
        print("RECOMMENDATION")
        print("=" * 80)
        print("❌ Official checkpoint is INCOMPATIBLE with current codebase")
        print("\nOptions:")
        print("  1. Modify code to match official dimensions (HIGH RISK)")
        print("  2. Create config matching official architecture (COMPLEX)")
        print("  3. Use Stage 1 checkpoint (RECOMMENDED)")
        print("\nTo proceed with option 1 or 2, you'll need to:")
        print("  - Disable automatic dimension additions in code")
        print("  - Create new config with d_in values from official checkpoint")
        print("  - Test loading with strict=False")
        print("  - Expect additional compatibility issues")
    else:
        print("\n✅ Checkpoints are compatible!")

    return compatible


check_compatibility(
    '/home/fredcui/Projects/hold/code/logs/cb20a1702/checkpoints/last.ckpt',
    'logs/140dc5c18/checkpoints/last.ckpt'
)