# Save as: scripts/verify_8layer_architecture.py
import torch
from omegaconf import OmegaConf
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')

from src.hold.hold import HOLD

print("=" * 80)
print("VERIFYING 8-LAYER ARCHITECTURE")
print("=" * 80)

# Load config
config_path = 'confs/stage1_hold_MC1_ho3d_8layer_implicit.yaml'
config = OmegaConf.load(config_path)

# Check for missing keys that might cause errors
print("\nChecking config completeness...")
required_keys = [
    'model.scene_bounding_sphere',
    'model.implicit_network.dims',
    'model.rendering_network.dims',
]

missing_keys = []
for key in required_keys:
    try:
        OmegaConf.select(config, key)
        print(f"  ✅ {key}")
    except:
        print(f"  ❌ {key} - MISSING!")
        missing_keys.append(key)

if missing_keys:
    print("\n" + "=" * 80)
    print("❌ CONFIG INCOMPLETE - Add missing keys before training:")
    print("=" * 80)
    for key in missing_keys:
        if 'scene_bounding_sphere' in key:
            print(f"  {key}: 3.0")
    print("\nExiting verification - fix config first.")
    exit(1)

# Add missing keys with defaults if needed
if not hasattr(config.model, 'scene_bounding_sphere'):
    print("\n⚠️  Auto-adding scene_bounding_sphere: 3.0")
    config.model.scene_bounding_sphere = 3.0

# Create dummy args
class Args:
    case = 'hold_MC1_ho3d'
    n_images = 144
    num_sample = 2048
    no_vis = False
    render_downsample = 1
    freeze_pose = False
    experiment = 'verify'
    log_every = 10
    log_dir = 'logs/verify'
    barf_s = 0
    barf_e = 0
    no_barf = True
    shape_init = ""
    exp_key = 'verify'
    debug = False


args = Args()

# Initialize model
print("\nInitializing model...")
try:
    model = HOLD(config, args)
    print("✅ Model initialized successfully!")
except Exception as e:
    print(f"❌ Model initialization failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Check network depths
print("\n" + "=" * 80)
print("NETWORK ARCHITECTURE")
print("=" * 80)


# Count layers
def count_layers(module, name):
    """Count only hidden layers (exclude output layer)."""
    layer_count = 0
    for child_name, child in module.named_children():
        if 'lin' in child_name:
            if child_name not in ['lin_out', 'lin_pose']:  # Exclude output and pose
                layer_count += 1

    # Subtract 1 for the final output layer (lin_final or highest numbered lin)
    max_lin = -1
    for child_name in module.named_children():
        if 'lin' in child_name[0] and child_name[0][3:].isdigit():
            max_lin = max(max_lin, int(child_name[0][3:]))

    if max_lin > 0:
        hidden_layers = max_lin  # lin0 to lin(n-1) are hidden, lin(n) is output
    else:
        hidden_layers = layer_count - 1  # Fallback: total - 1

    print(f"{name:<30} {hidden_layers} hidden layers (+ 1 output)")
    return hidden_layers


# Check foreground networks
if hasattr(model, 'model') and hasattr(model.model, 'nodes'):
    if hasattr(model.model.nodes, 'right'):
        right_impl_layers = count_layers(
            model.model.nodes.right.implicit_network,
            "Right Implicit"
        )
        right_rend_layers = count_layers(
            model.model.nodes.right.rendering_network,
            "Right Rendering"
        )

        if right_impl_layers == 8:
            print("  ✅ Right implicit has 8 layers (correct!)")
        else:
            print(f"  ❌ Right implicit has {right_impl_layers} layers (should be 8)")

        if right_rend_layers == 4:
            print("  ✅ Right rendering has 4 layers (correct!)")
        else:
            print(f"  ⚠️  Right rendering has {right_rend_layers} layers (should be 4)")

    if hasattr(model.model.nodes, 'object'):
        obj_impl_layers = count_layers(
            model.model.nodes.object.implicit_network,
            "Object Implicit"
        )
        obj_rend_layers = count_layers(
            model.model.nodes.object.rendering_network,
            "Object Rendering"
        )

        if obj_impl_layers == 8:
            print("  ✅ Object implicit has 8 layers (correct!)")
        else:
            print(f"  ❌ Object implicit has {obj_impl_layers} layers (should be 8)")

        if obj_rend_layers == 4:
            print("  ✅ Object rendering has 4 layers (correct!)")
        else:
            print(f"  ⚠️  Object rendering has {obj_rend_layers} layers (should be 4)")

# Check background network
if hasattr(model, 'model') and hasattr(model.model, 'background'):
    bg_impl_layers = count_layers(
        model.model.background.bg_implicit_network,
        "Background Implicit"
    )

    if bg_impl_layers == 8:
        print("  ✅ Background implicit has 8 layers (correct!)")
    else:
        print(f"  ⚠️  Background implicit has {bg_impl_layers} layers (should be 8)")

print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
print("\nIf all checks pass (✅), architecture is correct!")
print("Start training with: ./scripts/train_stage1_8layer.sh")
print("=" * 80)