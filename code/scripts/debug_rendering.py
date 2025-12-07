import torch
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint')
    parser.add_argument('--frame', type=int, default=0, help='Frame to debug')
    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print(f"CHECKPOINT DEBUG: {Path(args.checkpoint).name}")
    print(f"{'=' * 70}\n")

    ckpt = torch.load(args.checkpoint, map_location='cpu')

    print("[1] Top-level keys:")
    for key in ckpt.keys():
        print(f"  - {key}")

    print("\n[2] Hyperparameters:")
    if 'hyper_parameters' in ckpt:
        hp = ckpt['hyper_parameters']
        if 'scene_bounding_sphere' in hp:
            print(f"  scene_bounding_sphere: {hp['scene_bounding_sphere']}")
        if 'sdf_bounding_sphere' in hp:
            print(f"  sdf_bounding_sphere: {hp['sdf_bounding_sphere']}")
        if 'model' in hp and 'scene_bounding_sphere' in hp['model']:
            print(f"  model.scene_bounding_sphere: {hp['model']['scene_bounding_sphere']}")
    else:
        print("  ⚠️  No 'hyper_parameters' key found!")
        print("  Checkpoint may be from manual save or different framework")

    print("\n[3] Training metadata:")
    print(f"  Epoch: {ckpt.get('epoch', 'not found')}")
    print(f"  Global step: {ckpt.get('global_step', 'not found')}")

    print("\n[4] Object SDF network check:")
    object_keys = [k for k in ckpt['state_dict'].keys() if 'object' in k and 'implicit_network' in k]
    if object_keys:
        print(f"  Found {len(object_keys)} object implicit network parameters")

        # Find a weight tensor (skip scalars)
        weight_key = None
        for key in object_keys:
            param = ckpt['state_dict'][key]
            if param.ndim >= 1 and param.dtype in [torch.float32, torch.float16]:  # ← Skip scalars and non-float
                weight_key = key
                break

        if weight_key:
            sample_weight = ckpt['state_dict'][weight_key]
            print(f"  Sample weight '{weight_key}':")
            print(f"    Shape: {sample_weight.shape}")
            print(f"    Mean: {sample_weight.mean():.6f}")
            print(f"    Std: {sample_weight.std():.6f}")

            # Check if weights are near initialization
            if sample_weight.std() < 0.01:
                print("    ⚠️  Warning: Low std suggests network may be untrained!")
            else:
                print("    ✓ Network appears trained (std > 0.01)")
        else:
            print("  ⚠️ Could not find a suitable weight tensor to analyze")
    else:
        print("  ❌ No object implicit network found!")
        print("  This explains why object is invisible!")

    print(f"\n{'=' * 70}\n")


if __name__ == '__main__':
    main()

'''
python scripts/debug_rendering.py \
    --checkpoint /home/fredcui/Projects/hold/code/logs/cb20a1702/checkpoints/last.ckpt
'''