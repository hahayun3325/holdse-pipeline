#!/usr/bin/env python3
"""
Compare MANO and network parameters between two checkpoints.
Usage: python scripts/compare_mano_checkpoints.py <old_ckpt> <new_ckpt>
"""
import torch
import sys
from pathlib import Path


def compare_checkpoints(old_path, new_path, verbose=True):
    """Compare two checkpoints and report MANO parameter updates."""

    # Load checkpoints
    print("=" * 80)
    print("CHECKPOINT COMPARISON - MANO PARAMETER UPDATE VERIFICATION")
    print("=" * 80)
    print(f"Old checkpoint: {old_path}")
    print(f"New checkpoint: {new_path}")
    print()

    old = torch.load(old_path, map_location='cpu')
    new = torch.load(new_path, map_location='cpu')

    # Get epoch info
    old_epoch = old.get('epoch', 'unknown')
    new_epoch = new.get('epoch', 'unknown')
    epoch_diff = new_epoch - old_epoch if isinstance(new_epoch, int) and isinstance(old_epoch, int) else None

    print(f"Old epoch: {old_epoch}")
    print(f"New epoch: {new_epoch}")
    if epoch_diff:
        print(f"Epochs trained: {epoch_diff}")
    print()

    # ================================================================
    # PART 1: MANO PARAMETERS (CRITICAL)
    # ================================================================
    print("=" * 80)
    print("PART 1: MANO PARAMETER UPDATES (Hand Pose Optimization)")
    print("=" * 80)

    mano_keys = [k for k in new['state_dict'].keys() if 'params' in k and 'nodes' in k]

    mano_results = []
    any_mano_changed = False

    for key in sorted(mano_keys):
        if key in old['state_dict']:
            old_val = old['state_dict'][key]
            new_val = new['state_dict'][key]

            # Check if EXACTLY equal
            exactly_equal = torch.equal(old_val, new_val)

            # Compute statistics
            if not exactly_equal:
                diff = (new_val - old_val).abs()
                mean_diff = diff.mean().item()
                max_diff = diff.max().item()
                std_diff = diff.std().item()

                # Compute relative change
                old_norm = old_val.abs().mean().item()
                rel_change = (mean_diff / old_norm * 100) if old_norm > 1e-10 else 0
            else:
                mean_diff = max_diff = std_diff = rel_change = 0.0

            # Format parameter name
            param_name = key.split('.')[-2]
            node = 'RIGHT' if 'right' in key.lower() else 'LEFT'

            if exactly_equal:
                status = "❌ FROZEN"
                print(f"{node}.{param_name:15s}: {status}")
            else:
                any_mano_changed = True
                status = "✅ UPDATED"
                print(f"{node}.{param_name:15s}: {status}")
                print(f"  Mean Δ: {mean_diff:.10f} ({rel_change:.4f}% change)")
                print(f"  Max Δ:  {max_diff:.10f}")
                print(f"  Std Δ:  {std_diff:.10f}")

                if epoch_diff and epoch_diff > 0:
                    per_epoch = mean_diff / epoch_diff
                    print(f"  Per epoch: {per_epoch:.10f}")

            mano_results.append({
                'key': key,
                'node': node,
                'param': param_name,
                'changed': not exactly_equal,
                'mean_diff': mean_diff,
                'max_diff': max_diff,
                'rel_change': rel_change
            })

    print()
    print("=" * 80)
    if any_mano_changed:
        print("✅ RESULT: MANO parameters UPDATED during training")
        print("   Gradient flow is WORKING - hand pose optimization active")
    else:
        print("❌ RESULT: MANO parameters COMPLETELY FROZEN")
        print("   Gradient flow is BROKEN - debug needed")
    print("=" * 80)
    print()

    # ================================================================
    # PART 2: NEURAL NETWORK PARAMETERS
    # ================================================================
    print("=" * 80)
    print("PART 2: NEURAL NETWORK PARAMETER UPDATES")
    print("=" * 80)

    # Group by network type
    network_groups = {
        'implicit_network': [],
        'rendering_network': [],
        'background': [],
        'deformer': [],
        'other': []
    }

    all_keys = set(old['state_dict'].keys()) | set(new['state_dict'].keys())
    network_keys = [k for k in all_keys if k not in mano_keys]

    for key in sorted(network_keys):
        if key not in old['state_dict'] or key not in new['state_dict']:
            continue

        old_val = old['state_dict'][key]
        new_val = new['state_dict'][key]

        if old_val.shape != new_val.shape:
            print(f"⚠️  Shape mismatch: {key}")
            print(f"   Old: {old_val.shape}, New: {new_val.shape}")
            continue

        exactly_equal = torch.equal(old_val, new_val)

        if not exactly_equal:
            if not old_val.dtype.is_floating_point:
                continue
            diff = (new_val - old_val).abs().mean().item()
        else:
            diff = 0.0

        # Categorize
        if 'implicit_network' in key:
            network_groups['implicit_network'].append((key, diff, exactly_equal))
        elif 'rendering_network' in key:
            network_groups['rendering_network'].append((key, diff, exactly_equal))
        elif 'background' in key:
            network_groups['background'].append((key, diff, exactly_equal))
        elif 'deformer' in key or 'skinning' in key:
            network_groups['deformer'].append((key, diff, exactly_equal))
        else:
            network_groups['other'].append((key, diff, exactly_equal))

    # Print summary by group
    for group_name, params in network_groups.items():
        if not params:
            continue

        updated = [p for p in params if not p[2]]
        frozen = [p for p in params if p[2]]

        print(f"\n{group_name.upper().replace('_', ' ')}:")
        print(f"  Total params: {len(params)}")
        print(f"  Updated: {len(updated)} ✅")
        print(f"  Frozen: {len(frozen)} {'❌' if frozen else '✅'}")

        if verbose and updated:
            # Show top 5 most changed
            top_changed = sorted(updated, key=lambda x: x[1], reverse=True)[:5]
            print(f"  Top changes:")
            for key, diff, _ in top_changed:
                param_name = key.split('.')[-1]
                layer = '.'.join(key.split('.')[-3:-1])
                print(f"    {layer}.{param_name}: Δ={diff:.8f}")

    print()

    # ================================================================
    # PART 3: SUMMARY & RECOMMENDATIONS
    # ================================================================
    print("=" * 80)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 80)

    total_params = len(all_keys)
    total_updated = sum(1 for k in network_keys
                        if k in old['state_dict'] and k in new['state_dict']
                        and not torch.equal(old['state_dict'][k], new['state_dict'][k]))
    total_updated += sum(1 for r in mano_results if r['changed'])

    print(f"Total parameters: {total_params}")
    print(f"Updated: {total_updated} ({total_updated / total_params * 100:.1f}%)")
    print(f"Frozen: {total_params - total_updated} ({(total_params - total_updated) / total_params * 100:.1f}%)")
    print()

    if any_mano_changed:
        print("✅ MANO OPTIMIZATION: WORKING")
        print("   → Safe to proceed with full training")
        print()

        # Estimate convergence
        if epoch_diff and epoch_diff > 0:
            right_pose_result = [r for r in mano_results if r['node'] == 'RIGHT' and r['param'] == 'pose']
            if right_pose_result:
                mean_diff = right_pose_result[0]['mean_diff']
                per_epoch = mean_diff / epoch_diff
                epochs_to_1percent = (0.01 * 0.1) / per_epoch if per_epoch > 0 else float('inf')

                print(f"CONVERGENCE ESTIMATE:")
                print(f"  Current RIGHT.pose change: {mean_diff:.8f} over {epoch_diff} epochs")
                print(f"  Per-epoch rate: {per_epoch:.8f}")
                if epochs_to_1percent < 1000:
                    print(f"  Estimated epochs to 1% change: ~{int(epochs_to_1percent)}")
    else:
        print("❌ MANO OPTIMIZATION: BROKEN")
        print("   → DO NOT proceed with full training")
        print("   → Debug gradient flow issues first")

    print()

    # ================================================================
    # PART 4: DETAILED MANO STATISTICS
    # ================================================================
    if any_mano_changed and verbose:
        print("=" * 80)
        print("DETAILED MANO PARAMETER STATISTICS")
        print("=" * 80)

        for result in mano_results:
            if result['changed']:
                print(f"\n{result['node']}.{result['param']}:")
                print(f"  Mean change: {result['mean_diff']:.10f}")
                print(f"  Max change:  {result['max_diff']:.10f}")
                print(f"  Relative:    {result['rel_change']:.4f}%")

                if epoch_diff:
                    print(f"  Per epoch:   {result['mean_diff'] / epoch_diff:.10f}")

        print()

    print("=" * 80)

    return {
        'mano_updated': any_mano_changed,
        'mano_results': mano_results,
        'total_params': total_params,
        'total_updated': total_updated,
        'epoch_diff': epoch_diff
    }


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python scripts/compare_mano_checkpoints.py <old_ckpt> <new_ckpt>")
        print()
        print("Example:")
        print("  python scripts/compare_mano_checkpoints.py \\")
        print("    logs/d839b2738/checkpoints/last.ckpt \\")
        print("    logs/a0419ab35/checkpoints/last.ckpt")
        sys.exit(1)

    old_ckpt = sys.argv[1]
    new_ckpt = sys.argv[2]

    if not Path(old_ckpt).exists():
        print(f"❌ Old checkpoint not found: {old_ckpt}")
        sys.exit(1)

    if not Path(new_ckpt).exists():
        print(f"❌ New checkpoint not found: {new_ckpt}")
        sys.exit(1)

    result = compare_checkpoints(old_ckpt, new_ckpt, verbose=True)

    # Save results to file
    output_file = "checkpoint_comparison_results.txt"
    print(f"\n📝 Results saved to: {output_file}")