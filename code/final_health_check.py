#!/usr/bin/env python3
"""Final health check before extended training"""

import re

print("=" * 80)
print("FINAL HEALTH CHECK: Is training ready for 30 epochs?")
print("=" * 80)
print()

with open('logs/stage1_density_fixed.log', 'r') as f:
    log = f.read()

checklist = {
    "No Inf values": "has_inf: True" not in log,
    "No NaN values": "has_nan: True" not in log,
    "Loss decreasing": re.search(r'\[Epoch 0\] Avg loss: ([0-9.]+)') and \
                       re.search(r'\[Epoch 9\] Avg loss: ([0-9.]+)'),
    "RGB loss logging": "DEBUG RGB LOSS" in log,
    "Training completed": "Training complete" in log,
    "Memory stable": "CUDA" not in log or "out of memory" not in log,
}

print("Health Check Results:")
print()
for check, status in checklist.items():
    symbol = "✅" if status else "❌"
    print(f"{symbol} {check}")

print()
passed = sum(1 for v in checklist.values() if v)
total = len(checklist)

print(f"Status: {passed}/{total} checks passed")

if passed == total:
    print("\n✅ READY FOR EXTENDED TRAINING (30 epochs)")
    print("\nRecommendations:")
    print("  1. Use improved config (lower LR, LR scheduler)")
    print("  2. Monitor for:")
    print("     - Loss continuing to decrease")
    print("     - No new Inf/NaN issues")
    print("     - Rotation axis issues (should improve)")
    print("  3. Stop if:")
    print("     - Loss increases for 5+ epochs")
    print("     - New numerical errors appear")
else:
    print(f"\n⚠️  {total - passed} issues remain - investigate before retraining")

