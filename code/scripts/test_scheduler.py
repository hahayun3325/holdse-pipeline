#!/usr/bin/env python3
"""Test Phase5TrainingScheduler weight schedules"""
import sys
sys.path.insert(0, 'src')

from training.phase5_scheduler import Phase5TrainingScheduler

# Initialize with same config as training
scheduler = Phase5TrainingScheduler(
    total_iterations=60000,  # 30 epochs * 2000 steps
    warmup_iters=0,
    phase3_start=0,
    phase4_start=20000,
    phase5_start=20500,
    finetune_start=50000
)

print("=" * 70)
print("SCHEDULER WEIGHT TEST")
print("=" * 70)

# Test key steps
test_steps = [0, 100, 1000, 10000, 
              19999, 20000, 20100, 20200,  # Phase 4 transition
              20499, 20500, 20600, 20700,  # Phase 5 transition
              49999, 50000, 50100, 55000, 59999]  # Finetune transition

print("\nStep | SDS   | Contact | Temporal | Diffusion | Phase")
print("-" * 70)

for step in test_steps:
    weights = scheduler.get_loss_weights(step)
    phase = scheduler.get_phase_name(step)
    
    print(f"{step:5d} | {weights['sds']:5.2f} | {weights['contact']:7.3f} | "
          f"{weights['temporal']:8.3f} | {weights['diffusion']:9.2f} | {phase}")

print("=" * 70)

# Specific checks
print("\n=== CRITICAL CHECKS ===")

# Phase 4 should activate at step 20000
w_20000 = scheduler.get_loss_weights(20000)
print(f"Step 20000 contact weight: {w_20000['contact']:.4f}")
if w_20000['contact'] == 0.0:
    print("  ❌ FAIL: Contact weight is 0.0 at phase4_start!")
else:
    print(f"  ✅ PASS: Contact weight > 0")

# Phase 4 should be fully ramped at step 20100
w_20100 = scheduler.get_loss_weights(20100)
print(f"\nStep 20100 contact weight: {w_20100['contact']:.4f}")
if w_20100['contact'] >= 1.0:
    print("  ✅ PASS: Contact fully ramped (>=1.0)")
else:
    print(f"  ⚠️  WARNING: Contact not fully ramped yet ({w_20100['contact']:.4f}/1.0)")

# Phase 5 should activate at step 20500
w_20500 = scheduler.get_loss_weights(20500)
print(f"\nStep 20500 temporal weight: {w_20500['temporal']:.4f}")
if w_20500['temporal'] == 0.0:
    print("  ❌ FAIL: Temporal weight is 0.0 at phase5_start!")
else:
    print(f"  ✅ PASS: Temporal weight > 0")

# Finetune should reduce SDS at step 50000
w_49999 = scheduler.get_loss_weights(49999)
w_50100 = scheduler.get_loss_weights(50100)
print(f"\nSDS weight before finetune (step 49999): {w_49999['sds']:.4f}")
print(f"SDS weight during finetune (step 50100): {w_50100['sds']:.4f}")
if w_50100['sds'] < w_49999['sds']:
    print("  ✅ PASS: SDS reduced during finetune")
else:
    print("  ❌ FAIL: SDS not reducing during finetune")

