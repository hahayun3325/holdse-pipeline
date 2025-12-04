# Check training metadata from misc files
import numpy as np

official_misc = np.load('/home/fredcui/Projects/hold/code/logs/cb20a1702/misc/000080000.npy', allow_pickle=True).item()

print("=" * 80)
print("OFFICIAL CHECKPOINT TRAINING METADATA")
print("=" * 80)
for key, value in official_misc.items():
    print(f"{key:<30} {value}")

# If you have misc files for your checkpoint
import glob
stage1_misc_files = sorted(glob.glob('logs/140dc5c18/misc/*.npy'))
if stage1_misc_files:
    stage1_misc = np.load(stage1_misc_files[-1], allow_pickle=True).item()
    print("\n" + "=" * 80)
    print("STAGE 1 CHECKPOINT TRAINING METADATA")
    print("=" * 80)
    for key, value in stage1_misc.items():
        print(f"{key:<30} {value}")