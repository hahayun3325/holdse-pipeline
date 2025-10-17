# File: code/scripts/collect_evaluation_info.py
import subprocess
import json
from pathlib import Path
from datetime import datetime


def collect_all_evaluation_info():
    """Comprehensive information collection for evaluation decision"""

    info = {
        'timestamp': datetime.now().isoformat(),
        'project': 'HOLDSE',
        'training_run': 'training_validation'
    }

    print("=" * 70)
    print("COLLECTING EVALUATION INFORMATION")
    print("=" * 70)

    # 1. Checkpoint information
    print("\n[1/6] Checkpoint Information...")
    checkpoint_dir = Path('logs/training_validation/checkpoints')
    info['checkpoints'] = {
        'total': len(list(checkpoint_dir.glob('epoch=*.ckpt'))),
        'files': [f.name for f in sorted(checkpoint_dir.glob('epoch=*.ckpt'))]
    }

    # 2. Visual outputs
    print("[2/6] Visual Outputs...")
    visuals_dir = Path('logs/training_validation/visuals')
    info['visuals'] = {
        'types': [d.name for d in visuals_dir.iterdir() if d.is_dir()],
        'total_images': sum(len(list(d.glob('*.png')))
                            for d in visuals_dir.iterdir() if d.is_dir())
    }

    # 3. Training configuration
    print("[3/6] Training Configuration...")
    args_file = Path('logs/training_validation/args.json')
    if args_file.exists():
        with open(args_file, 'r') as f:
            info['config'] = json.load(f)

    # 4. Training log summary
    print("[4/6] Training Log...")
    log_file = Path('logs/training_validation/train.log')
    if log_file.exists():
        with open(log_file, 'r') as f:
            lines = f.readlines()
        info['log'] = {
            'total_lines': len(lines),
            'last_10_lines': lines[-10:]
        }

    # 5. GPU info
    print("[5/6] GPU Information...")
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
            capture_output=True, text=True
        )
        info['gpu'] = result.stdout.strip()
    except:
        info['gpu'] = 'Not available'

    # 6. Disk usage
    print("[6/6] Disk Usage...")
    try:
        result = subprocess.run(
            ['du', '-sh', 'logs/training_validation'],
            capture_output=True, text=True
        )
        info['disk_usage'] = result.stdout.strip().split()[0]
    except:
        info['disk_usage'] = 'Unknown'

    # Save comprehensive report
    output_file = 'evaluation_info_complete.json'
    with open(output_file, 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\n✓ Complete information saved to: {output_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Checkpoints: {info['checkpoints']['total']}")
    print(f"Visual types: {len(info['visuals']['types'])}")
    print(f"Total images: {info['visuals']['total_images']}")
    print(f"Disk usage: {info['disk_usage']}")
    print(f"GPU: {info['gpu']}")

    return info


if __name__ == '__main__':
    collect_all_evaluation_info()