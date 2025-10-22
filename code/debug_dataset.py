# File: code/debug_dataset.py
"""Debug why testset only has 3 items."""

import sys

sys.path = [".."] + sys.path
from src.utils.parser import parser_args
from src.datasets.utils import create_dataset
import numpy as np


def main():
    args, opt = parser_args()
    args.case = 'ghop_bottle_1'
    args.load_ckpt = 'logs/b2e4b039a/checkpoints/last.ckpt'

    # Create test dataset
    try:
        testset = create_dataset(opt.dataset.test, args)
        print("✅ Using test dataset")
    except:
        testset = create_dataset(opt.dataset.train, args)
        print("⚠️  Using train dataset (test not found)")

    print(f"\n📊 Dataset Analysis:")
    print(f"  DataLoader batches: {len(testset)}")
    print(f"  Batch size: {testset.batch_size}")

    # Get underlying dataset
    dataset = testset.dataset
    print(f"  Dataset type: {type(dataset)}")
    print(f"  Dataset length: {len(dataset)}")

    # Check wrapped dataset
    if hasattr(dataset, 'dataset'):
        inner = dataset.dataset
        print(f"  Inner dataset type: {type(inner)}")
        print(f"  Inner dataset length: {len(inner)}")

        if hasattr(inner, 'img_paths'):
            img_paths = np.array(inner.img_paths)
            print(f"  Image paths: {len(img_paths)}")
            print(f"    First 3: {img_paths[:3]}")
            print(f"    Last 3: {img_paths[-3:]}")

        # Check if there's a test_indices attribute
        if hasattr(inner, 'test_indices'):
            print(f"  Test indices: {inner.test_indices}")

        # Check if there's filtering
        if hasattr(inner, 'indices'):
            print(f"  Filtered indices: {inner.indices}")

    # Try to iterate
    print(f"\n🔄 Testing iteration:")
    count = 0
    for batch in testset:
        count += 1
        if count <= 3:
            print(f"  Batch {count}: idx={batch['idx']}")

    print(f"\n  Total batches iterated: {count}")

    if count == 3 and len(testset) == 71:
        print(f"\n❌ FOUND ISSUE:")
        print(f"   DataLoader says {len(testset)} batches")
        print(f"   But iteration stops at 3")
        print(f"   → Dataset __getitem__ or __len__ is wrong!")


if __name__ == '__main__':
    main()
