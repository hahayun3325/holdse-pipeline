import sys
sys.path.insert(0, '.')

# Monkey-patch TestDataset to only use 3 frames
import src.datasets.eval_datasets as eval_ds
original_init = eval_ds.TestDataset.__init__

def patched_init(self, args):
    original_init(self, args)
    print(f"[TEST MODE] Limiting dataset to 3 frames (was {len(self.eval_idx_list)})")
    self.eval_idx_list = self.eval_idx_list[:3]

eval_ds.TestDataset.__init__ = patched_init

# Run extraction
from scripts.extract_predictions import main
main()
