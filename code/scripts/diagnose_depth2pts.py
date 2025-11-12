"""Diagnose depth2pts_outside NaN production"""

import torch
import sys

sys.path.insert(0, '.')
sys.path.insert(0, '../common')

import os

os.environ['COMET_MODE'] = 'disabled'

from src.hold.hold import HOLD
from src.datasets.utils import create_dataset
from thing import thing2dev
from omegaconf import OmegaConf

config_path = 'confs/ghop_production_chunked_20251027_131408.yaml'
checkpoint_path = 'logs/ad1f0073b/checkpoints/last.ckpt'

opt = OmegaConf.load(config_path)
if not hasattr(opt.model, 'scene_bounding_sphere'):
    opt.model.scene_bounding_sphere = 3.0


class Args:
    case = 'ghop_bottle_1'
    n_images = 71
    num_sample = 2048
    infer_ckpt = checkpoint_path
    ckpt_p = checkpoint_path
    no_vis = False
    render_downsample = 2
    freeze_pose = False
    experiment = 'test'
    log_every = 10
    log_dir = 'logs/test'
    barf_s = 0
    barf_e = 0
    no_barf = True
    shape_init = ""
    exp_key = 'test'
    debug = False


args = Args()

print("Loading model...")
model = HOLD(opt, args)
model.phase3_enabled = False
model.phase4_enabled = False
model.phase5_enabled = False

ckpt = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(ckpt['state_dict'], strict=False)
model.cuda()
model.eval()

print("\n" + "=" * 70)
print("TESTING depth2pts_outside FOR NUMERICAL ISSUES")
print("=" * 70)

val_config = opt.dataset.val if hasattr(opt.dataset, 'val') else opt.dataset.valid
val_dataset = create_dataset(val_config, args)

for batch in val_dataset:
    batch_cuda = thing2dev(batch, 'cuda')

    with torch.no_grad():
        try:
            output = model.validation_step(batch_cuda)
            print("\n✅ Rendering completed")
            print(f"   rgb has_nan: {torch.isnan(output['rgb']).any().item()}")

            # Look for the debug prints we added to background.py
            break

        except Exception as e:
            print(f"\n❌ Error during rendering: {e}")
            import traceback

            traceback.print_exc()

    break

print("\n" + "=" * 70)
print("Check the output above for [DEBUG] lines from background.py")
print("=" * 70)