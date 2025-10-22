import sys

sys.path = [".."] + sys.path
import sys
from glob import glob

import common.comet_utils as comet_utils
from omegaconf import OmegaConf
import numpy as np
import os


def parser_args():
    import argparse
    from easydict import EasyDict as edict

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./confs/general.yaml")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--case", type=str, required=True)
    parser.add_argument("--shape_init", type=str, default="75268d864")
    parser.add_argument("--mute", help="No logging", action="store_true")
    parser.add_argument("--agent_id", type=int, default=0)
    parser.add_argument("--num_sample", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--exp_key", type=str, default="")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--num_epoch", type=int, default=200)
    # Disable Comet logging for debugging
    parser.add_argument(
        '--no-comet',
        action='store_true',
        default=False,
        help='Disable Comet ML logging for faster debugging (skips metric uploads)'
    )
    parser.add_argument("--freeze_pose", action="store_true", help="no optimize pose")
    parser.add_argument("--barf_s", type=int, default=1000)
    parser.add_argument("--barf_e", type=int, default=10000)
    parser.add_argument("--no_barf", action="store_true", help="no barf")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--offset", type=int, default=1)
    parser.add_argument("--no_meshing", action="store_true")
    parser.add_argument("--no_vis", action="store_true")
    parser.add_argument("--render_downsample", type=int, default=2)
    parser.add_argument(
        "-f",
        "--fast",
        dest="fast_dev_run",
        help="single batch for development",
        action="store_true",
    )
    parser.add_argument(
        "--infer_ckpt",
        type=str,
        default="",
        help="Resume training from checkpoint and keep logging in the same comet exp",
    )
    parser.add_argument(
        "--load_ckpt",
        type=str,
        default="",
        help="Resume training from checkpoint and keep logging in the same comet exp",
    )
    parser.add_argument(
        "--load_pose",
        type=str,
        default="",
        help="Resume training from checkpoint and keep logging in the same comet exp",
    )
    parser.add_argument(
        "--eval_every_epoch", type=int, default=6, help="Eval every k epochs"
    )
    parser.add_argument("--tempo_len", type=int, default=2000)
    parser.add_argument("--dump_eval_meshes", action="store_true")
    # ========================================================================
    # PHASE 3: GHOP Two-Stage Training Command-Line Arguments
    # ========================================================================
    # These arguments provide command-line control over GHOP integration
    # Priority: CLI args > config file (confs/general.yaml)
    # ========================================================================
    # MEMORY OPTIMIZATION: DataLoader pin_memory Control
    # ========================================================================
    parser.add_argument(
        '--use_ghop',
        action='store_true',
        help='Enable GHOP Phase 3 two-stage training (SDS + Contact losses). '
             'Overrides config file phase3.enabled if set.'
    )

    # VQ-VAE checkpoint path
    parser.add_argument(
        '--vqvae_ckpt',
        type=str,
        default='checkpoints/ghop/vqvae_last.ckpt',
        help='Path to VQ-VAE checkpoint for compressing 64³ SDF to 3×16³ latent. '
             'Download from GHOP repository if not available.'
    )

    # 3D U-Net diffusion model checkpoint
    parser.add_argument(
        '--unet_ckpt',
        type=str,
        default='checkpoints/ghop/unet_last.ckpt',
        help='Path to 3D U-Net diffusion model checkpoint for noise prediction. '
             'Download from GHOP repository if not available.'
    )

    # Classifier-free guidance scale
    parser.add_argument(
        '--guidance_scale',
        type=float,
        default=4.0,
        help='Classifier-free guidance scale (w) for SDS loss. '
             'Higher values = stronger category-specific priors. Typical range: 2.0-7.0'
    )

    # Stage 1 duration (SDS loss)
    parser.add_argument(
        '--sds_iters',
        type=int,
        default=500,
        help='Number of training iterations for Stage 1 (SDS loss). '
             'Coarse geometry alignment using diffusion prior. Recommended: 500-1000'
    )

    # Stage 2 duration (Contact loss)
    parser.add_argument(
        '--contact_iters',
        type=int,
        default=100,
        help='Number of training iterations for Stage 2 (Contact loss). '
             'Fine-grained surface contact refinement. Recommended: 50-200'
    )

    # Maximum SDS loss weight
    parser.add_argument(
        '--w_sds',
        type=float,
        default=5000.0,
        help='Maximum weight for SDS loss in Stage 1. '
             'Weight ramps from 0 to this value. Typical range: 1000-10000'
    )

    # Maximum contact loss weight
    parser.add_argument(
        '--w_contact',
        type=float,
        default=10.0,
        help='Maximum weight for contact loss in Stage 2. '
             'Weight ramps from 0 to this value. Typical range: 5.0-20.0'
    )

    # Grid resolution for SDF extraction
    parser.add_argument(
        '--grid_resolution',
        type=int,
        default=64,
        help='Grid resolution for SDF extraction (Phase 3 uses 64³ for high fidelity). '
             'Must match hand_field_resolution. Options: 16, 32, 64, 128'
    )

    # Hand field spatial limit
    parser.add_argument(
        '--spatial_lim',
        type=float,
        default=1.5,
        help='Spatial limit for hand field computation (±limit in meters). '
             'Defines bounding box for hand-object interaction. Default: 1.5m'
    )

    # Use modular initialization
    parser.add_argument(
        '--use_modular_init',
        action='store_true',
        help='Use modular component initialization (VQ-VAE, U-Net, Hand Field directly). '
             'Recommended for Phase 3. If false, uses legacy HOLDLoss wrapper.'
    )

    # ========================================================================
    # END PHASE 3 ARGUMENTS
    # ========================================================================

    # ========================================================================
    # MEMORY OPTIMIZATION: DataLoader pin_memory Control
    # ========================================================================
    parser.add_argument(
        '--no-pin-memory',
        dest='no_pin_memory',
        action='store_true',
        default=False,
        help='Disable pin_memory in DataLoader to prevent memory leaks.'
    )

    parser.add_argument(
        '--pin-memory',
        dest='force_pin_memory',
        action='store_true',
        default=False,
        help='Force enable pin_memory in DataLoader (use with caution).'
    )
    # ========================================================================

    args = parser.parse_args()
    args = edict(vars(args))
    opt = edict(OmegaConf.load(args.config))

    # ========================================================================
    # PROCESS MEMORY OPTIMIZATION FLAGS
    # ========================================================================
    if args.no_pin_memory and args.force_pin_memory:
        raise ValueError(
            "Cannot specify both --no-pin-memory and --pin-memory."
        )

    if args.no_pin_memory:
        args.pin_memory = False
        import logging
        logging.getLogger(__name__).warning(
            "[Memory] pin_memory DISABLED via --no-pin-memory"
        )
    elif args.force_pin_memory:
        args.pin_memory = True
        import logging
        logging.getLogger(__name__).warning(
            "[Memory] pin_memory FORCE ENABLED (may leak memory!)"
        )
    else:
        import logging
        logging.getLogger(__name__).info(
            "[Memory] pin_memory will be AUTO-DETECTED"
        )
    # ========================================================================

    cmd = " ".join(sys.argv)
    args.cmd = cmd
    args.project = "blaze"

    data = np.load(f"./data/{args.case}/build/data.npy", allow_pickle=True).item()
    opt.model.scene_bounding_sphere = data["scene_bounding_sphere"]

    if args.fast_dev_run:
        args.num_workers = 0
        args.eval_every_epoch = 1
        args.num_sample = 8
        args.tempo_len = 50
        args.log_every = 1

    args.total_step = int(
        args.num_epoch * args.tempo_len / opt.dataset.train.batch_size
    )

    # ================================================================
    # Comet experiment initialization
    # ================================================================
    # ALWAYS call init_experiment - it handles disabled mode internally
    if os.environ.get('COMET_MODE', 'online') == 'disabled':
        # Comet disabled - pass dummy credentials
        # init_experiment will check COMET_MODE and skip Comet setup
        experiment, args = comet_utils.init_experiment(
            args, api_key='disabled', workspace='disabled'
        )
    else:
        # Normal Comet flow
        api_key = os.environ["COMET_API_KEY"]
        workspace = os.environ["COMET_WORKSPACE"]

        experiment, args = comet_utils.init_experiment(
            args, api_key=api_key, workspace=workspace
        )

    # Only call save_args and log_exp_meta if experiment exists
    if experiment is not None:
        comet_utils.save_args(args, save_keys=["comet_key", "git_commit", "git_branch"])
        comet_utils.log_exp_meta(args)
    else:
        print("[Parser] Training without Comet logging - paths set up successfully")

    img_paths = sorted(glob(f"./data/{args.case}/build/image/*.png"))
    assert len(img_paths) > 0, "No images found"
    args.n_images = len(img_paths)
    return args, opt
