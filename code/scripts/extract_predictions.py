#!/usr/bin/env python3
# scripts/extract_predictions.py - PRODUCTION VERSION
"""Extract 3D predictions with coordinate transformation."""

import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import glob

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from src.hold.hold import HOLD
from src.datasets.utils import create_dataset
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from smplx import MANO
from pytorch3d.transforms import axis_angle_to_matrix
from src.utils.eval_modules import compute_bounding_box_centers


def thing2dev(data, device='cuda'):
    """Move data to device."""
    if isinstance(data, dict):
        return {k: thing2dev(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(thing2dev(item, device) for item in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(device)
    else:
        return data


def find_mano_path():
    """Find MANO model files."""
    candidates = [
        'data/mano', 'assets/mano', 'third_party/mano_v1_2/models',
        '../assets/mano', '../third_party/mano_v1_2/models',
    ]
    for path in candidates:
        p = Path(path)
        if p.exists() and (p / 'MANO_RIGHT.pkl').exists():
            return str(p.resolve())
    raise FileNotFoundError("MANO not found")


def map_deform2eval(verts, scale, _normalize_shift):
    """Transform from deformation space to evaluation space."""
    conversion_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    normalize_shift = _normalize_shift.copy()
    normalize_shift[0] *= -1

    src_verts = np.copy(verts)
    src_verts = np.dot(src_verts, conversion_matrix)
    src_verts *= scale
    src_verts += normalize_shift
    return src_verts


def extract_predictions(checkpoint_path, seq_name, config_path=None, flat_hand_mean=False):
    """Extract predictions with coordinate transformation to evaluation space."""

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    # Detect number of frames
    try:
        encoder_weight = ckpt['state_dict']['model.nodes.object.frame_latent_encoder.weight']
        n_frames = encoder_weight.shape[0]
        print(f"✓ Detected {n_frames} frames from checkpoint")
    except KeyError:
        n_frames = 144
        print(f"⚠️ Using default {n_frames} frames")

    # Load configuration
    if config_path is None:
        config_path = f'confs/stage1_{seq_name}.yaml'
    if not Path(config_path).exists():
        config_path = 'confs/general.yaml'

    print(f"Using config: {config_path}")
    opt = OmegaConf.load(config_path)

    if not hasattr(opt, 'model'):
        opt.model = OmegaConf.create({})
    if not hasattr(opt.model, 'scene_bounding_sphere'):
        opt.model.scene_bounding_sphere = 3.0

    # Load coordinate transformation parameters
    print("\n=== Loading coordinate transformation parameters ===")
    dataset_path = f'data/{seq_name}/build/data.npy'
    dataset = np.load(dataset_path, allow_pickle=True).item()
    normalize_shift = dataset['normalize_shift']
    print(f"Normalize shift: {normalize_shift}")

    checkpoint_dir = Path(checkpoint_path).parent.parent
    misc_files = sorted(glob.glob(str(checkpoint_dir / 'misc' / '*.npy')))

    if misc_files:
        misc = np.load(misc_files[-1], allow_pickle=True).item()
        scale = float(misc['scale'])
        print(f"Scale from misc: {scale}")
    else:
        scale = float(dataset.get('scene_bounding_sphere', 3.0))
        print(f"⚠️ Using scale from dataset: {scale}")

    inverse_scale = 1.0 / scale

    class Args:
        case = seq_name
        n_images = n_frames
        num_sample = 2048
        infer_ckpt = checkpoint_path
        ckpt_p = checkpoint_path
        no_vis = False
        render_downsample = 1
        freeze_pose = False
        experiment = 'extract_predictions'
        log_every = 100
        log_dir = 'logs/extraction'
        barf_s = 0
        barf_e = 0
        no_barf = True
        shape_init = ""
        exp_key = 'extraction'
        debug = False
        agent_id = -1
        offset = 1
        num_workers = 0

    args = Args()

    print("\nInitializing HOLD model...")
    model = HOLD(opt, args)
    model.phase3_enabled = False
    model.phase4_enabled = False
    model.phase5_enabled = False
    if hasattr(model, 'ghop_enabled'):
        model.ghop_enabled = False

    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.cuda()
    model.eval()

    mano_path = find_mano_path()
    print(f"✓ MANO: {mano_path}")

    mano_model = MANO(
        model_path=mano_path,
        is_rhand=True,
        use_pca=False,
        flat_hand_mean=flat_hand_mean,
    ).cuda()
    mano_model.eval()

    # Object vertices
    obj_v_cano = None
    obj_vertex_count = 1000
    try:
        obj_key = 'model.nodes.object.server.object_model.v3d_cano'
        if obj_key in ckpt['state_dict']:
            obj_v_cano = ckpt['state_dict'][obj_key].cuda()
            obj_vertex_count = obj_v_cano.shape[0]
            print(f"✓ Object: {obj_vertex_count} vertices")
    except:
        pass

    # Create dataset (TEST mode for all frames)
    print("Creating dataset (TEST mode for all frames)...")
    test_config = OmegaConf.create({
        'type': 'test',
        'batch_size': 1,
        'drop_last': False,
        'shuffle': False,
        'num_workers': 0,
        'pixel_per_batch': 512,
    })

    full_dataset = create_dataset(test_config, args)
    base_dataset = full_dataset.dataset if hasattr(full_dataset, 'dataset') else full_dataset

    print(f"✓ Dataset: {len(base_dataset)} frames")

    if len(base_dataset) != n_frames:
        print(f"⚠️ WARNING: Dataset has {len(base_dataset)} frames, checkpoint has {n_frames}")

    # Detect output keys
    print("\nDetecting output keys...")
    dataloader = DataLoader(base_dataset, batch_size=1, shuffle=False, num_workers=0)

    with torch.no_grad():
        sample_batch = next(iter(dataloader))
        sample_cuda = thing2dev(sample_batch, 'cuda')
        sample_output = model.validation_step(sample_cuda)

    param_keys = {}
    key_mappings = {
        'betas': ['right.betas', 'right_betas'],
        'hand_pose': ['right.pose', 'right_pose'],
        'global_orient': ['right.global_orient', 'right_global_orient'],
        'transl': ['right.transl', 'right_transl'],
        'obj_global_orient': ['object.global_orient', 'object_global_orient'],
        'obj_transl': ['object.transl', 'object_transl'],
    }

    for param, candidates in key_mappings.items():
        for key in candidates:
            if key in sample_output:
                param_keys[param] = key
                print(f"✓ {param}: '{key}'")
                break

    if len(param_keys) != len(key_mappings):
        print("❌ Missing keys!")
        return None

    # Extract all frames
    print(f"\nExtracting {len(base_dataset)} frames...")

    all_hand_verts = []
    all_hand_joints = []
    all_obj_verts = []
    fnames = []
    is_valid = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, total=len(dataloader), desc="Extracting")):
            batch_cuda = thing2dev(batch, 'cuda')

            try:
                output = model.validation_step(batch_cuda)

                betas = output[param_keys['betas']].detach()
                hand_pose = output[param_keys['hand_pose']].detach()
                global_orient = output[param_keys['global_orient']].detach()
                transl = output[param_keys['transl']].detach()

                if betas.ndim == 1: betas = betas.unsqueeze(0)
                if hand_pose.ndim == 1: hand_pose = hand_pose.unsqueeze(0)
                if global_orient.ndim == 1: global_orient = global_orient.unsqueeze(0)
                if transl.ndim == 1: transl = transl.unsqueeze(0)

                mano_output = mano_model(
                    betas=betas,
                    hand_pose=hand_pose,
                    global_orient=global_orient,
                    transl=transl,
                    return_verts=True
                )

                hand_verts = mano_output.vertices.detach().cpu().numpy()
                hand_joints = mano_output.joints.detach().cpu().numpy()

                # Object
                if obj_v_cano is not None:
                    try:
                        obj_go = output[param_keys['obj_global_orient']].detach()
                        obj_t = output[param_keys['obj_transl']].detach()

                        if obj_go.ndim == 1: obj_go = obj_go.unsqueeze(0)
                        if obj_t.ndim == 1: obj_t = obj_t.unsqueeze(0)

                        R = axis_angle_to_matrix(obj_go)
                        obj_verts = (
                                torch.matmul(obj_v_cano.unsqueeze(0), R.transpose(1, 2)) +
                                obj_t.unsqueeze(1)
                        ).detach().cpu().numpy()
                    except:
                        obj_verts = obj_v_cano.unsqueeze(0).detach().cpu().numpy()
                else:
                    obj_verts = np.zeros((1, obj_vertex_count, 3))

                all_hand_verts.append(hand_verts.squeeze(0))
                all_hand_joints.append(hand_joints.squeeze(0))
                all_obj_verts.append(obj_verts.squeeze(0))

                if 'idx' in batch:
                    idx = batch['idx'].item() if torch.is_tensor(batch['idx']) else int(batch['idx'])
                    fname = f"frame_{idx:04d}"
                else:
                    fname = f"frame_{batch_idx:04d}"

                fnames.append(fname)
                is_valid.append(True)

            except Exception as e:
                print(f"\n⚠️ Frame {batch_idx} failed: {e}")
                all_hand_verts.append(np.zeros((778, 3)))
                all_hand_joints.append(np.zeros((16, 3)))
                all_obj_verts.append(np.zeros((obj_vertex_count, 3)))
                fnames.append(f"frame_invalid_{batch_idx}")
                is_valid.append(False)

    hand_verts = np.stack(all_hand_verts, axis=0)
    hand_joints = np.stack(all_hand_joints, axis=0)
    obj_verts = np.stack(all_obj_verts, axis=0)
    is_valid = np.array(is_valid)

    print(f"\n✓ Extraction complete (deformation space):")
    print(f"  Hand vertices: {hand_verts.shape}, range [{hand_verts.min():.3f}, {hand_verts.max():.3f}]")
    print(f"  Hand joints: {hand_joints.shape}")
    print(f"  Object vertices: {obj_verts.shape}")
    print(f"  Valid: {is_valid.sum()}/{len(is_valid)}")

    # Apply coordinate transformation
    print("\n=== Applying map_deform2eval transformation ===")
    hand_verts_eval = np.array([map_deform2eval(frame, inverse_scale, normalize_shift) for frame in hand_verts])
    hand_joints_eval = np.array([map_deform2eval(frame, inverse_scale, normalize_shift) for frame in hand_joints])
    obj_verts_eval = np.array([map_deform2eval(frame, inverse_scale, normalize_shift) for frame in obj_verts])

    print(f"✓ Transformed to evaluation space:")
    print(f"  Hand vertices range: [{hand_verts_eval.min():.3f}, {hand_verts_eval.max():.3f}]")

    # Compute root-aligned coordinates
    print("\n=== Computing root-aligned coordinates ===")
    j3d_c_right = hand_joints_eval
    root_right = j3d_c_right[:, :1, :]
    j3d_ra_right = j3d_c_right - root_right

    v3d_c_object = obj_verts_eval
    root_object = compute_bounding_box_centers(v3d_c_object)
    v3d_ra_object = v3d_c_object - root_object[:, None, :]
    v3d_right_object = v3d_c_object - root_right.squeeze(1)[:, None, :]

    print(f"✓ Root-aligned coordinates computed")

    # Prepare predictions dictionary
    predictions = {
        'full_seq_name': seq_name,
        'fnames': np.array(fnames),
        'is_valid': is_valid,
        'v3d_c.right': hand_verts_eval,
        'j3d_c.right': hand_joints_eval,
        'v3d_c.object': obj_verts_eval,
        'j3d_ra.right': j3d_ra_right,
        'root.right': root_right.squeeze(1),
        'root.object': root_object,
        'v3d_ra.object': v3d_ra_object,
        'v3d_right.object': v3d_right_object,
    }

    # Convert to tensors (skip string arrays)
    print("\n=== Converting to PyTorch tensors ===")
    for key in list(predictions.keys()):
        if isinstance(predictions[key], np.ndarray):
            dtype = predictions[key].dtype
            if dtype.kind not in ['U', 'S', 'O']:
                predictions[key] = torch.from_numpy(predictions[key])
                print(f"  ✓ {key}: {predictions[key].shape}, {predictions[key].dtype}")

    # Add MANO faces
    faces_np = np.array(mano_model.faces, dtype=np.int64)
    predictions['faces'] = torch.from_numpy(faces_np)
    predictions['faces.right'] = predictions['faces']
    print(f"  ✓ faces: {predictions['faces'].shape}, {predictions['faces'].dtype}")

    print(f"\n✅ Extraction complete with coordinate transformation")
    return predictions


def main():
    parser = argparse.ArgumentParser(description='Extract 3D predictions from HOLD checkpoint')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint (.ckpt)')
    parser.add_argument('--seq_name', required=True, help='Sequence name (e.g., hold_MC1_ho3d)')
    parser.add_argument('--output', required=True, help='Output path for predictions (.pkl)')
    parser.add_argument('--config', default=None, help='Config file path (optional)')
    parser.add_argument('--flat_hand_mean', action='store_true', help='Use flat hand mean for MANO')
    args = parser.parse_args()

    predictions = extract_predictions(
        args.checkpoint, args.seq_name, args.config, args.flat_hand_mean
    )

    if predictions is None:
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(predictions, output_path)
    print(f"\n✓ Predictions saved to: {output_path}")

    # Print next steps
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print(f"1. Evaluate predictions:")
    print(f"   python scripts/evaluate_predictions.py --predictions {output_path}")
    print(f"\n2. Or compare with HOLD official:")
    print(f"   python scripts/evaluate_predictions.py \\")
    print(f"       --predictions {output_path} \\")
    print(f"       --compare ~/Projects/hold/code/logs/cb20a1702/checkpoints/last.ckpt.metric.json")
    print("=" * 70)


if __name__ == '__main__':
    main()

'''
# Make it executable
chmod +x scripts/extract_predictions.py

python scripts/extract_predictions.py \
    --checkpoint logs/e1c95c0d0/checkpoints/last.ckpt \
    --seq_name hold_MC1_ho3d \
    --config confs/stage1_hold_MC1_ho3d_8layer_implicit.yaml \
    --output logs/evaluation_results/MC1_stage1_e200_8layer_implicitnet_predictions.pkl \
    2>&1 | tee logs/evaluation_results/MC1_stage1_e200_8layer_implicitnet_extraction_debug.log

# Monitor the progress in a separate terminal
tail -f logs/evaluation_results/MC1_stage1_e200_8layer_implicitnet_extraction_debug_fresh.log | grep "Extracting:"
'''