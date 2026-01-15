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
from torch.utils.data import Subset

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

    pose_mean = ckpt['state_dict']['model.nodes.right.params.pose.weight'].mean()
    print(f"\n{'=' * 60}")
    print(f"CHECKPOINT VERIFICATION")
    print(f"{'=' * 60}")
    print(f"Path: {checkpoint_path}")
    print(f"Pose mean: {pose_mean:.6f}")
    print(f"Expected:")
    print(f"  Official: 0.017972")
    print(f"  Stage1:   0.018969")
    print(f"  Stage2:   0.017512")
    print(f"{'=' * 60}\n")

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
    misc_path = checkpoint_dir / 'misc' / '000080000.npy'

    if not misc_path.exists():
        raise FileNotFoundError(f"Required misc file not found: {misc_path}")

    misc = np.load(misc_path, allow_pickle=True).item()
    scale = float(misc['scale'])
    print(f"✓ Loaded scale from misc: {scale}")

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
        # ========== CHECKPOINT PRESERVATION FLAG ==========
        # This tells MANONode/ObjectNode to preserve checkpoint params
        # instead of loading from dataset
        loading_from_checkpoint = True
        # ==================================================

    args = Args()

    print("\nInitializing HOLD model...")
    model = HOLD(opt, args)
    model.phase3_enabled = False
    model.phase4_enabled = False
    model.phase5_enabled = False
    if hasattr(model, 'ghop_enabled'):
        model.ghop_enabled = False

    model.load_state_dict(ckpt['state_dict'], strict=False)

    # ========== VERIFICATION: What parameters were loaded? ==========
    print("\n" + "=" * 60)
    print("CHECKPOINT PARAMETER VERIFICATION")
    print("=" * 60)

    # First, discover the actual model structure
    print("\nModel top-level attributes:")
    for attr in dir(model):
        if not attr.startswith('_') and not callable(getattr(model, attr, None)):
            print(f"  - {attr}")

    # Try to find pose parameters by searching all state dict keys
    print("\nSearching for pose parameters in checkpoint...")
    pose_keys = [k for k in ckpt['state_dict'].keys() if 'pose' in k.lower()]
    print(f"Found {len(pose_keys)} pose-related keys:")
    for key in pose_keys[:10]:  # Show first 10
        value = ckpt['state_dict'][key]
        print(f"  {key}: shape={value.shape}, mean={value.mean():.6f}")

    # Try to find pose in loaded model
    print("\nSearching for pose parameters in loaded model...")
    model_pose_keys = [k for k, v in model.named_parameters() if 'pose' in k.lower()]
    print(f"Found {len(model_pose_keys)} pose-related parameters:")
    for key in model_pose_keys[:10]:
        param = dict(model.named_parameters())[key]
        print(f"  {key}: shape={param.shape}, mean={param.mean():.6f}")

    # If we found matching keys, compare them
    if pose_keys and model_pose_keys:
        print("\nComparing checkpoint vs loaded model:")
        # Try to match first pose key
        ckpt_key = pose_keys[0]
        model_key = model_pose_keys[0]

        ckpt_pose = ckpt['state_dict'][ckpt_key]
        model_pose = dict(model.named_parameters())[model_key]

        print(f"\nCheckpoint '{ckpt_key}':")
        print(f"  Mean: {ckpt_pose.mean():.6f}")
        print(f"  Std: {ckpt_pose.std():.6f}")
        print(f"  Shape: {ckpt_pose.shape}")

        print(f"\nModel '{model_key}':")
        print(f"  Mean: {model_pose.mean():.6f}")
        print(f"  Std: {model_pose.std():.6f}")
        print(f"  Shape: {model_pose.shape}")
        print(f"  Requires grad: {model_pose.requires_grad}")

        # Check if they match
        if ckpt_key == model_key:
            if torch.allclose(ckpt_pose, model_pose, atol=1e-6):
                print(f"\n  ✅ Checkpoint pose successfully loaded")
            else:
                print(f"\n  ❌ WARNING: Pose values differ!")
                print(f"     Difference: {(ckpt_pose - model_pose).abs().mean():.8f}")
        else:
            print(f"\n  ⚠️ Key names don't match ('{ckpt_key}' vs '{model_key}')")
    else:
        print("\n  ⚠️ Could not find pose parameters to compare")

    print("=" * 60 + "\n")
    # ========== END VERIFICATION ==========

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

    # ========== SAMPLING FOR FAST TESTING (ADD THIS BLOCK) ==========
    SAMPLE_EVERY_N_FRAMES = 100  # Change this to 5, 10, or 20

    print(f"\n{'=' * 60}")
    print(f"FAST TESTING MODE: Sampling every {SAMPLE_EVERY_N_FRAMES} frames")
    print(f"{'=' * 60}")

    # Create sampled indices
    original_length = len(base_dataset)
    sampled_indices = list(range(0, original_length, SAMPLE_EVERY_N_FRAMES))

    # Create a subset dataset using Subset wrapper
    base_dataset = Subset(base_dataset, sampled_indices)

    print(f"  Original frames: {original_length}")
    print(f"  Sampled frames:  {len(sampled_indices)} ({len(sampled_indices) / original_length * 100:.1f}%)")
    print(f"  Speed multiplier: ~{SAMPLE_EVERY_N_FRAMES}x faster")
    print(f"  Estimated time: {240 / SAMPLE_EVERY_N_FRAMES:.1f} minutes (was ~240 min)")
    print(f"{'=' * 60}\n")

    # IMPORTANT: Store for evaluation script
    SAMPLED_FRAME_INDICES = np.array(sampled_indices)  # ADD THIS LINE
    TOTAL_FRAMES_IN_SEQUENCE = original_length          # ADD THIS LINE
    # ============== END OF SAMPLING BLOCK ==============

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

            if batch_idx == 0:  # First batch only
                print("\n" + "="*70)
                print("POSE SOURCE TRACING - BEFORE validation_step")
                print("="*70)

                # Check what's in batch before validation_step
                print("\n1. Batch pose (from dataset):")
                if 'gt.right.hand_pose' in batch_cuda:
                    gt_pose = batch_cuda['gt.right.hand_pose']
                    print(f"   Shape: {gt_pose.shape}")
                    print(f"   Mean: {gt_pose.mean():.6f}")
                    print(f"   First 5: {gt_pose[0, :5]}")

                # Check model's stored parameters
                print("\n2. Model checkpoint params.pose.weight:")
                for name, param in model.named_parameters():
                    if 'params.pose.weight' in name:
                        print(f"   {name}:")
                        print(f"   Mean: {param.mean():.6f}")
                        print(f"   Frame 0 first 5: {param[0, :5]}")
                        break

                # Check what node.params() returns
                print("\n3. Calling node.params(batch['idx']):")
                try:
                    if isinstance(model, HOLD):
                        for node_name, node in model.nodes.items():
                            if node_name == 'right':
                                returned_params = node.params(batch_cuda['idx'])
                                if 'hand_pose' in returned_params:
                                    hp = returned_params['hand_pose']
                                    print(f"   Returned hand_pose shape: {hp.shape}")
                                    print(f"   Returned hand_pose mean: {hp.mean():.6f}")
                                    print(f"   Returned hand_pose first 5: {hp[0, :5] if hp.ndim > 1 else hp[:5]}")
                                elif 'full_pose' in returned_params:
                                    fp = returned_params['full_pose']
                                    print(f"   Returned full_pose shape: {fp.shape}")
                                    print(f"   Returned full_pose mean: {fp.mean():.6f}")
                except Exception as e:
                    print(f"   Error calling node.params(): {e}")

                print("="*70 + "\n")

            try:
                output = model.validation_step(batch_cuda)

                # ========== CHECK: What pose is in the output? ==========
                if batch_idx == 0:
                    print("\n" + "="*70)
                    print("POSE SOURCE TRACING - AFTER validation_step")
                    print("="*70)

                    # Check output
                    for key in ['hand_pose', 'full_pose', 'right.hand_pose']:
                        if key in output:
                            pose_out = output[key]
                            print(f"\nOutput['{key}']:")
                            print(f"   Shape: {pose_out.shape}")
                            print(f"   Mean: {pose_out.mean():.6f}")
                            print(f"   First 5: {pose_out[0, :5] if pose_out.ndim > 1 else pose_out[:5]}")

                    print("="*70 + "\n")
                # ========== END CHECK ==========
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

                # ============================================================
                # CRITICAL FIX: Subset to HO3D's 16 joints BEFORE any transforms
                # ============================================================
                HO3D_JOINT_INDICES = [0, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]
                print(f"[FIX] Subsetting joints from {hand_joints.shape} to 16 joints...")
                hand_joints = hand_joints[:, HO3D_JOINT_INDICES, :]  # [1, 21, 3] → [1, 16, 3]
                print(f"[FIX] After subsetting: {hand_joints.shape}")
                # ============================================================

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

    # ========== DIAGNOSTIC: Compare deformation space statistics ==========
    print(f"\n{'='*60}")
    print(f"DEFORMATION SPACE STATISTICS")
    print(f"{'='*60}")
    print(f"Hand vertices (deformation space):")
    print(f"  Mean: {hand_verts.mean():.6f}")
    print(f"  Std:  {hand_verts.std():.6f}")
    print(f"  Range: [{hand_verts.min():.6f}, {hand_verts.max():.6f}]")
    print(f"  Frame 0 mean: {hand_verts[0].mean():.6f}")
    print(f"  Frame 0 range: [{hand_verts[0].min():.6f}, {hand_verts[0].max():.6f}]")
    print(f"\nHand joints (deformation space):")
    print(f"  Mean: {hand_joints.mean():.6f}")
    print(f"  Std:  {hand_joints.std():.6f}")
    print(f"  Wrist (joint 0) positions:")
    for i in range(min(5, len(hand_joints))):
        print(f"    Frame {i}: {hand_joints[i, 0]}")
    print(f"{'='*60}\n")
    # ========== END DIAGNOSTIC ==========

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
        'sampled_frame_indices': SAMPLED_FRAME_INDICES if SAMPLE_EVERY_N_FRAMES > 1 else None,  # ADD
        'total_frames_in_sequence': TOTAL_FRAMES_IN_SEQUENCE if SAMPLE_EVERY_N_FRAMES > 1 else None,  # ADD
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
    --output logs/evaluation_results/MC1_stage1_e200_8layer_implicitnet_predictions_$(date +%Y%m%d_%H%M%S).pkl \
    2>&1 | tee logs/evaluation_results/MC1_stage1_e200_8layer_implicitnet_extraction_$(date +%Y%m%d_%H%M%S).log


# Monitor the progress in a separate terminal
tail -f logs/evaluation_results/MC1_stage1_e200_8layer_implicitnet_extraction_debug_fresh.log | grep "Extracting:"

# Fast testing button:
When ready for final evaluation, simply change:
SAMPLE_EVERY_N_FRAMES = 10  # Fast testing
to:
SAMPLE_EVERY_N_FRAMES = 1   # Full extraction (all frames)
This gives you a single switch to control extraction speed without modifying multiple places in the code.
'''