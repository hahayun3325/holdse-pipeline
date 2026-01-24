import numpy as np
import torch
import json
import src.utils.eval_modules as eval_m
import src.utils.io.gt as gt

device = "cuda:0"

eval_fn_dict = {
    "mpjpe_ra_r": eval_m.eval_mpjpe_right,
    "mrrpe_ho": eval_m.eval_mrrpe_ho_right,
    "cd_f_ra": eval_m.eval_cd_f_ra,
    "cd_f_right": eval_m.eval_cd_f_right,
}


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sd_p", type=str, required=False)
    args = parser.parse_args()
    from easydict import EasyDict

    args = EasyDict(vars(args))
    return args


def main():
    from tqdm import tqdm

    args = parse_args()
    import src.utils.io.ours as ours

    data_pred = ours.load_data(args.sd_p)
    print(f"[DEBUG] data_pred keys: {list(data_pred.keys())}")
    # ADD THIS:
    print(f"\n[DEBUG] full_seq_name value: '{data_pred.get('full_seq_name', 'KEY NOT FOUND')}'")
    print(f"[DEBUG] full_seq_name type: {type(data_pred.get('full_seq_name'))}")
    if 'full_seq_name' in data_pred:
        fsn = data_pred['full_seq_name']
        print(f"[DEBUG] full_seq_name split('_'): {fsn.split('_')}")
        print(f"[DEBUG] Number of parts after split: {len(fsn.split('_'))}")
    for key, value in data_pred.items():
        if hasattr(value, 'shape'):
            print(f"[DEBUG] data_pred['{key}'] shape: {value.shape}")
        elif isinstance(value, (list, tuple)):
            print(f"[DEBUG] data_pred['{key}'] length: {len(value)}")
            if len(value) > 0 and hasattr(value[0], 'shape'):
                print(f"[DEBUG]   first element shape: {value[0].shape}")

    print("\n[DEBUG] Loading GT data...")
    data_gt = gt.load_data(data_pred["full_seq_name"])
    print(f"[DEBUG] data_gt keys: {list(data_gt.keys())}")
    for key, value in data_gt.items():
        if hasattr(value, 'shape'):
            print(f"[DEBUG] data_gt['{key}'] shape: {value.shape}")
        elif isinstance(value, (list, tuple)):
            print(f"[DEBUG] data_gt['{key}'] length: {len(value)}")
            if len(value) > 0 and hasattr(value[0], 'shape'):
                print(f"[DEBUG]   first element shape: {value[0].shape}")

    # Downsample predictions to match GT joint count
    if data_pred["j3d_ra.right"].shape[1] != data_gt["j3d_ra.right"].shape[1]:
        num_gt_joints = data_gt["j3d_ra.right"].shape[1]
        print(f"[INFO] Downsampling predictions from {data_pred['j3d_ra.right'].shape[1]} to {num_gt_joints} joints")

        # Extract sliced tensors
        j3d_ra_downsampled = data_pred["j3d_ra.right"][:, :num_gt_joints, :]
        j3d_c_downsampled = data_pred["j3d_c.right"][:, :num_gt_joints, :]
        jnts_downsampled = data_pred["jnts.right"][:, :num_gt_joints, :]

        # Delete original keys
        del data_pred["j3d_ra.right"]
        del data_pred["j3d_c.right"]
        del data_pred["jnts.right"]

        # Add downsampled versions
        data_pred["j3d_ra.right"] = j3d_ra_downsampled
        data_pred["j3d_c.right"] = j3d_c_downsampled
        data_pred["jnts.right"] = jnts_downsampled

    seq_name = data_pred["full_seq_name"]
    out_p = args.sd_p
    eval_fn_dict["icp"] = eval_m.eval_icp_first_frame

    print("------------------")
    print("Involving the following eval_fn:")
    for eval_fn_name in eval_fn_dict.keys():
        print(eval_fn_name)
    print("------------------")

    # Initialize the metrics dictionaries
    metric_dict = {}
    # Evaluate each metric using the corresponding function
    pbar = tqdm(eval_fn_dict.items())
    for eval_fn_name, eval_fn in pbar:
        pbar.set_description(f"Evaluating {eval_fn_name}")
        metric_dict = eval_fn(data_pred, data_gt, metric_dict)

    # Dictionary to store mean values of metrics
    mean_metrics = {}

    # Print out the mean of each metric and store the results
    for metric_name, values in metric_dict.items():
        mean_value = float(
            np.nanmean(values)
        )  # Convert mean value to native Python float
        mean_metrics[metric_name] = mean_value

    # sort by key
    mean_metrics = dict(sorted(mean_metrics.items(), key=lambda item: item[0]))

    for metric_name, mean_value in mean_metrics.items():
        print(f"{metric_name.upper()}: {mean_value:.2f}")

    # Define the file paths
    json_path = out_p + ".metric.json"
    npy_path = out_p + ".metric_all.npy"

    from datetime import datetime

    current_time = datetime.now()
    time_str = current_time.strftime("%m-%d %H:%M")
    mean_metrics["timestamp"] = time_str
    mean_metrics["seq_name"] = seq_name
    print("Units: CD (cm**2), F-score (percentage), MPJPE (mm)")

    # Save the mean_metrics dictionary to a JSON file with indentation
    with open(json_path, "w") as f:
        json.dump(mean_metrics, f, indent=4)
        print(f"Saved mean metrics to {json_path}")

    # Save the metric_all numpy array
    np.save(npy_path, metric_dict)
    print(f"Saved metric_all numpy array to {npy_path}")


if __name__ == "__main__":
    main()
'''
python evaluate.py --sd_p logs/694416964/checkpoints/last.ckpt 2>&1 | tee logs/694416964/evaluation_$(date +%Y%m%d_%H%M%S).log
'''