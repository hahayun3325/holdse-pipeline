import torch
import torch.nn as nn
from torchmetrics.image import PeakSignalNoiseRatio
import common.comet_utils as comet_utils


class Metrics(nn.Module):
    def __init__(self, experiment):
        super().__init__()
        # metrics to evaluate
        self.metrics = ["psnr"]

        self.eval_fns = {
            "psnr": self.evaluate_psnr,
        }
        self.metric_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.experiment = experiment

    def evaluate_psnr(self, preds, targets):
        """
        Evaluate PSNR metric with proper memory management.

        ✅ CRITICAL: Ensures tensors are detached before computation
        """
        # ✅ CRITICAL: Detach pred_rgb to prevent gradient retention
        pred_rgb = preds["rgb"].detach()

        # ✅ FIX: Handle missing gt.rgb field for GHOP dataset
        if "gt.rgb" in targets:
            gt_rgb = targets["gt.rgb"].detach().view(-1, 3)
        elif "rgb" in targets:
            gt_rgb = targets["rgb"].detach().view(-1, 3)
        else:
            # No ground truth RGB - return zero metric
            return torch.tensor(0.0, device=pred_rgb.device)

        # Compute PSNR
        psnr = self.metric_psnr(pred_rgb, gt_rgb)

        # ✅ CRITICAL: Detach result before returning
        return psnr.detach()

    def forward(self, preds, targets, global_step, epoch):
        with torch.no_grad():
            metrics = {}
            for k in self.metrics:
                # ✅ CRITICAL: Evaluate metric
                metric_value = self.eval_fns[k](preds, targets)

                # ✅ CRITICAL: Detach and convert to Python scalar
                # This prevents gradient graph retention
                if isinstance(metric_value, torch.Tensor):
                    metrics["metrics/" + k] = metric_value.detach().cpu().item()
                else:
                    metrics["metrics/" + k] = metric_value

            comet_utils.log_dict(
                self.experiment, metrics, step=global_step, epoch=epoch
            )
            return metrics

    def reset(self):
        """
        Reset internal metric states.

        ✅ CRITICAL: Prevents metric history accumulation
        Called at epoch boundaries to clear cached states.
        """
        if hasattr(self.metric_psnr, 'reset'):
            self.metric_psnr.reset()