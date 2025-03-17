from typing import Any, Dict
import torch
from lightning import Fabric
from torchmetrics import MeanMetric
from src.utils import pylogger

log = pylogger.RankedLogger(__name__)

class MetricLogger(object):
    """Callback for tracking and logging metrics specific to PART-MAE-v3."""

    def __init__(self):
        """Initialize the PART-MAE-v3 metric logger."""
        self.metrics = {
            "loss_intra_t": MeanMetric(),
            "loss_inter_t": MeanMetric(),
            "loss_intra_s": MeanMetric(),
            "loss_inter_s": MeanMetric(),
            "loss_t": MeanMetric(),
            "loss_s": MeanMetric(),
            "loss": MeanMetric(),
            ### non-loss
            "pred_dt_std": MeanMetric(),
            "pred_ds_std": MeanMetric(),
            "gt_dt_std": MeanMetric(),
            "gt_ds_std": MeanMetric(),
        }

    def on_train_start(self, fabric: Fabric, **kwargs) -> None:
        """Initialize metrics at the start of training."""
        for k in self.metrics:
            self.metrics[k] = self.metrics[k].to(fabric.device)
            self.metrics[k].reset()

    def _log(
        self, fabric: Fabric, global_step: int, metrics: Dict[str, Any]
    ):
        if fabric.is_global_zero:
            fabric.log_dict(metrics, step=global_step)

    def _log_lr(
        self, fabric: Fabric, global_step: int, optimizer: torch.optim.Optimizer
    ):
        lr = optimizer.param_groups[0]["lr"]
        self._log(fabric, global_step, {"train/lr": lr})

    def on_train_epoch_start(
        self,
        fabric: Fabric,
        global_step: int,
        optimizer: torch.optim.Optimizer,
        **kwargs,
    ) -> None:
        """Log current learning rate at the start of each training epoch."""
        self._log_lr(fabric, global_step, optimizer)

    def update_metrics(self, outputs: Dict[str, torch.Tensor]):
        """Update metrics from model outputs."""
        for k in self.metrics:
            if k.startswith("loss"):
                self.metrics[k](outputs[k])

        self.metrics["pred_dt_std"](outputs["pred_dT"][..., :2].std())
        self.metrics["pred_ds_std"](outputs["pred_dT"][..., 2:].std())
        self.metrics["gt_dt_std"](outputs["gt_dT"][..., :2].std())
        self.metrics["gt_ds_std"](outputs["gt_dT"][..., 2:].std())

    def get_current_metrics(self):
        """Compute and return all metrics."""
        return {k: v.compute() for k, v in self.metrics.items()}

    def on_train_batch_end(self, outputs: Dict, **kwargs) -> None:
        """Update metrics at the end of each training batch."""
        self.update_metrics(outputs)

    def on_train_epoch_end(
        self, fabric: Fabric, global_step: int, epoch: int, **kwargs
    ) -> None:
        """Log summary metrics at the end of each training epoch."""
        log.info(f"Computing epoch {epoch} metrics...")
        final_metrics = self.get_current_metrics()
        log.info(f"Metrics: {final_metrics}")
        train_metrics = {f"train/{k}": v for k, v in final_metrics.items()}
        train_metrics["epoch"] = epoch

        self._log(fabric, global_step, train_metrics)

        for k in self.metrics:
            self.metrics[k].reset()
