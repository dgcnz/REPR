from typing import Any, Dict, Optional
import torch
from lightning import Fabric
from torchmetrics import MeanMetric
from torchmetrics.regression import MeanSquaredError


class PARTMAEv2MetricLoggerCallback:
    """Callback for tracking and logging metrics specific to PART-MAE-v2."""

    def __init__(self):
        """Initialize the PART-MAE-v2 metric logger."""
        self.train_loss = MeanMetric()
        self.train_rmse = MeanSquaredError(squared=False, num_outputs=2)
        self.transform_std = MeanMetric()
        self.current_lr = None

    def on_train_start(self, fabric: Fabric, **kwargs) -> None:
        """Initialize metrics at the start of training."""
        self.train_loss = self.train_loss.to(fabric.device)
        self.train_rmse = self.train_rmse.to(fabric.device)
        self.transform_std = self.transform_std.to(fabric.device)

    def _log(self, fabric: Fabric, global_step: int, metrics: Dict[str, Any]):
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
        """Reset metrics and log current learning rate at the start of each training epoch."""
        self.train_loss.reset()
        self.train_rmse.reset()
        self.transform_std.reset()

        # Log learning rate at the start of each epoch since it's updated per epoch
        self._log_lr(fabric, global_step, optimizer)

    def update_metrics(self, outputs: Dict[str, torch.Tensor]):
        self.train_loss(outputs["loss"])
        self.train_rmse(outputs["pred_T"].flatten(0, 1), outputs["gt_T"].flatten(0, 1))
        self.transform_std(outputs["pred_T"].std().item())

    def get_current_metrics(self):
        rmse_y, rmse_x = self.train_rmse.compute()
        return {
            "loss": self.train_loss.compute(),
            "rmse": (rmse_y + rmse_x) / 2,
            "rmse_y": rmse_y,
            "rmse_x": rmse_x,
            "transform_std": self.transform_std.compute(),
        }

    def on_train_batch_end(self, outputs: Dict, **kwargs) -> None:
        self.update_metrics(outputs)

    def on_train_epoch_end(self, fabric: Fabric, global_step: int, **kwargs) -> None:
        """Log summary metrics at the end of each training epoch."""
        final_metrics = self.get_current_metrics()
        train_metrics = {
            "train/loss": final_metrics["loss"],
            "train/rmse": final_metrics["rmse"],
            "train/rmse_y": final_metrics["rmse_y"],
            "train/rmse_x": final_metrics["rmse_x"],
        }
        self._log(fabric, global_step, train_metrics)
