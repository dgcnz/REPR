from typing import Any, Dict
import torch
from lightning import Fabric
from src.utils import pylogger
from torchmetrics import Metric

log = pylogger.RankedLogger(__name__)


class MetricLogger(object):
    """Callback for tracking and logging metrics specific to PART-MAE-v3."""
    # TODO: add support for "log lr" and "log grad norm" options

    def on_train_start(
        self, fabric: Fabric, metric_collection: Metric, **kwargs
    ) -> None:
        """Initialize metrics at the start of training."""
        # TODO: maybe needs a .to(fabric.device)
        metric_collection.reset()

    def _log(self, fabric: Fabric, global_step: int, metrics: dict[str, Any]):
        if fabric.is_global_zero:
            fabric.log_dict(metrics, step=global_step)
        # fabric.barrier(f"metric_logger:_log:{global_step}")

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

    def on_train_batch_end(
        self, outputs: Dict, metric_collection: Metric, **kwargs
    ) -> None:
        """Update metrics at the end of each training batch."""
        metric_collection.update(outputs)

    def on_train_epoch_end(
        self,
        fabric: Fabric,
        global_step: int,
        epoch: int,
        metric_collection: Metric,
        **kwargs,
    ) -> None:
        final_metrics: dict = metric_collection.compute()
        train_metrics = {f"train/{k}": v for k, v in final_metrics.items()}
        train_metrics["epoch"] = epoch
        self._log(fabric, global_step, train_metrics)
        metric_collection.reset()
