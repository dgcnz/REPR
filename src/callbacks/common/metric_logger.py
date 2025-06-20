from typing import Any, Dict
import torch
from lightning import Fabric
from src.utils import pylogger
from torchmetrics import Metric

log = pylogger.RankedLogger(__name__)


class MetricLogger(object):
    """Callback for tracking and logging metrics specific to PART-MAE-v3."""

    def __init__(self, every_n_steps: int = 0):
        self.every_n_steps = every_n_steps
        self._has_updates = False

    def on_train_start(
        self, fabric: Fabric, metric_collection: Metric, **kwargs
    ) -> None:
        """Initialize metrics at the start of training."""
        metric_collection.reset()
        self._has_updates = False

    def _log(self, fabric: Fabric, global_step: int, metrics: dict[str, Any]):
        if fabric.is_global_zero:
            fabric.log_dict(metrics, step=global_step)

    def _log_lr(
        self, fabric: Fabric, global_step: int, optimizer: torch.optim.Optimizer
    ):
        metrics = {}
        for i, group in enumerate(optimizer.param_groups):
            key = "train/lr" if i == 0 else f"train/lr{i}"
            metrics[key] = group["lr"]
        self._log(fabric, global_step, metrics)

    def on_train_epoch_start(
        self,
        fabric: Fabric,
        global_step: int,
        optimizer: torch.optim.Optimizer,
        **kwargs,
    ) -> None:
        """Log current learning rate at the start of each training epoch."""
        self._log_lr(fabric, global_step, optimizer)
        self._has_updates = False

    def on_train_batch_end(
        self,
        fabric: Fabric,
        global_step: int,
        epoch: int,
        outputs: Dict,
        metric_collection: Metric,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
        **kwargs,
    ) -> None:
        """Update metrics at the end of each training batch."""
        metric_collection.update(outputs)
        self._has_updates = True
        if self.every_n_steps and batch_idx % self.every_n_steps == 0:
            self._compute_and_log(
                fabric,
                global_step,
                metric_collection,
                epoch,
                optimizer,
            )

    def on_train_epoch_end(
        self,
        fabric: Fabric,
        global_step: int,
        epoch: int,
        metric_collection: Metric,
        optimizer: torch.optim.Optimizer,
        **kwargs,
    ) -> None:
        self._compute_and_log(
            fabric,
            global_step,
            metric_collection,
            epoch,
            optimizer,
        )

    def _compute_and_log(
        self,
        fabric: Fabric,
        global_step: int,
        metric_collection: Metric,
        epoch: int,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Compute and log metrics."""
        if not self._has_updates:
            return

        final_metrics = metric_collection.compute()
        train_metrics = {f"train/{k}": v for k, v in final_metrics.items()}
        train_metrics["epoch"] = epoch
        self._log(fabric, global_step, train_metrics)
        self._log_lr(fabric, global_step, optimizer)
        metric_collection.reset()
        self._has_updates = False
