from typing import Any
from lightning import Fabric
from src.utils import pylogger
from torchmetrics import MeanMetric
from fvcore.common.timer import Timer

log = pylogger.RankedLogger(__name__)


class IterTimer(object):
    """Callback for measuring and logging time between iterations."""
    
    def __init__(self, every_n_steps: int = 100):
        self.every_n_steps = every_n_steps
        self.timer = Timer()
        self.mean_iter_time = MeanMetric(sync_on_compute=False)
        assert every_n_steps > 0, "every_n_steps must be greater than 0"

    def on_train_start(self, fabric: Fabric, **kwargs) -> None:
        """Reset timer at the start of training."""
        self.timer.reset()
        self.mean_iter_time.reset()

    def _log(self, fabric: Fabric, global_step: int, metrics: dict[str, Any]):
        """Log metrics if this process is the global zero rank."""
        if fabric.is_global_zero:
            fabric.log_dict(metrics, step=global_step)

    def on_train_batch_end(
        self,
        fabric: Fabric,
        global_step: int,
        batch_idx: int,
        **kwargs,
    ) -> None:
        """Measure time since last batch end and log it."""
        # Only track times after the first batch to avoid cache/compile overhead
        iter_time = self.timer.seconds()
        self.timer.reset()
        if batch_idx == 0:
            return

        # Update running average
        self.mean_iter_time.update(iter_time)
        
        # Log the current and average iteration times
        if batch_idx % self.every_n_steps == 0:
            self._log(fabric, global_step, {
                "train/iter_time": self.mean_iter_time.compute()
            })
