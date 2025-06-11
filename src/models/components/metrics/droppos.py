from torchmetrics import MeanMetric, Metric
from torchmetrics.wrappers.abstract import WrapperMetric
from torch import nn, Tensor
import torch
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__)

class DropPosMetrics(WrapperMetric):
    metrics: dict[str, Metric]

    def __init__(self, **metric_kwargs):
        super().__init__()
        self.metrics = nn.ModuleDict(
            {
                # Loss components
                "loss": MeanMetric(**metric_kwargs),
                "grad_norm": MeanMetric(**metric_kwargs),
                
                # For vanilla_drop_pos mode
                "acc1": MeanMetric(**metric_kwargs),
                
                # For mae_pos_target mode
                "acc_dp": MeanMetric(**metric_kwargs),
                
                # For multi_task mode
                "loss_mae": MeanMetric(**metric_kwargs),
                "loss_drop_pos": MeanMetric(**metric_kwargs),
            }
        )

    @torch.no_grad()
    def update(self, outputs: dict[str, Tensor]):
        """Update metrics from model outputs."""
        for k in outputs:
            if k in self.metrics:
                self.metrics[k].update(outputs[k])

    def compute(self):
        return {k: v.compute() for k, v in self.metrics.items()}

    def reset(self):
        for k in self.metrics:
            self.metrics[k].reset()
