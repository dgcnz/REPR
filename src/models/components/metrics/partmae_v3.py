# from torchmetrics.wrappers import ClasswiseWrapper # inspired by ClasswiseWrapper
from torchmetrics import MeanMetric, Metric
from torchmetrics.wrappers.abstract import WrapperMetric
from torch import nn, Tensor


class V3Metrics(WrapperMetric):
    metrics: dict[str, Metric]

    def __init__(self):
        super().__init__()
        self.metrics = nn.ModuleDict(
            {
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
        )

    def update(self, outputs: dict[str, Tensor]):
        """Update metrics from model outputs."""
        for k in self.metrics:
            if k.startswith("loss"):
                self.metrics[k].update(outputs[k])

        self.metrics["pred_dt_std"].update(outputs["pred_dT"][..., :2].std())
        self.metrics["pred_ds_std"].update(outputs["pred_dT"][..., 2:].std())
        self.metrics["gt_dt_std"].update(outputs["gt_dT"][..., :2].std())
        self.metrics["gt_ds_std"].update(outputs["gt_dT"][..., 2:].std())

    def compute(self):
        return {k: v.compute() for k, v in self.metrics.items()}

    def reset(self):
        for k in self.metrics:
            self.metrics[k].reset()
