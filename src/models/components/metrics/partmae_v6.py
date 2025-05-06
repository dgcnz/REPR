# from torchmetrics.wrappers import ClasswiseWrapper # inspired by ClasswiseWrapper
from torchmetrics import MeanMetric, Metric, MaxMetric, MeanSquaredError
from torchmetrics.wrappers.abstract import WrapperMetric
from torch import nn, Tensor
import torch
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__)


class V6Metrics(WrapperMetric):
    metrics: dict[str, Metric]

    def __init__(self, **metric_kwargs):
        super().__init__()
        self.metrics = nn.ModuleDict(
            {
                "loss_pose_intra_t": MeanMetric(**metric_kwargs),
                "loss_pose_inter_t": MeanMetric(**metric_kwargs),
                "loss_pose_intra_s": MeanMetric(**metric_kwargs),
                "loss_pose_inter_s": MeanMetric(**metric_kwargs),
                "loss_pose_t": MeanMetric(**metric_kwargs),
                "loss_pose_s": MeanMetric(**metric_kwargs),
                "loss_pose": MeanMetric(**metric_kwargs),
                "loss_patch": MeanMetric(**metric_kwargs),
                "loss_cos": MeanMetric(**metric_kwargs),
                "loss_dino": MeanMetric(**metric_kwargs),
                "loss_dino_comp": MeanMetric(**metric_kwargs),
                "loss_dino_expa": MeanMetric(**metric_kwargs),
                "loss": MeanMetric(**metric_kwargs),
                ### non-loss
                "pred_dt_std": MeanMetric(**metric_kwargs),
                "pred_ds_std": MeanMetric(**metric_kwargs),
                "gt_dt_std": MeanMetric(**metric_kwargs),
                "gt_ds_std": MeanMetric(**metric_kwargs),
                "rmse_pred_dt": MeanSquaredError(squared=True, **metric_kwargs),
                "rmse_pred_ds": MeanSquaredError(squared=True, **metric_kwargs),
                ### gradient norms
                "grad_mean_norm": MeanMetric(**metric_kwargs),
                "grad_max_norm": MaxMetric(**metric_kwargs),
            }
        )

    @torch.no_grad()
    def update(self, outputs: dict[str, Tensor]):
        """Update metrics from model outputs.

        Args:
            outputs: Dictionary of model outputs
            model: Optional model to compute gradient norms from. If provided,
                   it overrides the model set during initialization.
        """
        for k in self.metrics:
            if k.startswith("loss"):
                self.metrics[k].update(outputs[k])

        self.metrics["pred_dt_std"].update(outputs["pred_dT"][..., :2].std())
        self.metrics["pred_ds_std"].update(outputs["pred_dT"][..., 2:].std())
        self.metrics["gt_dt_std"].update(outputs["gt_dT"][..., :2].std())
        self.metrics["gt_ds_std"].update(outputs["gt_dT"][..., 2:].std())
        self.metrics["rmse_pred_dt"].update(
            outputs["pred_dT"][..., :2].flatten(), outputs["gt_dT"][..., :2].flatten()
        )
        self.metrics["rmse_pred_ds"].update(
            outputs["pred_dT"][..., 2:].flatten(), outputs["gt_dT"][..., 2:].flatten()
        )

        if "grad_norm" in outputs:
            self.metrics["grad_mean_norm"].update(outputs["grad_norm"].mean())
            self.metrics["grad_max_norm"].update(outputs["grad_norm"].max())

    def compute(self):
        return {k: v.compute() for k, v in self.metrics.items()}

    def reset(self):
        for k in self.metrics:
            self.metrics[k].reset()
