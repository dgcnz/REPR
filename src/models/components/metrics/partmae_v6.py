# from torchmetrics.wrappers import ClasswiseWrapper # inspired by ClasswiseWrapper
from torchmetrics import MeanMetric, Metric, MaxMetric, MeanSquaredError
from torchmetrics.wrappers.abstract import WrapperMetric
from torch import nn, Tensor
import torch
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__)


class V6Metrics(WrapperMetric):
    metrics: dict[str, Metric]

    def __init__(self, nan_strategy='disable', sync_on_compute = False):
        super().__init__()
        mean_kwargs = {
            "nan_strategy": nan_strategy,
            "sync_on_compute": sync_on_compute,
        }
        mse_kwargs = {"sync_on_compute": sync_on_compute}
        self.metrics = nn.ModuleDict(
            {
                "loss_pose_intra_t": MeanMetric(**mean_kwargs),
                "loss_pose_inter_t": MeanMetric(**mean_kwargs),
                "loss_pose_intra_s": MeanMetric(**mean_kwargs),
                "loss_pose_inter_s": MeanMetric(**mean_kwargs),
                "loss_pose_t": MeanMetric(**mean_kwargs),
                "loss_pose_s": MeanMetric(**mean_kwargs),
                "loss_pose": MeanMetric(**mean_kwargs),
                # "loss_psmooth": MeanMetric(**mean_kwargs),
                # "loss_pstress": MeanMetric(**mean_kwargs),
                "loss_pmatch": MeanMetric(**mean_kwargs),
                "loss_pmatch_cos_sim": MeanMetric(**mean_kwargs),
                "loss_pmatch_mindist_max": MeanMetric(**mean_kwargs),
                "loss_pmatch_mindist_mean": MeanMetric(**mean_kwargs),
                "loss_pcr": MeanMetric(**mean_kwargs),
                # "loss_cos": MeanMetric(**mean_kwargs),
                "loss_ccr": MeanMetric(**mean_kwargs),
                "loss_cinv": MeanMetric(**mean_kwargs),
                "loss": MeanMetric(**mean_kwargs),
                ### non-loss
                "pred_dt_std": MeanMetric(**mean_kwargs),
                "pred_ds_std": MeanMetric(**mean_kwargs),
                "gt_dt_std": MeanMetric(**mean_kwargs),
                "gt_ds_std": MeanMetric(**mean_kwargs),
                "rmse_pred_dt": MeanSquaredError(squared=True, **mse_kwargs),
                "rmse_pred_ds": MeanSquaredError(squared=True, **mse_kwargs),
                ### gradient norms
                "grad_mean_norm": MeanMetric(**mean_kwargs),
                "grad_max_norm": MaxMetric(**mean_kwargs),
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
            if k.startswith("loss") and k in outputs:
                self.metrics[k].update(outputs[k])

        if "pred_dT" in outputs and "gt_dT" in outputs:
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
