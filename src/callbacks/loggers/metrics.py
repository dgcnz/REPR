import lightning as L
import wandb
import matplotlib.pyplot as plt
import plotly.express as px
from src.callbacks.base_callback import BaseCallback
from lightning.pytorch.utilities import rank_zero_only


class MetricLogger(BaseCallback):
    def on_train_start(self, trainer, pl_module):
        pl_module.metrics["val/loss"].reset()
        pl_module.metrics["val/rmse"].reset()
        pl_module.metrics["val/rmse_best"].reset()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # update and log metrics
        self.compute_common(pl_module, outputs, "train")
        self.log_common(pl_module, outputs, "train", batch_idx)
        # intrinsically train logs
        if (
            hasattr(pl_module.net, "logit_scale")
            and pl_module.net.logit_scale_learnable
        ):
            pl_module.log(
                "train/logit_scale",
                pl_module.net.logit_scale,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        if pl_module.hparams.symmetry_penalty > 0:
            pl_module.log(
                "train/symmetry_loss",
                outputs["symmetry_loss"],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )


    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.compute_common(pl_module, outputs, "val")
        self.log_common(pl_module, outputs, "val", batch_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.compute_common(pl_module, outputs, "test")
        self.log_common(pl_module, outputs, "test", batch_idx)

    def on_validation_epoch_end(self, trainer, pl_module):
        mse = pl_module.metrics["val/rmse"].compute()
        pl_module.metrics["val/rmse_best"](mse)
        pl_module.log(
            "val/rmse_best",
            pl_module.metrics["val/rmse_best"],
            sync_dist=True,
            prog_bar=True,
            rank_zero_only=True,
        )

    def compute_common(self, pl_module: L.LightningModule, output: dict, stage: str):
        try:
            pl_module.metrics[f"{stage}/loss"](output["loss"])
            pl_module.metrics[f"{stage}/rmse"](
                output["pred_T"].flatten(0, 1), output["gt_T"].flatten(0, 1)
            )
        except Exception as e:
            print("ERROR ON COMPUTE COMMON")
            print(e.__traceback__)
            print(e)
            raise e

    def log_common(
        self, pl_module: L.LightningModule, output: dict, stage: str, batch_idx: int
    ):
        pl_module.log(
            f"{stage}/loss",
            pl_module.metrics[f"{stage}/loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        rmse_y, rmse_x = pl_module.metrics[f"{stage}/rmse"].compute()
        rmse = (rmse_y + rmse_x) / 2
        pl_module.log_dict(
            {
                f"{stage}/rmse_y": rmse_y,
                f"{stage}/rmse_x": rmse_x,
                f"{stage}/rmse": rmse,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        if batch_idx == 0:
            # TODO: Maybe move this to a separate callback, it throws an error:
            # https://github.com/pytorch/pytorch/issues/64947
            # pl_module.log(
            #     f"{stage}/transform_distribution/sample_median",
            #     output["pred_T"].median(),
            #     on_step=False,
            #     on_epoch=True,
            #     prog_bar=True,
            #     sync_dist=True,
            # )
            # pl_module.log(
            #     f"{stage}/transform_distribution/sample_iqr",
            #     output["pred_T"].float().quantile(0.75)
            #     - output["pred_T"].float().quantile(0.25),
            #     on_step=False,
            #     on_epoch=True,
            #     prog_bar=True,
            #     sync_dist=True,
            # )
            # In the meantime log the mean and std
            pl_module.log(
                f"{stage}/transform_distribution/sample_mean",
                output["pred_T"].mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            pl_module.log(
                f"{stage}/transform_distribution/sample_std",
                output["pred_T"].std(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
