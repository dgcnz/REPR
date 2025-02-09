import lightning as L
import wandb
import matplotlib.pyplot as plt
import plotly.express as px
from src.callbacks.base_callback import BaseCallback
from lightning.pytorch.utilities import rank_zero_only


class MetricLogger(BaseCallback):
    def on_train_start(self, trainer, pl_module):
        pl_module.cli_logger.info("Training started.")
        pl_module.metrics["val/loss"].reset()
        pl_module.metrics["val/rmse"].reset()
        pl_module.metrics["val/rmse_best"].reset()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # update and log metrics
        self.compute_common(pl_module, outputs, "train")
        self.log_common(pl_module, outputs, "train", batch_idx)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.compute_common(pl_module, outputs, "val")
        self.log_common(pl_module, outputs, "val", batch_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.compute_common(pl_module, outputs, "test")
        self.log_common(pl_module, outputs, "test", batch_idx)

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.cli_logger.info("Train epoch ended.")

    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.cli_logger.info("Validation epoch ended.")
        # TODO: Sometimes this is called before the validation batch, I think 
        # it's because of the sanity check.
        mse = pl_module.metrics["val/rmse"].compute()
        pl_module.metrics["val/rmse_best"](mse)
        pl_module.log(
            "val/rmse_best",
            pl_module.metrics["val/rmse_best"],
            sync_dist=True,
            prog_bar=True,
            on_epoch=True,
            rank_zero_only=True,
        )

    def compute_common(self, pl_module: L.LightningModule, output: dict, stage: str):
        pl_module.metrics[f"{stage}/loss"](output["loss"])
        pl_module.metrics[f"{stage}/rmse"](
            output["pred_T"].flatten(0, 1), output["gt_T"].flatten(0, 1)
        )

    def log_common(
        self, pl_module: L.LightningModule, output: dict, stage: str, batch_idx: int
    ):
        common_kwargs = dict(on_step=False, on_epoch=True, sync_dist=True)
        rmse_y, rmse_x = pl_module.metrics[f"{stage}/rmse"].compute()
        rmse = (rmse_y + rmse_x) / 2
        pl_module.log_dict(
            {
                f"{stage}/loss": pl_module.metrics[f"{stage}/loss"],
                f"{stage}/rmse": rmse,
            },
            **common_kwargs,
            prog_bar=True,
        )
        pl_module.log_dict(
            {
                f"{stage}/rmse_y": rmse_y,
                f"{stage}/rmse_x": rmse_x,
            },
            prog_bar=False, # not necessary to show this
            **common_kwargs,
        )

        if batch_idx == 0:
            pl_module.log_dict(
                {
                    f"{stage}/transform_distribution/sample_mean": output["pred_T"].mean(),
                    f"{stage}/transform_distribution/sample_std": output["pred_T"].std(),
                },
                prog_bar=False,
                **common_kwargs,
            )