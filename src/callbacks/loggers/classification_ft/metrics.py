import lightning as L
from src.callbacks.base_callback import BaseCallback


class ClassificationFTMetricLogger(BaseCallback):
    def on_train_start(self, trainer, pl_module):
        pl_module.metrics["val/loss"].reset()
        pl_module.metrics["val/acc"].reset()
        pl_module.metrics["val/acc_best"].reset()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.compute_common(pl_module, outputs, "train")
        self.log_common(pl_module, outputs, "train", batch_idx)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.compute_common(pl_module, outputs, "val")
        self.log_common(pl_module, outputs, "val", batch_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.compute_common(pl_module, outputs, "test")
        self.log_common(pl_module, outputs, "test", batch_idx)

    def on_train_epoch_start(self, trainer, pl_module):
        pl_module.cli_logger.info("Train epoch started.")

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.cli_logger.info("Train epoch ended.")

    def on_validation_epoch_start(self, trainer, pl_module):
        pl_module.cli_logger.info("Validation epoch started.")

    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.cli_logger.info("Validation epoch ended.")
        acc = pl_module.metrics["val/acc"].compute()
        pl_module.metrics["val/acc_best"](acc)
        pl_module.log(
            "val/acc_best",
            pl_module.metrics["val/acc_best"].compute(),
            sync_dist=True,
            prog_bar=True,
        )

    def _log_acc(self, output, stage):
        mixup_on = (output["preds"].ndim == 1 and output["targets"].ndim > 1) 
        return not (stage == "train" and mixup_on)

    def compute_common(self, pl_module: L.LightningModule, output: dict, stage: str):
        pl_module.metrics[f"{stage}/loss"](output["loss"])
        if self._log_acc(output, stage):
            pl_module.metrics[f"{stage}/acc"](output["preds"], output["targets"])

    def log_common(
        self, pl_module: L.LightningModule, output: dict, stage: str, batch_idx: int
    ):
        metric_dict = {
            f"{stage}/loss": pl_module.metrics[f"{stage}/loss"],
        }
        if self._log_acc(output, stage):
            metric_dict[f"{stage}/acc"] = pl_module.metrics[f"{stage}/acc"]

        pl_module.log_dict(
            metric_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )