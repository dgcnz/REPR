from types import SimpleNamespace
from typing import Any, Callable
import torch
from torch import Tensor
import lightning as L
from torchmetrics import MaxMetric, MeanMetric, Accuracy
from src.utils import pylogger
from jaxtyping import Float, Int
from functools import partial


class ImageClassificationModule(L.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        criterion_fn: Callable,
        num_classes: int,
        scheduler_interval: str = "epoch",
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = criterion_fn()

        # metric objects for calculating and averaging accuracy across batches
        self.metrics = torch.nn.ModuleDict(
            {
                "train/loss": MeanMetric(),
                "train/acc": Accuracy(task="multiclass", num_classes=num_classes),
                "val/loss": MeanMetric(),
                "val/acc": Accuracy(task="multiclass", num_classes=num_classes),
                "val/acc_best": MaxMetric(),
                "test/loss": MeanMetric(),
                "test/acc": Accuracy(task="multiclass", num_classes=num_classes),
            }
        )
        self.cli_logger = pylogger.RankedLogger(__name__, rank_zero_only=True)
        self.cache = SimpleNamespace()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def model_step(
        self, batch: tuple[Tensor, Tensor] | dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A dictionary containing the loss and other metrics.
        """
        xid, yid = (0, 1) if isinstance(batch, (tuple, list)) else ("image", "label")
        x, y = batch[xid], batch[yid]

        logits = self.forward(x)
        preds = torch.argmax(logits, dim=1)
        loss = self.criterion(logits, y)

        return {
            "loss": loss,
            "preds": preds,
            "targets": y
        }

    def training_step(
        self, batch: tuple[Tensor, Tensor] | dict[str, Tensor], batch_idx: int
    ) -> Tensor:
        return self.model_step(batch)

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        return self.model_step(batch)

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        return self.model_step(batch)

    def configure_optimizers(self) -> dict[str, Any]:
        if hasattr(self.hparams, "lr"):
            self.hparams.optimizer = partial(self.hparams.optimizer, lr=self.hparams.lr)
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": self.hparams.scheduler_interval,
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)
