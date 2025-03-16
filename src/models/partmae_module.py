from types import SimpleNamespace
from typing import Any, Dict, Tuple
import torch
from torch import Tensor
import lightning as L
from torchmetrics import MinMetric, MeanMetric
from torchmetrics.regression import MeanSquaredError
from src.utils import pylogger


class PARTModule(L.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        scheduler_interval: str = "step",  # previously "epoch"
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = net
        self.cache = SimpleNamespace()

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        return self.net(x)

    def model_step(
        self, batch: tuple[Tensor, Tensor] | dict[str, Tensor]
    ) -> dict[str, Tensor]:
        idx = 0 if isinstance(batch, tuple) else "image"
        out = self.forward(batch[idx])
        return out

    def training_step(
        self, batch: tuple[Tensor, Tensor] | dict[str, Tensor], batch_idx: int
    ) -> Tensor:
        return self.model_step(batch)

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        return self.model_step(batch)

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        return self.model_step(batch)

    def configure_optimizers(self) -> Dict[str, Any]:
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

    @classmethod
    def save_backbone(cls, module_ckpt_path: str):
        out_path = module_ckpt_path.replace(".ckpt", "_backbone.ckpt")
        torch.save(
            cls.load_from_checkpoint(module_ckpt_path).net.backbone.state_dict(), out_path
        )
        return out_path
