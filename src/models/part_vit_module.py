from types import SimpleNamespace
from typing import Any, Dict, Tuple, Callable
import torch
from torch import Tensor
import lightning as L
from torchmetrics import MinMetric, MeanMetric
from torchmetrics.regression import MeanSquaredError
from src.data.components.sampling_utils import sample_and_stitch
from src.models.components.utils.part_utils import (
    compute_gt_transform,
    get_all_pairs,
    compute_symmetry_loss,
)
from src.utils import pylogger
from jaxtyping import Float, Int


class PARTViTModule(L.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        patch_size: int,
        criterion_fn: Callable,
        sample_mode: str = "offgrid",
        symmetry_penalty: float = 0.0,
        pair_sampler: Callable[[int, int, str], Int[Tensor, "B NP 2"]] = get_all_pairs,
        scheduler_interval: str = "step",  # previously "epoch"
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = net

        self.criterion = criterion_fn()

        self.metrics = torch.nn.ModuleDict(
            {
                "train/rmse": MeanSquaredError(squared=False, num_outputs=2),
                "train/loss": MeanMetric(),
                "val/loss": MeanMetric(),
                "val/rmse": MeanSquaredError(squared=False, num_outputs=2),
                "val/rmse_best": MinMetric(),
                "test/rmse": MeanSquaredError(squared=False, num_outputs=2),
                "test/loss": MeanMetric(),
            }
        )
        self.cli_logger = pylogger.RankedLogger(__name__, rank_zero_only=True)
        self.pair_sampler = pair_sampler
        self.cache = SimpleNamespace()

    def forward(
        self, x: torch.Tensor, patch_pair_indices: torch.Tensor
    ) -> torch.Tensor:
        return self.net(x, patch_pair_indices)

    def patch_info(self, x: Tensor):
        img_size = x.shape[-2:]
        patch_size = self.hparams.patch_size
        # assume divisible by patch_size
        n_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        return patch_size, n_patches, img_size

    def model_step(
        self, batch: tuple[Tensor, Tensor] | dict[str, Tensor]
    ) -> dict[str, Tensor]:
        idx = 0 if isinstance(batch, tuple) else "image"
        x_original = batch[idx]
        # get input
        x, patch_positions = sample_and_stitch(
            x_original, self.hparams.patch_size, self.hparams.sample_mode
        )
        batch_size, n_patches, _ = patch_positions.shape
        patch_pair_indices = self.pair_sampler(batch_size, n_patches, device=x.device)
        # compute prediction
        pred_T = self.forward(x, patch_pair_indices)
        # compute loss
        gt_T = compute_gt_transform(patch_pair_indices, patch_positions)

        # normalize transformations to [-1, 1] to improve training stability
        pred_T = pred_T / self.net.logit_scale
        gt_T = gt_T / self.net.logit_scale

        loss = self.criterion(pred_T, gt_T)  # 1d tensor
        symmetry_loss = None
        if self.hparams.symmetry_penalty > 0:
            symmetry_loss = compute_symmetry_loss(patch_pair_indices, pred_T, n_patches)
            symmetry_loss = symmetry_loss * self.hparams.symmetry_penalty
            loss += symmetry_loss

        pred_T = pred_T * self.net.logit_scale
        gt_T = gt_T * self.net.logit_scale

        return {
            "loss": loss,
            "symmetry_loss": symmetry_loss,
            "pred_T": pred_T,
            "gt_T": gt_T,
            "x": x,
            "x_original": x_original,
            "patch_positions": patch_positions,
            "patch_pair_indices": patch_pair_indices,
        }

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


if __name__ == "__main__":
    _ = PARTViTModule(None, None, None, None)
