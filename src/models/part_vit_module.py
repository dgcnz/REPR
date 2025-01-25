from types import SimpleNamespace
from typing import Any, Dict, Tuple, Callable
import torch
from torch import Tensor
import lightning as L
from torchmetrics import MinMetric, MeanMetric
from torchmetrics.regression import MeanSquaredError
from src.data.components.sampling_utils import sample_and_stitch
from src.models.components.part_utils import compute_gt_transform, get_all_pairs
import matplotlib.pyplot as plt
from src.utils import pylogger
from jaxtyping import Float, Int


class PARTLoss(torch.nn.Module):
    def __init__(self, img_size: tuple[int, int], f: Callable | None = None):
        super().__init__()
        self.f = f
        self.img_size = img_size
        self.max_T = max(img_size)
        self.min_T = -self.max_T

    def forward(self, pred_T: Tensor, gt_T: Tensor):
        pred_T = (pred_T - self.min_T) / (self.max_T - self.min_T)
        gt_T = (gt_T - self.min_T) / (self.max_T - self.min_T)
        loss = self.f(torch.abs(pred_T - gt_T)).sum()
        return loss


def compute_symmetry_loss(
    patch_pair_indices: Int[Tensor, "B n_pairs 2"],
    transforms: Float[Tensor, "B n_pairs 2"],
    n_patches: int,
) -> Tensor:
    batch_size = patch_pair_indices.shape[0]
    T_matrix = torch.zeros(
        batch_size,
        n_patches,
        n_patches,
        2,
        device=transforms.device,
        dtype=transforms.dtype,
    )

    batch_idx = (
        torch.arange(batch_size, device=transforms.device)
        .unsqueeze(1)
        .repeat(1, patch_pair_indices.shape[1])
    )
    T_matrix[batch_idx, patch_pair_indices[..., 0], patch_pair_indices[..., 1]] = (
        transforms
    )

    symmetry_loss = (T_matrix + T_matrix.transpose(1, 2)).abs().mean()
    return symmetry_loss



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
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
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
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :param patch_pair_indices: A tensor of patch pair indices.
        :return: A tensor of logits.
        """
        return self.net(x, patch_pair_indices)

    def patch_info(self, x: Tensor):
        # return patch_size, num_patches, img_size
        img_size = x.shape[-2:]
        patch_size = self.hparams.patch_size
        # assume divisible by patch_size
        n_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        return patch_size, n_patches, img_size

    def model_step(
        self, batch: tuple[Tensor, Tensor] | dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A dictionary containing the loss and other metrics.
        """
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
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
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


if __name__ == "__main__":
    _ = PARTViTModule(None, None, None, None)
