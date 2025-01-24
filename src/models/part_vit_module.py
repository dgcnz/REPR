import matplotlib
from types import SimpleNamespace

matplotlib.use("Agg")
from typing import Any, Dict, Tuple, Callable
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from copy import copy
import torch
import math
from torch import Tensor
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
from torchmetrics.regression import MeanSquaredError
from src.data.components.sampling_utils import sample_and_stitch
from src.models.components.part_utils import compute_gt_transform, get_all_pairs
import matplotlib.pyplot as plt
from src.utils.visualization.visualization import (
    create_provenance_grid,
    plot_provenance,
    plot_transform_distribution,
    plot_patch_pair_transform_matrices,
    plot_patch_pair_coverage,
    get_transforms_from_reference_patch,
)
from src.utils.visualization.reconstruction import (
    create_image_from_transforms,
    reconstruct_image_from_sampling,
)
from src.utils import pylogger
import wandb
from jaxtyping import Float, Int
from matplotlib.patches import Rectangle


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


class PARTViTModule(LightningModule):
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
        compile: bool = False,
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
        :return: A tensor of logits.
        """
        return self.net(x, patch_pair_indices)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.metrics["val/loss"].reset()
        self.metrics["val/rmse"].reset()
        self.metrics["val/rmse_best"].reset()
        if self.hparams.compile and self.trainer.accelerator == "cpu":
            self.cli_logger.warning(
                "Model compilation is enabled but the trainer is not using GPU. "
            )

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

    def log_common(self, output: dict, stage: str, batch_idx: int):
        self.metrics[f"{stage}/loss"](output["loss"])
        self.metrics[f"{stage}/rmse"](
            output["pred_T"].flatten(0, 1), output["gt_T"].flatten(0, 1)
        )
        self.log(
            f"{stage}/loss",
            self.metrics[f"{stage}/loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        rmse_y, rmse_x = self.metrics[f"{stage}/rmse"].compute()
        rmse = (rmse_y + rmse_x) / 2
        self.log_dict(
            {
                f"{stage}/rmse_y": rmse_y,
                f"{stage}/rmse_x": rmse_x,
                f"{stage}/rmse": rmse,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # TODO: might want to move this out to a callback if it slows down training
        if batch_idx == 0:
            self.log(
                f"{stage}/transform_distribution/sample_median",
                output["pred_T"].median(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                f"{stage}/transform_distribution/sample_iqr",
                output["pred_T"].float().quantile(0.75)
                - output["pred_T"].float().quantile(0.25),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

    def training_step(
        self, batch: tuple[Tensor, Tensor] | dict[str, Tensor], batch_idx: int
    ) -> Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        # forward pass
        output = self.model_step(batch)

        # update and log metrics
        self.log_common(output, "train", batch_idx)

        # intrinsically train logs
        if hasattr(self.net, "logit_scale") and self.net.logit_scale_learnable:
            self.log(
                "train/logit_scale", self.net.logit_scale, on_step=False, on_epoch=True
            )
        if self.hparams.symmetry_penalty > 0:
            self.log(
                "train/symmetry_loss",
                output["symmetry_loss"],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        return output

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        output = self.model_step(batch)

        # update and log metrics
        self.log_common(output, "val", batch_idx)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        mse = self.metrics["val/rmse"].compute()
        self.metrics["val/rmse_best"](mse)
        self.log(
            "val/rmse_best",
            self.metrics["val/rmse_best"],
            sync_dist=True,
            prog_bar=True,
        )

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        output = self.model_step(batch)

        # update and log metrics
        self.log_common(output, "test", batch_idx)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self._model_step = self.model_step
            self.model_step = torch.compile(self.model_step)

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
