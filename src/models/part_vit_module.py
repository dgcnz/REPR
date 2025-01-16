import matplotlib

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
    reconstruct_image_from_sampling,
    create_image_from_transforms,
    create_provenance_grid,
    plot_provenance,
    plot_transform_distribution,
    plot_patch_pair_transform_matrices,
    plot_patch_pair_coverage,
    get_transforms_from_reference_patch,
)
from src.utils import pylogger
import wandb
from jaxtyping import Float, Int
from matplotlib.patches import Rectangle


class PARTMAELossWeighted(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_T, gt_T):
        # x: Float[Tensor, "B P 2"]=  (pred_T - gt_T).abs()
        # return x.sum(2).mean()
        # weigh it higher if transformation is shorter
        x = (pred_T - gt_T).abs()
        return (x.sum(2) / (gt_T.abs().sum(2) + 1e-6)).mean()


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
        compile: bool,
        patch_size: int,
        sample_mode: str = "offgrid",
        symmetry_penalty: float = 0.0,
        pair_sampler: Callable[[int, int, str], Int[Tensor, "B NP 2"]] = get_all_pairs,
        criterion_fn: Callable = torch.nn.MSELoss,
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
        self.train_rmse = MeanSquaredError(squared=False, num_outputs=2)
        self.val_rmse = MeanSquaredError(squared=False, num_outputs=2)
        self.test_rmse = MeanSquaredError(squared=False, num_outputs=2)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_mse_best = MinMetric()
        self.cli_logger = pylogger.RankedLogger(__name__, rank_zero_only=True)
        self.pair_sampler = pair_sampler

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
        self.val_loss.reset()
        self.val_rmse.reset()
        self.val_mse_best.reset()
        if self.hparams.compile and self.trainer.accelerator == "cpu":
            self.cli_logger.warning(
                "Model compilation is enabled but the trainer is not using GPU. "
            )

    def norm_T(self, T: Tensor, img_size: tuple[int, int]) -> Tensor:
        # normalize to [0, 1]
        max_T = max(img_size)
        min_T = -max_T
        T = (T - min_T) / (max_T - min_T)
        return T

    def unnorm_T(self, T: Tensor, img_size: tuple[int, int]) -> Tensor:
        # unnormalize to [0, 1]
        max_T = max(img_size)
        min_T = -max_T
        T = T * (max_T - min_T) + min_T
        return T

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
        pred_T = self.norm_T(pred_T, img_size=(x.shape[-2], x.shape[-1]))
        gt_T = self.norm_T(gt_T, img_size=(x.shape[-2], x.shape[-1]))

        loss = mse = self.criterion(pred_T, gt_T)
        symmetry_loss = None
        if self.hparams.symmetry_penalty > 0:
            symmetry_loss = compute_symmetry_loss(patch_pair_indices, pred_T, n_patches)
            symmetry_loss = symmetry_loss * self.hparams.symmetry_penalty
            loss += symmetry_loss

        pred_T = self.unnorm_T(pred_T, img_size=(x.shape[-2], x.shape[-1]))
        gt_T = self.unnorm_T(gt_T, img_size=(x.shape[-2], x.shape[-1]))

        return {
            "loss": loss,
            "mse": mse,
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
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        output = self.model_step(batch)
        loss, preds, targets = output["loss"], output["pred_T"], output["gt_T"]

        # update and log metrics
        self.train_loss(loss)
        self.train_rmse(preds.flatten(0,1), targets.flatten(0,1))
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log_dict(
            dict(zip(["train/rmse_y", "train/rmse_x"], self.train_rmse.compute())),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        if hasattr(self.net, "logit_scale"):
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

        if (
            self.trainer.overfit_batches != 0
            and self.trainer.current_epoch % self.trainer.check_val_every_n_epoch == 0
        ) or (self.trainer.overfit_batches == 0 and batch_idx == 0):
            self._log_all_plots(
                output_0={
                    k: self._to_cpu(v)[0]
                    for k, v in output.items()
                    if isinstance(v, Tensor) and v.dim() > 1
                },
                stage="train",
            )

        return loss

    def _to_cpu(self, tensor: torch.Tensor):
        # if half precision, convert to float
        if tensor.is_floating_point():
            tensor = tensor.float()
        tensor = tensor.detach().cpu()
        return tensor

    def _log_patch_pair_indices_plot(
        self, patch_pair_indices, num_patches: int, stage: str
    ):
        fig, ax = plt.subplots(1, 1)
        plot_patch_pair_coverage(ax, patch_pair_indices, num_patches)
        self._log_plot(fig, f"{stage}/patch_pair_indices")

    def _log_provenance(self, root, transforms, patch_positions, img_shape, stage: str):
        provenance_abs, provenance_rel = create_provenance_grid(
            patch_positions, root, transforms, self.hparams.patch_size, img_shape
        )
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        # plot x and y separately of abs
        ax[0].imshow(provenance_abs[:, :, 0])
        ax[0].set_title("y")
        ax[1].imshow(provenance_abs[:, :, 1])
        ax[1].set_title("x")
        self._log_plot(fig, f"{stage}/provenance_x_y")
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        plot_provenance(ax[0], provenance_rel, abs=True)  # Abs
        plot_provenance(ax[1], provenance_rel, abs=False)  # Rel
        self._log_plot(fig, f"{stage}/provenance_abs_rel")

    def _log_reconstruction_plots(
        self,
        x_original: Tensor,
        x: Tensor,
        pred_T: Tensor,
        patch_positions: Tensor,
        patch_pair_indices: Tensor,
        refpatch_id: int,
        stage: str,
    ):
        ref_transforms = get_transforms_from_reference_patch(
            refpatch_id, pred_T, patch_pair_indices, patch_positions
        )
        self._log_reconstruction(
            refpatch_id,
            ref_transforms,
            x_original,
            x,
            patch_positions,
            stage,
        )
        H, W = x_original.shape[-2:]
        self._log_provenance(
            refpatch_id, ref_transforms, patch_positions, img_shape=(H, W), stage=stage
        )
        self._plot_refpatch_coverage(
            refpatch_id,
            patch_positions,
            H,
            W,
            self.hparams.patch_size,
            ref_transforms,
            stage,
        )

    def _log_plot(
        self,
        fig: plt.Figure,
        name: str,
    ):
        if isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_figure(name, fig, global_step=self.current_epoch)
        elif isinstance(self.logger, WandbLogger):
            wandb.log({name: fig})
        elif self.logger.__class__.__name__ == "AimLogger":
            from aim import Image
            from aim.pytorch_lightning import AimLogger

            self.logger: AimLogger
            img = Image(fig)
            self.logger.experiment.track(
                img,
                name=name,
                step=self.trainer.global_step,
                epoch=self.trainer.current_epoch,
            )
        else:
            self.cli_logger.warning(
                f"{type(self.logger)} is unable to log the {name} image. "
            )
        plt.close(fig)

    def _log_reconstruction(
        self,
        refpatch_id: int,
        transforms,
        img_raw,
        img_input,
        patch_positions,
        stage: str,
    ):
        reconstructed_image = create_image_from_transforms(
            ref_transforms=transforms,
            patch_positions=patch_positions,
            patch_size=self.hparams.patch_size,
            img=img_raw,
            refpatch_id=refpatch_id,
        )

        sampled_image = reconstruct_image_from_sampling(
            patch_positions=patch_positions,
            patch_size=self.hparams.patch_size,
            img=img_raw,
        )

        patch_size = self.hparams.patch_size
        y, x = patch_positions[refpatch_id]
        rect = Rectangle(
            (x, y),
            patch_size,
            patch_size,
            linewidth=1,
            edgecolor="red",
            facecolor="none",
        )

        fig, ax = plt.subplots(1, 4)
        ax[0].imshow(img_raw.permute(1, 2, 0))
        ax[0].set_title("Original Image")
        ax[0].add_patch(copy(rect))
        ax[1].imshow(sampled_image.permute(1, 2, 0))
        ax[1].set_title("Sampled Image")
        ax[1].add_patch(copy(rect))

        ax[2].imshow(reconstructed_image.permute(1, 2, 0))
        ax[2].set_title("Reconstructed Image")
        ax[2].add_patch(copy(rect))

        # ax 3 for input image
        ax[3].imshow(img_input.permute(1, 2, 0))
        ax[3].set_title("Input Image")
        self._log_plot(fig, f"{stage}/reconstruction")

    def _log_transform_distribution(self, pred_T, gt_T, stage: str):
        fig, ax = plt.subplots(2, 2)
        plot_transform_distribution(transforms=pred_T, name="Predicted", axes=ax[0])
        plot_transform_distribution(transforms=gt_T, name="Ground Truth", axes=ax[1])
        self._log_plot(fig, f"{stage}/transform_distribution")
        self.log(
            f"{stage}/transform_distribution/sample_median",
            pred_T.median(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"{stage}/transform_distribution/sample_iqr",
            pred_T.quantile(0.75) - pred_T.quantile(0.25),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def _log_transform_matrix(
        self, num_patches, pred_T, gt_T, patch_pair_indices, stage: str
    ):
        fig, ax = plt.subplots(2, 2)
        plot_patch_pair_transform_matrices(
            patch_pair_indices=patch_pair_indices,
            transforms=pred_T,
            num_patches=num_patches,
            axes=ax[0],
            name="Predicted",
        )
        plot_patch_pair_transform_matrices(
            patch_pair_indices=patch_pair_indices,
            transforms=gt_T,
            num_patches=num_patches,
            axes=ax[1],
            name="Ground Truth",
        )
        self._log_plot(fig, f"{stage}/transform_matrix")

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        output = self.model_step(batch)
        loss, preds, targets = output["loss"], output["pred_T"], output["gt_T"]

        # update and log metrics
        self.val_loss(loss)
        self.val_rmse(preds.flatten(0,1), targets.flatten(0,1))
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(
            dict(zip(["val/rmse_y", "val/rmse_x"], self.val_rmse.compute())),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        if batch_idx == 0:
            self._log_all_plots(
                output_0={
                    k: self._to_cpu(v)[0]
                    for k, v in output.items()
                    if isinstance(v, Tensor) and v.dim() > 1
                },
                stage="val",
            )

    def _get_refpatch_id(
        self, patch_positions: Int[Tensor, "n_patches 2"], img_size: tuple[int, int]
    ):
        # strategy:
        # - get the position closest to the center of the image (ideal)
        # - get the closest patch to that position

        # get the ideal refpatch position
        ideal_refpatch_yx = torch.tensor(img_size) // 2
        # get the closest (in l1 distance) index to the ideal refpatch from the patch_positions
        refpatch_id = torch.argmin(
            (patch_positions - ideal_refpatch_yx).abs().sum(1)
        ).item()
        return refpatch_id

    def _log_all_plots(self, output_0, stage: str = "train"):
        H, W = output_0["x_original"].shape[-2:]
        patch_size = self.hparams.patch_size
        num_patches = (H // patch_size) * (W // patch_size)
        refpatch_id = self._get_refpatch_id(output_0["patch_positions"], (H, W))
        self._log_reconstruction_plots(
            output_0["x_original"],
            output_0["x"],
            output_0["pred_T"],
            output_0["patch_positions"],
            output_0["patch_pair_indices"],
            refpatch_id,
            stage,
        )
        self._log_transform_distribution(output_0["pred_T"], output_0["gt_T"], stage)
        self._log_transform_matrix(
            num_patches,
            output_0["pred_T"],
            output_0["gt_T"],
            output_0["patch_pair_indices"],
            stage,
        )
        self._log_patch_pair_indices_plot(
            output_0["patch_pair_indices"], num_patches, stage
        )

    def _plot_refpatch_coverage(
        self,
        refpatch_id,
        patch_positions,
        H,
        W,
        patch_size,
        ref_transforms: dict,
        stage: str,
    ):
        """
        Image where 1 if patch is connected to root, 0 otherwise.
        """
        n_patches = (H // patch_size) * (W // patch_size)
        occupancy = torch.zeros(H, W)
        for patch_idx in ref_transforms.keys():
            y = patch_positions[patch_idx, 0]
            x = patch_positions[patch_idx, 1]
            occupancy[y : y + patch_size, x : x + patch_size] = 1

        n_coverage = len(set(ref_transforms.keys()))
        fig, ax = plt.subplots(1, 1)
        ax.imshow(occupancy, cmap="binary")
        ax.set_title(f"Coverage of {refpatch_id}: {n_coverage}/{n_patches}")
        # add legend
        legend_elements = [
            plt.Rectangle(
                (0, 0), 1, 1, facecolor="black", edgecolor="black", label="Disconnected"
            ),
            plt.Rectangle(
                (0, 0), 1, 1, facecolor="white", edgecolor="black", label="Connected"
            ),
        ]
        ax.legend(handles=legend_elements)
        self._log_plot(fig, f"{stage}/root_coverage")

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        mse = self.val_rmse.compute()  # get current val mse
        self.val_mse_best(mse)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(
            "val/rmse_best", self.val_mse_best.compute(), sync_dist=True, prog_bar=True
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
        loss, preds, targets = output["loss"], output["pred_T"], output["gt_T"]

        # update and log metrics
        self.test_loss(loss)
        self.test_rmse(preds.flatten(0,1), targets.flatten(0,1))
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log_dict(
            dict(zip(["test/rmse_y", "test/rmse_x"], self.test_rmse.compute())),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        if batch_idx == 0:
            self._log_all_plots(
                output_0={
                    k: self._to_cpu(v)[0]
                    for k, v in output.items()
                    if isinstance(v, Tensor) and v.dim() > 1
                },
                stage="test",
            )

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
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)


if __name__ == "__main__":
    _ = PARTViTModule(None, None, None, None)
