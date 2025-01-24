from src.callbacks.base_callback import BaseCallback
from lightning.pytorch.utilities import rank_zero_only
from src.utils.misc import should_log
from src.utils.visualization import (
    plot_transform_distribution,
    plot_patch_pair_transform_matrices,
)
import matplotlib.pyplot as plt


class TransformLogger(BaseCallback):
    def __init__(self, every_n_steps: int = -1, num_samples: int = 1):
        super().__init__()
        self.every_n_steps = every_n_steps
        self.num_samples = num_samples
        if self.num_samples != 1:
            raise NotImplementedError("num_samples > 1 is not supported yet.")

    @rank_zero_only
    def on_stage_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, stage: str
    ):
        if not should_log(batch_idx, self.every_n_steps):
            return

        c = pl_module.cache
        _, n_patches, _ = pl_module.patch_info(c.x_original)

        self._log_transform_distribution(
            pred_T=c.pred_T[: self.num_samples],
            gt_T=c.gt_T[: self.num_samples],
            stage=stage,
        )
        self._log_transform_matrix(
            num_patches=n_patches,
            pred_T=c.pred_T[: self.num_samples],
            gt_T=c.gt_T[: self.num_samples],
            patch_pair_indices=c.patch_pair_indices[: self.num_samples],
            stage=stage,
        )

    def _log_transform_distribution(self, pred_T, gt_T, stage: str):
        fig, ax = plt.subplots(2, 2)
        plot_transform_distribution(transforms=pred_T[0], name="Predicted", axes=ax[0])
        plot_transform_distribution(transforms=gt_T[0], name="Ground Truth", axes=ax[1])
        self.log_plots({f"{stage}/transform_distribution": fig})

    def _log_transform_matrix(
        self, num_patches, pred_T, gt_T, patch_pair_indices, stage: str
    ):
        fig, ax = plt.subplots(2, 2)
        plot_patch_pair_transform_matrices(
            patch_pair_indices=patch_pair_indices[0],
            transforms=pred_T[0],
            num_patches=num_patches,
            axes=ax[0],
            name="Predicted",
        )
        plot_patch_pair_transform_matrices(
            patch_pair_indices=patch_pair_indices[0],
            transforms=gt_T[0],
            num_patches=num_patches,
            axes=ax[1],
            name="Ground Truth",
        )
        self.log_plots({f"{stage}/transform_matrix": fig})
