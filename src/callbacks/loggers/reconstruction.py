from src.utils.visualization import (
    plot_reconstructions,
    get_transforms_from_reference_patch_batch,
    plot_provenances_batch,
)
from src.utils.visualization.reconstruction import _get_refpatch_id_batch
from lightning.pytorch.utilities import rank_zero_only
from src.utils.misc import should_log
from src.callbacks.base_callback import BaseCallback


class ReconstructionLogger(BaseCallback):
    def __init__(self, every_n_steps: int = -1, num_samples: int = 1):
        super().__init__()
        self.every_n_steps = every_n_steps
        self.num_samples = num_samples

    def _plot(
        self, pl_module, x_original, x, patch_positions, pred_T, patch_pair_indices
    ):
        img_size: tuple[int, int] = x_original.shape[-2:]
        refpatch_id = _get_refpatch_id_batch(
            patch_positions[: self.num_samples], img_size
        )
        refpatch_id = refpatch_id.tolist()

        ref_transforms = get_transforms_from_reference_patch_batch(
            refpatch_id, pred_T, patch_pair_indices, patch_positions
        )
        fig_rec = plot_reconstructions(
            refpatch_ids=refpatch_id,
            ref_transforms=ref_transforms,
            patch_positions=patch_positions[: self.num_samples],
            img_original=x_original[: self.num_samples],
            img_input=x[: self.num_samples],
            patch_size=pl_module.hparams.patch_size,
        )
        fig_prov = plot_provenances_batch(
            refpatch_ids=refpatch_id,
            ref_transforms=ref_transforms,
            patch_positions=patch_positions[: self.num_samples],
            img_shape=img_size,
            patch_size=pl_module.hparams.patch_size,
        )
        return fig_rec, fig_prov

    @rank_zero_only
    def on_stage_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, stage: str
    ):
        if not should_log(batch_idx, self.every_n_steps):
            return
        c = pl_module.cache
        fig_rec, fig_prov = self._plot(
            pl_module,
            c.x_original,
            c.x,
            c.patch_positions,
            c.pred_T,
            c.patch_pair_indices,
        )
        self.log_plots(
            {
                f"{stage}/reconstruction": fig_rec,
                f"{stage}/provenance": fig_prov,
            }
        )
