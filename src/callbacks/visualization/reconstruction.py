from src.utils.visualization import (
    get_center_refpatch_id,
    plot_reconstructions,
    get_transforms_from_reference_patch_batch,
    plot_provenances_batch,
)
from src.utils.visualization.reconstruction import _get_refpatch_id_batch
import lightning as L
from lightning.pytorch.utilities import rank_zero_only
from src.utils.misc import should_log
from src.callbacks.base_callback import BaseCallback


class ReconstructionLogger(BaseCallback):
    def __init__(self, every_n_steps: int = -1, num_samples: int = 1):
        super().__init__()
        self.every_n_steps = every_n_steps
        self.num_samples = num_samples

    @rank_zero_only
    def on_stage_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, stage: str
    ):
        if not should_log(batch_idx, self.every_n_steps):
            return
        c = pl_module.cache
        pl_module.cli_logger.info(f"Logging reconstruction plots for {stage}")
        img_size: tuple[int, int] = c.x_original.shape[-2:]
        # refpatch_id: list[int] = get_center_refpatch_id(c.patch_positions, img_size)
        # refpatch_id = get_center_refpatch_id(c.patch_positions, img_size)
        refpatch_id = _get_refpatch_id_batch(c.patch_positions, img_size)
        ref_transforms = get_transforms_from_reference_patch_batch(
            refpatch_id, c.pred_T, c.patch_pair_indices, c.patch_positions
        )
        fig_rec = plot_reconstructions(
            refpatch_ids=refpatch_id,
            ref_transforms=ref_transforms,
            patch_positions=c.patch_positions[: self.num_samples],
            img_original=c.x_original[: self.num_samples],
            img_input=c.x[: self.num_samples],
            patch_size=pl_module.hparams.patch_size,
        )
        fig_prov = plot_provenances_batch(
            refpatch_ids=refpatch_id,
            ref_transforms=ref_transforms,
            patch_positions=c.patch_positions[: self.num_samples],
            img_shape=img_size,
            patch_size=pl_module.hparams.patch_size,
        )
        self.log_plots(
            {
                f"{stage}/reconstruction": fig_rec,
                f"{stage}/provenance": fig_prov,
            }
        )
