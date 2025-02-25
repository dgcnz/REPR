from src.utils.visualization import (
    plot_reconstructions_v2,
    get_transforms_from_reference_patch_batch,
)
from src.utils.visualization.reconstruction import _get_refpatch_id_batch_v2
from lightning.pytorch.utilities import rank_zero_only
from src.utils.misc import should_log
from src.callbacks.base_callback import BaseCallback


class ReconstructionLogger(BaseCallback):
    def __init__(
        self, every_n_steps: int = -1, num_samples: int = 1, every_n_epochs: int = 1
    ):
        super().__init__()
        self.every_n_steps = every_n_steps
        self.num_samples = num_samples
        self.every_n_epochs = every_n_epochs

    def _plot(
        self,
        pl_module,
        img,
        patch_positions_vis,
        pred_T,
        patch_pair_indices,
        ids_remove_pos,
    ):
        img_size: tuple[int, int] = img.shape[-2:]
        refpatch_id = _get_refpatch_id_batch_v2(
            patch_positions_vis,
            img_size,
            ids_remove_pos,
        )
        refpatch_id = refpatch_id.squeeze(-1).tolist()

        ref_transforms = get_transforms_from_reference_patch_batch(
            refpatch_id, pred_T, patch_pair_indices, patch_positions_vis
        )
        fig_rec = plot_reconstructions_v2(
            refpatch_ids=refpatch_id,
            ref_transforms=ref_transforms,
            patch_positions_vis=patch_positions_vis,
            img=img,
            patch_size=pl_module.net.patch_size,
        )
        return fig_rec

    @rank_zero_only
    def on_stage_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, stage: str
    ):
        if not should_log(
            batch_idx, self.every_n_steps, trainer.current_epoch, self.every_n_epochs
        ):
            return
        fig_rec = self._plot(
            pl_module,
            batch["image"][: self.num_samples].detach().cpu(),
            outputs["patch_positions_vis"][: self.num_samples].detach().cpu(),
            outputs["pred_T"][: self.num_samples].detach().cpu(),
            outputs["patch_pair_indices"][: self.num_samples].detach().cpu(),
            outputs["ids_remove_pos"][: self.num_samples].detach().cpu(),
        )
        self.log_plots(
            {
                f"{stage}/reconstruction": fig_rec,
            }
        )
