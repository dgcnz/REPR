from lightning.pytorch.utilities import rank_zero_only
from src.utils.misc import should_log, clean_tensor
from src.callbacks.base_callback import BaseCallback


class OutputCache(BaseCallback):
    def __init__(
        self,
        num_samples: int = 1,
        ignore: list[str] = ["loss", "symmetry_loss"],
        keys: list[str] = [
            "pred_T",
            "gt_T",
            "x",
            "x_original",
            "patch_positions",
            "patch_pair_indices",
            "ids_nopos",
        ],
        every_n_steps: int = -1,
        every_n_epochs: int = 1,
        invalidate_after_n_steps: int = 1,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.ignore = ignore
        self.every_n_steps = every_n_steps
        self.every_n_epochs = every_n_epochs
        self.invalidate_after_n_steps = invalidate_after_n_steps
        self.invalidate_at_idx = -1
        self.keys = keys

    @rank_zero_only
    def on_stage_batch_start(self, trainer, pl_module, batch, batch_idx, stage: str):
        if self.invalidate_at_idx == batch_idx:
            # log "invalidating cache

            pl_module.cli_logger.info("Invalidating cache")
            for k in self.keys:
                pl_module.cache.__dict__.pop(k, None)

    @rank_zero_only
    def on_stage_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, stage: str
    ):
        if not should_log(batch_idx, self.every_n_steps, trainer.current_epoch, self.every_n_epochs):
            return
        pl_module.cache.__dict__.update(
            {
                k: clean_tensor(v[: self.num_samples])
                for k, v in outputs.items()
                if k not in self.ignore
            }
        )
        self.invalidate_at_idx = batch_idx + self.invalidate_after_n_steps
        pl_module.cli_logger.info(f"Updated cache keys: {list(outputs.keys())} at batch_idx:{batch_idx}")
