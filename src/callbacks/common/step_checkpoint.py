import os
from typing import Optional
from lightning import Fabric
import torch

from src.utils import pylogger, checkpointer

log = pylogger.RankedLogger(__name__)


class StepModelCheckpoint(object):
    """Callback for saving model checkpoints during training based on steps."""

    def __init__(
        self,
        dirpath: str,
        save_perm_every_n_steps: int,
        save_temp_every_n_steps: Optional[int] = None,
        verbose: bool = True,
    ):
        """Initialize the checkpoint callback.

        Args:
            dirpath: Directory to save checkpoints
            save_perm_every_n_steps: Save a permanent checkpoint every N training steps.
            save_temp_every_n_steps: Save/Update a "last.ckpt" every N training steps.
                                           If None, no temporary checkpoint is saved during steps.
            verbose: Whether to print logging information
        """
        self.dirpath = dirpath
        self.save_perm_every_n_steps = save_perm_every_n_steps
        self.save_temp_every_n_steps = save_temp_every_n_steps
        self.verbose = verbose
        assert save_perm_every_n_steps > 0, (
            "save_perm_every_n_steps must be greater than 0"
        )
        if self.save_temp_every_n_steps is not None:
            assert self.save_temp_every_n_steps > 0, (
                "save_temp_every_n_steps must be greater than 0 if set"
            )

        # Create checkpoint directory if it doesn't exist
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)

    def on_train_batch_end(
        self,
        fabric: Fabric,
        model: torch.nn.Module,
        epoch: int,
        global_step: int,  # 0-indexed, number of optimizer steps completed so far
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        **kwargs,
    ) -> None:
        """Save checkpoint after training batch if configured."""
        current_step_1_indexed = global_step + 1

        # Save permanent step checkpoint
        if current_step_1_indexed % self.save_perm_every_n_steps == 0:
            periodic_filepath = os.path.join(
                self.dirpath, f"step_{current_step_1_indexed:07d}.ckpt"
            )
            checkpointer.save_checkpoint(
                fabric=fabric,
                model=model,
                optimizer=optimizer,
                epoch=epoch,  # Current 0-indexed epoch
                global_step=current_step_1_indexed,
                filepath=periodic_filepath,
                scheduler=scheduler,
                verbose=self.verbose,
            )

        # Save temporary "last_step" checkpoint if configured
        if (
            self.save_temp_every_n_steps is not None
            and current_step_1_indexed % self.save_temp_every_n_steps == 0
        ):
            last_filepath = os.path.join(self.dirpath, "last.ckpt")
            checkpointer.save_checkpoint(
                fabric=fabric,
                model=model,
                optimizer=optimizer,
                epoch=epoch,  # Current 0-indexed epoch
                global_step=current_step_1_indexed,
                filepath=last_filepath,
                scheduler=scheduler,
                verbose=False,  # Usually less verbose for temporary checkpoints
            )

    def on_train_end(
        self,
        fabric: Fabric,
        model: torch.nn.Module,
        epoch: int,  # Last completed epoch (0-indexed)
        global_step: int,  # Total number of optimizer steps completed
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        **kwargs,
    ) -> None:
        """Save the final checkpoint at the end of training."""

        filepath = os.path.join(self.dirpath, "last.ckpt")
        checkpointer.save_checkpoint(
            fabric=fabric,
            model=model,
            optimizer=optimizer,
            epoch=epoch,  # Last completed 0-indexed epoch
            global_step=global_step,
            filepath=filepath,
            scheduler=scheduler,
            verbose=True,
        )
        log.info(
            f"Training completed. Final step-based checkpoint saved to {filepath}"
        )