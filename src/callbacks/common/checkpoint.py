import os
from typing import Optional
from lightning import Fabric
import torch

from src.utils import pylogger, checkpointer

log = pylogger.RankedLogger(__name__)


class ModelCheckpoint(object):
    """Callback for saving model checkpoints during training."""

    def __init__(
        self,
        dirpath: str,
        every_n_epochs: int = 10,
        save_last: bool = False,
        verbose: bool = True,
    ):
        """Initialize the checkpoint callback.

        Args:
            dirpath: Directory to save checkpoints
            every_n_epochs: Save a permanent checkpoint every N epochs
            save_last: Save an ephemeral "last" checkpoint
            verbose: Whether to print logging information
        """
        self.dirpath = dirpath
        self.every_n_epochs = every_n_epochs
        self.save_last = save_last
        self.verbose = verbose
        assert every_n_epochs > 0, "every_n_epochs must be greater than 0"

        # Create checkpoint directory if it doesn't exist
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)

    def on_train_epoch_end(
        self,
        fabric: Fabric,
        model: torch.nn.Module,
        epoch: int,
        global_step: int,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        **kwargs,
    ) -> None:
        """Save checkpoint after training epoch if configured."""
        # Save permanent epoch checkpoint
        if (epoch + 1) % self.every_n_epochs == 0:
            periodic_filepath = os.path.join(self.dirpath, f"epoch_{epoch:04d}.ckpt")
            checkpointer.save_checkpoint(
                fabric=fabric,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                filepath=periodic_filepath,
                scheduler=scheduler,
                verbose=self.verbose,
            )

        # Save ephemeral "last" checkpoint if configured
        if self.save_last:
            last_filepath = os.path.join(self.dirpath, "last.ckpt")
            log.info(f"Saving last checkpoint to {last_filepath}")
            checkpointer.save_checkpoint(
                fabric=fabric,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                filepath=last_filepath,
                scheduler=scheduler,
                verbose=False,  # Use less verbose output for ephemeral checkpoint
            )
            log.info(f"Saved checkpoint: {last_filepath}")

    def on_train_end(
        self,
        fabric: Fabric,
        model: torch.nn.Module,
        epoch: int,
        global_step: int,
        optimizer,
        scheduler,
        **kwargs,
    ) -> None:
        """Save the final checkpoint at the end of training."""
        final_filepath = os.path.join(self.dirpath, "last.ckpt")
        checkpointer.save_checkpoint(
            fabric=fabric,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            filepath=final_filepath,
            scheduler=scheduler,
            verbose=True,  # Always be verbose for final checkpoint
        )

        log.info(f"Training completed. Final checkpoint saved to {final_filepath}")
