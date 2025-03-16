#!/usr/bin/env python3
# Fabric callback for logging reconstructions visualizations

from typing import Any, Dict
import torch
import wandb
from lightning import Fabric

from src.utils.visualization import (
    plot_reconstructions_v2,
    get_transforms_from_reference_patch_batch,
)
from src.utils.visualization.reconstruction import _get_refpatch_id_batch_v2
from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


class ReconstructionLoggerCallback:
    """Callback for logging reconstruction visualizations."""

    def __init__(self, log_every_n_steps: int = 50, num_samples: int = 2):
        """Initialize the reconstruction logger callback.

        Args:
            log_every_n_steps: Number of steps between logging visualizations
            num_samples: Number of samples to visualize in each batch
        """
        self.log_every_n_steps = log_every_n_steps
        self.num_samples = num_samples

    def get_reconstruction_visualization(
        self,
        fabric: Fabric,
        batch: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
        model: Any,
    ):
        """Create reconstruction visualization figure.

        Args:
            fabric: Fabric instance
            batch: Training batch dictionary containing:
                - image: Input images
            outputs: Model outputs dictionary containing:
                - patch_positions_vis: Visible patch positions
                - pred_T: Predicted transforms
                - patch_pair_indices: Patch pair indices
                - ids_remove_pos: IDs of removed positions
            model: The model being trained (used to get patch size)

        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        try:
            # Take first N samples and move to CPU
            img = batch["image"][: self.num_samples].detach().cpu()
            patch_positions_vis = (
                outputs["patch_positions_vis"][: self.num_samples].detach().cpu()
            )
            pred_T = outputs["pred_T"][: self.num_samples].detach().cpu()
            patch_pair_indices = (
                outputs["patch_pair_indices"][: self.num_samples].detach().cpu()
            )
            ids_remove_pos = (
                outputs["ids_remove_pos"][: self.num_samples].detach().cpu()
            )

            # Get image size and reference patch IDs
            img_size = img.shape[-2:]
            refpatch_id = _get_refpatch_id_batch_v2(
                patch_positions_vis,
                img_size,
                ids_remove_pos,
            )
            refpatch_id = refpatch_id.squeeze(-1).tolist()

            # Get reference transforms
            ref_transforms = get_transforms_from_reference_patch_batch(
                refpatch_id, pred_T, patch_pair_indices, patch_positions_vis
            )

            # Create visualization
            fig_rec = plot_reconstructions_v2(
                refpatch_ids=refpatch_id,
                ref_transforms=ref_transforms,
                patch_positions_vis=patch_positions_vis,
                img=img,
                patch_size=model.patch_size,
            )

            return fig_rec

        except Exception as e:
            log.error(f"Error in visualization: {e}")
            return None

    def on_train_batch_end(
        self,
        fabric: Fabric,
        model: torch.nn.Module,
        outputs: Dict,
        batch: Any,
        batch_idx: int,
        global_step: int,
        **kwargs,
    ):
        """Log visualizations periodically during training."""
        if fabric.global_rank > 0 or wandb.run is None:
            return

        if batch_idx % self.log_every_n_steps == 0:
            fig = self.get_reconstruction_visualization(fabric, batch, outputs, model)
            if fig is not None:
                wandb.log({"reconstruction": wandb.Image(fig)}, step=global_step)
