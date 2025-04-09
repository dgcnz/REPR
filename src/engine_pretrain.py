from typing import Optional

from torch import Tensor, nn
import torch
from lightning import Fabric
from lightning.fabric.utilities.apply_func import apply_to_collection
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics import Metric

from src.utils import pylogger

log = pylogger.RankedLogger(__name__)


def train_one_epoch(
    fabric: Fabric,
    model: nn.Module,
    data_loader: DataLoader,
    metric_collection: Metric,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    accum_iter: int = 1,
    clip_grad: float = 0.0,
    track_grad_norm: bool = False,
):
    """Train model for one epoch. Logging and metrics are handled by callbacks."""
    model.train()

    # Initialize optimizer
    optimizer.zero_grad()

    # Wrap data loader with progress bar
    log.debug(f"on_train_epoch_start {epoch}")
    fabric.call(
        "on_train_epoch_start",
        fabric=fabric,
        model=model,
        epoch=epoch,
        global_step=global_step,
        optimizer=optimizer,
    )
    tqdm_kwargs = {
        "total": len(data_loader),
        "desc": f"Epoch {epoch}",
        "disable": not fabric.is_global_zero,  # Only show progress bar on global zero
    }
    with tqdm(data_loader, **tqdm_kwargs) as pbar:
        for batch_idx, batch in enumerate(pbar):
            # Forward pass
            outputs = model(*batch)
            loss = outputs["loss"]

            # Scale loss for gradient accumulation
            loss = loss / accum_iter

            # Check if this batch completes an accumulation cycle and requires optimizer step
            is_optim_step = (batch_idx + 1) % accum_iter == 0

            # Backward pass with no_backward_sync when not taking optimizer step
            with fabric.no_backward_sync(model, enabled=not is_optim_step):
                fabric.backward(loss)

            outputs = apply_to_collection(outputs, Tensor, lambda x: x.detach())

            # Skip optimizer step if we're still accumulating
            if not is_optim_step:
                continue

            # Gradient clipping
            if track_grad_norm:
                outputs["grad_norm"] = _compute_grad_norm(model)

            if clip_grad is not None and clip_grad > 0:
                fabric.clip_gradients(model, optimizer, max_norm=clip_grad)


            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()

            # Only increase global step when optimizer step is taken
            fabric.call(
                "on_train_batch_end",
                fabric=fabric,
                model=model,
                outputs=outputs,
                batch=batch,
                batch_idx=batch_idx,
                global_step=global_step,
                epoch=epoch,
                metric_collection=metric_collection,
            )

            global_step += 1

    log.debug(f"Epoch {epoch} ended.")
    fabric.call(
        "on_train_epoch_end",
        fabric=fabric,
        model=model,
        epoch=epoch,  # Using current epoch instead of epoch+1
        global_step=global_step,
        optimizer=optimizer,
        scheduler=scheduler,
        metric_collection=metric_collection,
    )

    log.debug(f"Scheduler step {epoch}.")
    if scheduler is not None:
        scheduler.step(epoch=epoch)

    log.debug(f"Finished train_one_epoch {epoch}.")
    return global_step


@torch.no_grad()
def _compute_grad_norm(model: nn.Module, norm_type: int = 2):
    """Compute the gradient norm of the model parameters."""
    parameters = [p for p in model.parameters() if p.grad is not None]
    grad_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]), norm_type
    )
    return grad_norm