from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn.functional as F
from lightning import Fabric
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics import Metric

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def train_one_epoch(
    fabric: Fabric,
    model: torch.nn.Module,
    data_loader: DataLoader,
    metric_collection: Metric,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    accum_iter: int = 1,
    clip_grad: float = 0.0,
):
    """Train model for one epoch. Logging and metrics are handled by callbacks."""
    model.train()

    # Initialize optimizer
    optimizer.zero_grad()

    # Wrap data loader with progress bar

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
        "leave": False,
        "disable": not fabric.is_global_zero, # Only show progress bar on global zero
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

            # Skip optimizer step if we're still accumulating
            if not is_optim_step:
                continue

            # Gradient clipping
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

    # Update epoch-based scheduler
    if scheduler is not None:
        scheduler.step(epoch=epoch)

    # Return the updated global step
    return global_step
