"""Train one epoch of DropPos pretraining using Fabric."""
import math
from typing import Optional
from lightning import Fabric 
import torch
from torch import Tensor, nn
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
    scheduler: Optional[object] = None,
    accum_iter: int = 1,
    clip_grad: float = 0.0,
    track_grad_norm: bool = False,
):
    """Train model for one epoch. Logging and metrics are handled by callbacks."""
    model.train()
    fabric.call(
        "on_train_epoch_start",
        fabric=fabric,
        model=model,
        epoch=epoch,
        global_step=global_step,
        optimizer=optimizer,
    )

    # Calculate number of optimization steps
    n_steps = len(data_loader) // accum_iter
    tqdm_kwargs = {
        "total": n_steps,
        "desc": f"Epoch {epoch}",
        "disable": not fabric.is_global_zero,
    }

    # Convert dataloader to iterator for manual batch fetching
    data_iter = iter(data_loader)

    with tqdm(range(n_steps), **tqdm_kwargs) as pbar:
        for step in pbar:
            torch.compiler.cudagraph_mark_step_begin()
            optimizer.zero_grad()

            # Accumulate gradients over multiple batches 
            for i in range(accum_iter):
                batch, _ = next(data_iter)  # Ignore target as we're doing self-supervised learning

                # Forward pass and loss computation
                outputs = model(*batch) if isinstance(batch, tuple) else model(batch)
                loss = outputs["loss"] / accum_iter

                # Handle gradient accumulation
                with fabric.no_backward_sync(model, enabled=bool(i < accum_iter - 1)):
                    fabric.backward(loss)

            # Track gradient norm if requested
            if track_grad_norm:
                outputs["grad_norm"] = _compute_grad_norm(model)

            # Gradient clipping if enabled
            if clip_grad is not None and clip_grad > 0:
                fabric.clip_gradients(model, optimizer, max_norm=clip_grad)

            # Optimizer step
            optimizer.step()

            # Update scheduler if it exists
            if scheduler is not None:
                scheduler.step_update(global_step + 1)

            # Update metrics and call callbacks
            fabric.call(
                "on_train_batch_end",  
                fabric=fabric,
                model=model,
                outputs=outputs,
                batch=batch,
                batch_idx=step * accum_iter,
                global_step=global_step,
                epoch=epoch,
                metric_collection=metric_collection,
                optimizer=optimizer,
            )

            global_step += 1

    # Handle end of epoch callbacks
    fabric.call(
        "on_train_epoch_end",
        fabric=fabric, 
        model=model,
        epoch=epoch,
        global_step=global_step,
        optimizer=optimizer,
        scheduler=scheduler,
        metric_collection=metric_collection,
    )

    return global_step


@torch.no_grad()
def _compute_grad_norm(model: nn.Module, norm_type: int = 2):
    """Compute gradient norm of model parameters."""
    parameters = [p for p in model.parameters() if p.grad is not None]
    grad_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
        norm_type,
    )
    return grad_norm
