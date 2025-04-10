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

    # Calculate number of optimization steps
    n_steps = len(data_loader) // accum_iter
    tqdm_kwargs = {
        "total": n_steps,
        "desc": f"Epoch {epoch}",
        "disable": not fabric.is_global_zero,  # Only show progress bar on global zero
    }

    # Convert dataloader to iterator to handle manual batch fetching
    data_iter = iter(data_loader)
    # warmup for compile

    with tqdm(range(n_steps), **tqdm_kwargs) as pbar:
        for step in pbar:
            torch.compiler.cudagraph_mark_step_begin()  
            optimizer.zero_grad()

            # Accumulate gradients over multiple batches
            for i in range(accum_iter):
                batch = next(data_iter)

                outputs = model(*batch)
                loss = outputs["loss"] / accum_iter

                with fabric.no_backward_sync(model, enabled=bool(i < accum_iter - 1)):
                    fabric.backward(loss)

            # Process is now at the end of accumulation or dataset
            outputs = apply_to_collection(outputs, Tensor, lambda x: x.detach())

            # Track gradient norm if requested
            if track_grad_norm:
                outputs["grad_norm"] = _compute_grad_norm(model)

            # Gradient clipping if enabled
            if clip_grad is not None and clip_grad > 0:
                fabric.clip_gradients(model, optimizer, max_norm=clip_grad)

            # Optimizer step
            optimizer.step()

            # Update metrics and call callbacks
            fabric.call(
                "on_train_batch_end",
                fabric=fabric,
                model=model,
                outputs=outputs,
                batch=batch,  # Last batch processed
                batch_idx=step * accum_iter,
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
        torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
        norm_type,
    )
    return grad_norm
