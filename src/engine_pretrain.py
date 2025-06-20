from typing import Optional

import timm.scheduler
import timm.scheduler.scheduler
from torch import Tensor, nn
import torch
from lightning import Fabric
import timm
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
    scheduler: Optional[timm.scheduler.scheduler.Scheduler] = None,
    accum_iter: int = 1,
    clip_grad: float = 0.0,
    track_grad_norm: bool = False,
):
    """Train model for one epoch. Logging and metrics are handled by callbacks."""
    model.train()

    def ctx(step):
        return {
            "fabric": fabric,
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "epoch": epoch,
            "metric_collection": metric_collection,
            "global_step": step,
        }

    log.debug(f"on_train_epoch_start {epoch}")
    fabric.call("on_train_epoch_start", **ctx(global_step))

    # Calculate number of optimization steps
    n_steps = len(data_loader) // accum_iter
    tqdm_kwargs = {
        "total": n_steps,
        "desc": f"Epoch {epoch}",
        "disable": not fabric.is_global_zero,  # Only show progress bar on global zero
    }

    # Convert dataloader to iterator to handle manual batch fetching
    data_iter = iter(data_loader)

    nan_counter = 0
    with tqdm(range(n_steps), **tqdm_kwargs) as pbar:
        for step in pbar:
            fabric.call("on_train_batch_start", **ctx(global_step))
            torch.compiler.cudagraph_mark_step_begin()

            optimizer.zero_grad()
            outputs, batch = train_one_step(
                fabric,
                model,
                accum_iter,
                data_iter,
                global_step,
                ctx,
            )
            if outputs is None:
                log.warning(
                    f"Loss is NaN at step {step} of epoch {epoch}. Skipping step."
                )
                nan_counter += 1
                if nan_counter >= 5:
                    raise RuntimeError("Too many NaN losses. Stopping training.")
                continue
            nan_counter = 0

            # Process is now at the end of accumulation or dataset
            outputs = apply_to_collection(outputs, Tensor, lambda x: x.detach())

            # Track gradient norm if requested
            if track_grad_norm:
                outputs["grad_norm"] = _compute_grad_norm(model)

            # Gradient clipping if enabled
            if clip_grad is not None and clip_grad > 0:
                fabric.clip_gradients(
                    model, optimizer, max_norm=clip_grad, error_if_nonfinite=False
                )

            fabric.call("on_train_optimizer_step_start", **ctx(global_step))
            optimizer.step()
            fabric.call("on_train_optimizer_step_end", **ctx(global_step))

            log.debug(f"Scheduler step {epoch}.")
            if scheduler is not None:
                scheduler.step_update(global_step + 1)

            # Update metrics and call callbacks
            fabric.call(
                "on_train_batch_end",
                outputs=outputs,
                batch=batch,  # Last batch processed
                batch_idx=step * accum_iter,
                **ctx(global_step),
            )

            global_step += 1

    log.debug(f"Epoch {epoch} ended.")
    fabric.call("on_train_epoch_end", **ctx(global_step))
    return global_step


def train_one_step(
    fabric: Fabric,
    model: nn.Module,
    accum_iter: int,
    data_iter: iter,
    global_step: int,
    ctx: callable,
):
    for i in range(accum_iter):
        batch = next(data_iter)

        fabric.call("on_train_forward_start", **ctx(global_step))
        outputs = model(*batch)
        fabric.call("on_train_forward_end", **ctx(global_step))

        loss = outputs["loss"] / accum_iter
        if not torch.isfinite(loss.detach()):
            raise ValueError(
                f"Loss is not finite at step {global_step}: {loss.detach().item()}"
            )

        with fabric.no_backward_sync(model, enabled=bool(i < accum_iter - 1)):
            fabric.call("on_train_backward_start", **ctx(global_step))
            fabric.backward(loss)
            fabric.call("on_train_backward_end", **ctx(global_step))

    return outputs, batch


@torch.no_grad()
def _compute_grad_norm(model: nn.Module, norm_type: int = 2):
    """Compute the gradient norm of the model parameters."""
    parameters = [p for p in model.parameters() if p.grad is not None]
    grad_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
        norm_type,
    )
    return grad_norm
