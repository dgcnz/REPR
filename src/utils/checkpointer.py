#!/usr/bin/env python3
# Utility module for checkpoint saving and loading operations

import os
import torch
import wandb
from typing import Any, Dict, Optional, Union
from pathlib import Path
from lightning import Fabric
from lightning.fabric.wrappers import _unwrap_objects

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def save_checkpoint(
    fabric: Fabric,
    model: Any,
    optimizer: Any,
    epoch: int,
    filepath: str,
    verbose: bool = True,
    global_step: int = None,
    scheduler=None,
    **kwargs
) -> None:
    """Save a model checkpoint.
    
    Args:
        fabric: Fabric instance
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch number
        filepath: Path to save checkpoint to
        verbose: Whether to print logging information
        global_step: Current global step number
        scheduler: Optional learning rate scheduler to save
        **kwargs: Additional items to save in the checkpoint
    """
    if not fabric.is_global_zero:
        return
        
    model_unwrapped = _unwrap_objects(model)
    optimizer_unwrapped = _unwrap_objects(optimizer)
    
    checkpoint = {
        "model": model_unwrapped.state_dict(),
        "optimizer": optimizer_unwrapped.state_dict(),
        "epoch": epoch,
    }
    
    # Save global step if provided
    if global_step is not None:
        checkpoint["global_step"] = global_step
    
    # Save scheduler if provided
    if scheduler is not None:
        scheduler_unwrapped = _unwrap_objects(scheduler)
        checkpoint["scheduler"] = scheduler_unwrapped.state_dict()
    
    # Save wandb run ID in checkpoint if available
    if wandb.run is not None:
        checkpoint["wandb_run_id"] = wandb.run.id
    
    # Save any additional items provided
    for key, value in kwargs.items():
        checkpoint[key] = value
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save the checkpoint
    fabric.save(filepath, checkpoint)
    
    if verbose:
        log.info(f"Saved checkpoint: {filepath}")


def load_checkpoint(
    fabric: Fabric,
    model: Any,
    optimizer: Optional[Any] = None,
    checkpoint_path: str = None,
    scheduler: Optional[Any] = None,
    verbose: bool = True,
    strict: bool = True,
) -> Dict[str, Any]:
    """Load a model checkpoint.
    
    Args:
        fabric: Fabric instance
        model: Model to load checkpoint into
        optimizer: Optional optimizer to load checkpoint into
        checkpoint_path: Path to checkpoint file
        scheduler: Optional scheduler to load checkpoint into
        verbose: Whether to print logging information
        strict: Whether to enforce strict state dict loading
    
    Returns:
        Dictionary containing checkpoint metadata and any extra stored values
    """
    if not os.path.exists(checkpoint_path):
        if verbose:
            log.info(f"No checkpoint found at {checkpoint_path}")
        return {"epoch": 0, "global_step": 0}
    
    if verbose:
        log.info(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = fabric.load(checkpoint_path)
    
    # Load model state
    model_unwrapped = _unwrap_objects(model)
    model_unwrapped.load_state_dict(checkpoint["model"], strict=strict)
    
    # Load optimizer state if available and optimizer is provided
    if optimizer is not None and "optimizer" in checkpoint:
        try:
            optimizer_unwrapped = _unwrap_objects(optimizer)
            optimizer_unwrapped.load_state_dict(checkpoint["optimizer"])
            if verbose:
                log.info("Loaded optimizer state")
        except Exception as e:
            log.warning(f"Failed to load optimizer state: {e}")
    
    # Load scheduler state if available and scheduler is provided
    if scheduler is not None and "scheduler" in checkpoint:
        try:
            scheduler_unwrapped = _unwrap_objects(scheduler)
            scheduler_unwrapped.load_state_dict(checkpoint["scheduler"])
            if verbose:
                log.info("Loaded scheduler state")
        except Exception as e:
            log.warning(f"Failed to load scheduler state: {e}")
    
    # Initialize wandb with the same run if we have a run ID
    if "wandb_run_id" in checkpoint and fabric.global_rank == 0:
        current_run_id = wandb.run.id if wandb.run else None
        if current_run_id != checkpoint["wandb_run_id"]:
            log.info(f"Previous run ID: {checkpoint['wandb_run_id']}, current: {current_run_id}")
    
    if verbose:
        log.info(f"Loaded checkpoint from {checkpoint_path}")
        if "epoch" in checkpoint:
            log.info(f"Resuming from epoch {checkpoint['epoch']}")
        if "global_step" in checkpoint:
            log.info(f"Resuming from global step {checkpoint['global_step']}")
    
    # Make sure we have at least empty defaults
    if "epoch" not in checkpoint:
        checkpoint["epoch"] = 0
    if "global_step" not in checkpoint:
        checkpoint["global_step"] = 0
        
    return checkpoint


def find_latest_checkpoint(
    dirpath: str, 
    pattern: str = "checkpoint-*.ckpt"
) -> Optional[str]:
    """Find the latest checkpoint in the directory.
    
    Args:
        dirpath: Directory to search for checkpoints
        pattern: Glob pattern for checkpoint files
    
    Returns:
        Path to the latest checkpoint, or None if no checkpoints found
    """
    checkpoints = sorted(Path(dirpath).glob(pattern))
    if not checkpoints:
        return None
    return str(checkpoints[-1])


def get_checkpoint_path(
    dirpath: str,
    specific_checkpoint: Optional[str] = None,
    use_last: bool = True
) -> Optional[str]:
    """Get the checkpoint path to use for loading.
    
    Args:
        dirpath: Directory to search for checkpoints
        specific_checkpoint: Optional specific checkpoint path to use
        use_last: Whether to use the last.ckpt file
    
    Returns:
        Path to the checkpoint to use, or None if no checkpoint found
    """
    # First try the specific checkpoint if provided
    if specific_checkpoint and os.path.exists(specific_checkpoint):
        return specific_checkpoint
    
    # Then try the last checkpoint if enabled
    if use_last:
        last_ckpt = os.path.join(dirpath, "last.ckpt")
        if os.path.exists(last_ckpt):
            return last_ckpt
    
    # Finally try to find the latest numbered checkpoint
    return find_latest_checkpoint(dirpath)

