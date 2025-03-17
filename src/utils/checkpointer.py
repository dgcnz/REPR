#!/usr/bin/env python3
# Utility module for checkpoint saving and loading operations

import os
import torch
import sys
import wandb
from typing import Any, Dict, Optional, Union
from pathlib import Path
from lightning import Fabric
from lightning.fabric.wrappers import _unwrap_objects

from src.utils import pylogger

log = pylogger.RankedLogger(__name__)


def save_checkpoint(
    fabric: Fabric,
    model: Any,
    optimizer: Any,
    epoch: int,
    filepath: str,
    global_step: int,
    verbose: bool = True,
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
    checkpoint = {
        "model": model,
        "optimizer": optimizer,
        "epoch": epoch,
        "global_step": global_step,
    }
    
    # Save scheduler if provided
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler
    
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
        log.info(f"No checkpoint found at {checkpoint_path}")
        sys.exit(0)
    
    log.info(f"Loading checkpoint from {checkpoint_path}")
    
    state = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }
    
    remainder = fabric.load(checkpoint_path, state, strict=strict)
    return remainder


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

