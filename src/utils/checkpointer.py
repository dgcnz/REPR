#!/usr/bin/env python3
# Utility module for checkpoint saving and loading operations

import os
import torch
import sys
import wandb
from typing import Any, Dict, Optional, Union, Literal
from pathlib import Path
from lightning import Fabric
from lightning.fabric.wrappers import _unwrap_objects
from torch import nn, optim

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
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    ckpt_path: str = None,
    ckpt_mode: Literal["checkpoint", "backbone"] = "checkpoint",
) -> tuple[int, int]:
    """Load a model checkpoint.
    
    Args:
        fabric: Fabric instance
        model: Model to load checkpoint into
        optimizer: Optional optimizer to load checkpoint into
        ckpt_path: Path to checkpoint file
        scheduler: Optional scheduler to load checkpoint into
        verbose: Whether to print logging information
        strict: Whether to enforce strict state dict loading
    
    Returns:
        Dictionary containing checkpoint metadata and any extra stored values
    """
    if not os.path.exists(ckpt_path):
        log.info(f"No checkpoint found at {ckpt_path}")
        sys.exit(0)
    
    log.info(f"Loading checkpoint from {ckpt_path}")
    
    state = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "global_step": 0,
        "epoch": 0,
    }

    if ckpt_mode == "backbone":
        fabric.load_raw(ckpt_path, model, strict=False)
        return 0, 0 
    
    fabric.load(ckpt_path, state, strict=True)
    return state["epoch"], state["global_step"]