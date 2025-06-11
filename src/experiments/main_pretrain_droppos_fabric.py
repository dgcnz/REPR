"""Main entry point for DropPos pretraining using Fabric."""
import os
import hydra
import torch
import rootutils
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import lightning as L
from lightning import Fabric
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import wandb

from src.utils import pylogger, extras, checkpointer
from src.utils.instantiators import instantiate_callbacks, instantiate_loggers
from src.engine_pretrain_droppos_fabric import train_one_epoch

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
log = pylogger.RankedLogger(__name__)

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("ieval", lambda x: hydra.utils.get_object(x))

def setup(cfg: DictConfig) -> Tuple[Fabric, Dict[str, Any]]:
    """Set up training components."""
    # Set seed for reproducibility
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # Configure float32 matmul precision
    if cfg.get("float32_matmul_precision"):
        torch.set_float32_matmul_precision(cfg.float32_matmul_precision)
    torch.backends.cuda.matmul.allow_tf32 = True

    # cuDNN optimization
    if cfg.get("cudnn_benchmark", False):
        cudnn.benchmark = True

    # Get checkpoint path from config
    ckpt_path = cfg.get("ckpt_path", None)

    # Setup callbacks and loggers
    callbacks = instantiate_callbacks(cfg.get("callbacks"))
    loggers = instantiate_loggers(cfg.get("logger"))

    # Create Fabric instance
    fabric = Fabric(
        accelerator=cfg.trainer.get("accelerator", "auto"),
        devices=cfg.trainer.get("devices", "auto"),
        num_nodes=cfg.trainer.get("num_nodes", 1),
        precision=cfg.trainer.get("precision", 32),
        strategy=cfg.trainer.get("strategy", "auto"),
        callbacks=callbacks,
        loggers=loggers,
    )

    # Log hyperparameters
    for logger in fabric._loggers:
        logger.log_hyperparams(cfg)

    fabric.launch()

    # Initialize dataset with mask transform
    log.info(f"Instantiating dataset <{cfg.data._target_}>")
    dataset = hydra.utils.instantiate(cfg.data)
    train_dataloader = DataLoader(dataset, **cfg.train_dataloader)

    # Initialize model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    with fabric.init_module(empty_init=ckpt_path is not None):
        model = hydra.utils.instantiate(cfg.model)

    # Initialize metrics
    metric_collection = hydra.utils.instantiate(cfg.metric_collection)

    # Model compilation if enabled
    if cfg.get("compile", False):
        for key, val in cfg.get("compile_expr", {}).items():
            module_path, _, attr_name = key.rpartition('.')
            log.info(f"Setting {key} to {val}")
            setattr(__import__(module_path, fromlist=[attr_name]), attr_name, val)
            
        compile_kwargs = {"fullgraph": True}
        compile_kwargs.update(cfg.get("compile_kwargs", {}))
        if compile_fn := cfg.get("compile_fn", ""):
            original_fn = getattr(model, compile_fn)
            compiled_fn = torch.compile(original_fn, **compile_kwargs)
            setattr(model, compile_fn, compiled_fn)
        else:
            model = torch.compile(model, **compile_kwargs)

    # Initialize optimizer and scheduler
    log.info(f"Instantiating optimizer <{cfg.optimizer._target_}>")
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())()

    scheduler = None
    if "scheduler" in cfg:
        log.info(f"Instantiating LR scheduler <{cfg.scheduler._target_}>")
        scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)()

    return fabric, model, optimizer, scheduler, train_dataloader, metric_collection

def train(
    fabric: Fabric,
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    metric_collection: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    ckpt_path: Optional[str] = None,
    max_epochs: int = 1000,
    accumulate_grad_batches: int = 1,
    gradient_clip_val: float = 0,
    track_grad_norm: bool = False,
):
    """Main training loop."""
    start_epoch, global_step = 0, 0

    # Load checkpoint if provided
    if ckpt_path:
        checkpoint_data = checkpointer.load_checkpoint(
            fabric=fabric,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ckpt_path=ckpt_path,
            verbose=True,
        )
        start_epoch = checkpoint_data["epoch"] + 1
        global_step = checkpoint_data["global_step"]

    # Call on_train_start
    fabric.call(
        "on_train_start",
        fabric=fabric,
        model=model,
        metric_collection=metric_collection,
        epoch=start_epoch,
        global_step=global_step,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    log.info(f"Starting training for {max_epochs} epochs from epoch {start_epoch}")

    for epoch in range(start_epoch, max_epochs):
        if hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)

        # Train one epoch
        global_step = train_one_epoch(
            fabric=fabric,
            model=model,
            data_loader=train_dataloader,
            optimizer=optimizer,
            metric_collection=metric_collection,
            epoch=epoch,
            global_step=global_step,
            scheduler=scheduler,
            accum_iter=accumulate_grad_batches,
            clip_grad=gradient_clip_val,
            track_grad_norm=track_grad_norm,
        )

    # Call on_train_end
    fabric.call(
        "on_train_end",
        fabric=fabric,
        model=model,
        metric_collection=metric_collection,
        epoch=max_epochs - 1,
        global_step=global_step,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    log.info("Training completed!")

def main(cfg: DictConfig) -> None:
    """Main entry point for training."""
    # Create output directory
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup training components
    fabric, model, optimizer, scheduler, train_dataloader, metric_collection = setup(cfg)

    # Setup with fabric
    model, optimizer = fabric.setup(model, optimizer)
    metric_collection = metric_collection.to(fabric.device)
    train_dataloader = fabric.setup_dataloaders(train_dataloader)

    # Print configuration if on global zero
    if fabric.is_global_zero:
        log.info(f"Configuration: {OmegaConf.to_yaml(cfg)}")
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"Model: {model.__class__.__name__}")
        log.info(f"Number of parameters: {n_parameters / 1e6:.2f}M")

    try:
        train(
            fabric=fabric,
            model=model,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            metric_collection=metric_collection,
            max_epochs=cfg.trainer.get("max_epochs", 1000),
            accumulate_grad_batches=cfg.trainer.get("accumulate_grad_batches", 1),
            scheduler=scheduler,
            ckpt_path=cfg.get("ckpt_path"),
            gradient_clip_val=cfg.trainer.get("gradient_clip_val", 0),
            track_grad_norm=cfg.trainer.get("track_grad_norm", False),
        )
    except Exception as e:
        log.error(f"Training failed with error: {e}")
        raise e
    finally:
        if wandb.run:
            log.info("Closing wandb!")
            wandb.finish()

@hydra.main(version_base="1.3", config_path="../fabric_configs", config_name="droppos_pretrain")
def hydra_main(cfg: DictConfig) -> None:
    """Hydra entry point for training."""
    extras(cfg)
    main(cfg)

if __name__ == "__main__":
    hydra_main()
