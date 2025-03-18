import os
import hydra
import torch
import rootutils
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from torchmetrics import Metric

import lightning as L
from lightning import Fabric
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import wandb

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import pylogger, extras, checkpointer
from src.utils.instantiators import instantiate_callbacks, instantiate_loggers
from src.engine_pretrain import train_one_epoch

log = pylogger.RankedLogger(__name__)


def ieval(expr):
    return hydra.utils.get_object(expr)


OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("ieval", ieval)


def setup(cfg: DictConfig) -> Tuple[Fabric, Dict[str, Any]]:
    # Set seed for reproducibility
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # Set float32 matmul precision
    if cfg.get("float32_matmul_precision"):
        torch.set_float32_matmul_precision(cfg.float32_matmul_precision)

    # TODO: add option to cfg
    torch.backends.cuda.matmul.allow_tf32 = True

    # cuDNN optimization
    if cfg.get("cudnn_benchmark", False):
        cudnn.benchmark = True

    # Get checkpoint path from config if specified
    ckpt_path = cfg.get("ckpt_path", None)

    # Setup callbacks using instantiate_callbacks
    callbacks = instantiate_callbacks(cfg.get("callbacks"))

    # Set up loggers using instantiate_loggers
    loggers = instantiate_loggers(cfg.get("logger"))

    # Create Fabric instance with appropriate strategy
    fabric = Fabric(
        accelerator=cfg.trainer.get("accelerator", "auto"),
        devices=cfg.trainer.get("devices", "auto"),
        num_nodes=cfg.trainer.get("num_nodes", 1),
        precision=cfg.trainer.get("precision", 32),
        strategy=cfg.trainer.get("strategy", "auto"),
        callbacks=callbacks,
        loggers=loggers,
    )

    for logger in fabric._loggers:
        logger.log_hyperparams(cfg)

    fabric.launch()

    # Initialize datamodule
    log.info(f"Instantiating dataset <{cfg.data._target_}>")
    dataset = hydra.utils.instantiate(cfg.data)
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        **cfg.train_dataloader,
    )

    # Initialize model with correct device and precision
    log.info(f"Instantiating model <{cfg.model._target_}>")

    with fabric.init_module(empty_init=ckpt_path is not None):
        model = hydra.utils.instantiate(cfg.model)

    metric_collection = hydra.utils.instantiate(cfg.metric_collection)

    if cfg.get("compile"):
        torch._dynamo.config.optimize_ddp = False
        # Disable ddp optimization
        model = torch.compile(model)

    # Initialize optimizer
    log.info(f"Instantiating optimizer <{cfg.optimizer._target_}>")
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())()

    # Initialize scheduler
    scheduler = None
    if "scheduler" in cfg:
        log.info(f"Instantiating LR scheduler <{cfg.scheduler._target_}>")
        scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)()

    return fabric, model, optimizer, scheduler, train_dataloader, metric_collection


def train(
    fabric: Fabric,
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    metric_collection: Metric,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    ckpt_path: Optional[str] = None,
    # Trainer arguments
    max_epochs: int = 1000,
    accumulate_grad_batches: int = 1,
    gradient_clip_val: float = 0,
    track_grad_norm: bool = False,
):
    start_epoch, global_step = 0, 0

    # Load checkpoint if provided
    if ckpt_path:
        checkpoint_data = checkpointer.load_checkpoint(
            fabric=fabric,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_path=ckpt_path,
            verbose=True,
        )
        start_epoch = checkpoint_data["epoch"] + 1  # Resume from next epoch
        global_step = checkpoint_data["global_step"]

    # Call on_train_start at the start of training
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

        log.debug(f"train_one_epoch {epoch} ... ")
        # Train for one epoch and get updated global step
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
        log.debug(f"train_one_epoch {epoch} done")

    # Call on_train_end at the end of training to save final checkpoint
    fabric.call(
        "on_train_end",
        fabric=fabric,
        model=model,
        metric_collection=metric_collection,
        epoch=max_epochs - 1,  # Use the last epoch index
        global_step=global_step,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    log.info("Training completed!")


def main(cfg: DictConfig) -> None:
    # Create output directory
    output_dir = Path(cfg.paths.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    fabric, model, optimizer, scheduler, train_dataloader, metric_collection = setup(
        cfg
    )

    model, optimizer = fabric.setup(model, optimizer)
    metric_collection = metric_collection.to(fabric.device)
    train_dataloader = fabric.setup_dataloaders(train_dataloader)

    # Print configuration with fabric
    if fabric.is_global_zero:
        log.info(f"Configuration: {OmegaConf.to_yaml(cfg)}")
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"Model: {model.__class__.__name__}")
        log.info(f"Number of parameters: {n_parameters / 1e6:.2f}M")

    # Get checkpoint path from config
    ckpt_path = cfg.get("ckpt_path")
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
            ckpt_path=ckpt_path,
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


@hydra.main(
    version_base="1.3", config_path="../fabric_configs", config_name="pretrain.yaml"
)
def hydra_main(cfg: DictConfig) -> None:
    """Main entry point for training."""
    # Apply extra utilities
    extras(cfg)

    # Train the model
    main(cfg)


if __name__ == "__main__":
    hydra_main()
