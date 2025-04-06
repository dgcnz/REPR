#!/usr/bin/env python
"""
main_finetune_det.py

This script sets up object detection finetuning using a Fabric–based training loop.
It loads configuration via LazyConfig, instantiates the model, dataloaders, optimizer,
and detection hooks, and then trains (or evaluates) the model using FabricDetectionTrainer.

Key features:
  • Mixed precision is handled by Fabric.
  • Gradient accumulation is supported via the 'accumulate_grad_batches' config.
  • Detectron2's LazyConfig and hooks (e.g. IterationTimer, LRScheduler, EvalHook) are reused.
  • Distributed training (DDP) is configured via command–line args (num_gpus, num_machines, etc.).
  • Fabric/DDP and wandb initialization is performed via a common helper.
"""

import logging
import warnings
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import (
    default_argument_parser,
    default_setup,
    launch,
    default_writers,
    hooks,
)
from detectron2.evaluation import inference_on_dataset, print_csv_format
import wandb
from lightning import Fabric
from src.engine_finetune_det import FabricDetectionTrainer

warnings.simplefilter(action="ignore", category=FutureWarning)
logger = logging.getLogger("detectron2")


def setup_fabric_ddp(cfg, args):
    """
    Common function to initialize Fabric (handling accelerator, devices, etc.) and DDP.
    Uses command–line args for distributed settings.
    Also initializes wandb if this is the global (main) process.
    """
    precision = None
    if cfg.train.amp.enabled:
        precision = cfg.train.amp.get("precision", "bf16-mixed")

    logger.info("Using precision: {}".format(precision))
    fabric = Fabric(
        accelerator=cfg.train.get("accelerator", "auto"),
        devices=args.num_gpus,  # use command–line number of GPUs per machine
        num_nodes=args.num_machines,  # use command–line total number of machines
        precision=precision,
        strategy=cfg.train.get("strategy", "auto"),
    )
    fabric.launch()
    return fabric


def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        evaluator = instantiate(cfg.dataloader.evaluator)
        test_loader = instantiate(cfg.dataloader.test)
        ret = inference_on_dataset(model, test_loader, evaluator)
        print_csv_format(ret)
        return ret


def do_train(args, cfg):
    # Instantiate model and move it to the proper device.
    model = instantiate(cfg.model)
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    # Instantiate optimizer.
    cfg.optimizer.params.model = model
    optimizer = instantiate(cfg.optimizer)

    # Instantiate the training dataloader.
    train_loader = instantiate(cfg.dataloader.train)

    # Set up Fabric/DDP using the common helper (using args for distributed settings).
    fabric = setup_fabric_ddp(cfg, args)

    if fabric.is_global_zero:
        wandb.init(project="PART-detection", sync_tensorboard=True)

    # Setup model, optimizer, and dataloader with Fabric.
    model, optimizer = fabric.setup(model, optimizer)
    if args.num_gpus > 1 and cfg.train.ddp.fp16_compression:
        # register comm hooks for DDP
        from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks

        logger.info("Registering DDP comm hooks for fp16 compression")
        model._forward_module.register_comm_hook(
            state=None, hook=comm_hooks.fp16_compress_hook
        )

    # Create the detection trainer using our FabricDetectionTrainer.
    accum_iter = cfg.train.get("accumulate_grad_batches", 1)
    clip_grad = cfg.train.get("gradient_clip_val", 0.0)
    track_grad_norm = cfg.train.get("track_grad_norm", False)
    trainer = FabricDetectionTrainer(
        fabric, model, train_loader, optimizer, accum_iter, clip_grad, track_grad_norm
    )

    # Register detectron2 hooks.
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            (
                hooks.PeriodicCheckpointer(
                    DetectionCheckpointer(model, cfg.train.output_dir, trainer=trainer),
                    **cfg.train.checkpointer,
                )
                if fabric.is_global_zero
                else None
            ),
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            (
                hooks.PeriodicWriter(
                    default_writers(cfg.train.output_dir, cfg.train.max_iter),
                    period=cfg.train.log_period,
                )
                if fabric.is_global_zero
                else None
            ),
        ]
    )

    # Load from checkpoint if available.
    checkpointer = DetectionCheckpointer(model, cfg.train.output_dir, trainer=trainer)
    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        start_iter = trainer.iter + 1
    else:
        start_iter = 0

    # Start the training loop.
    trainer.train(start_iter, cfg.train.max_iter)


def do_eval(cfg, args):
    # Set up Fabric/DDP using the common helper.
    fabric = setup_fabric_ddp(cfg, args)
    if fabric.is_global_zero:
        wandb.init(project="PART-detection", sync_tensorboard=True)
    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    # Wrap model with Fabric/DDP.
    model = fabric.setup(model)
    DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
    print(do_test(cfg, model))


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    if args.eval_only:
        do_eval(cfg, args)
    else:
        do_train(args, cfg)


if __name__ == "__main__":
    try:
        args = default_argument_parser().parse_args()
        main(args)
    except Exception as e:
        raise e
    finally:
        if wandb.run:
            wandb.finish()
