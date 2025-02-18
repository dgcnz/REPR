# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import builtins

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

# assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from util import utils
import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_vit

from engine_finetune import train_one_epoch, evaluate

import hydra
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass, field

@dataclass
class Config:
    # Training parameters
    batch_size: int = field(default=64, metadata={"help": "Batch size per GPU (effective: batch_size * accum_iter * # gpus)"})
    epochs: int = field(default=50, metadata={"help": "Number of training epochs"})
    accum_iter: int = field(default=1, metadata={"help": "Gradient accumulation iterations"})
    # Model parameters
    model: str = field(default="vit_large_patch16", metadata={"help": "Name of model to train"})
    input_size: int = field(default=224, metadata={"help": "Image input size"})
    drop_path: float = field(default=0.1, metadata={"help": "Drop path rate"})
    fix_pos_emb: bool = field(default=False, metadata={"help": "Fix positional embeddings when fine-tuning"})
    # Optimizer parameters
    clip_grad: float = field(default=None, metadata={"help": "Clip gradient norm (None for no clipping)"})
    weight_decay: float = field(default=0.05, metadata={"help": "Weight decay"})
    lr: float = field(default=None, metadata={"help": "Learning rate (absolute)"})
    blr: float = field(default=1e-3, metadata={"help": "Base learning rate"})
    layer_decay: float = field(default=0.75, metadata={"help": "Layer-wise lr decay"})
    min_lr: float = field(default=1e-6, metadata={"help": "Lower lr bound"})
    warmup_epochs: int = field(default=5, metadata={"help": "Epochs to warmup LR"})
    # Augmentation parameters
    color_jitter: float = field(default=None, metadata={"help": "Color jitter factor"})
    aa: str = field(default="rand-m9-mstd0.5-inc1", metadata={"help": "AutoAugment policy"})
    smoothing: float = field(default=0.1, metadata={"help": "Label smoothing factor"})
    # Mixup and related params
    mixup: float = field(default=0, metadata={"help": "Mixup alpha"})
    cutmix: float = field(default=0, metadata={"help": "Cutmix alpha"})
    cutmix_minmax: list = field(default_factory=lambda: None, metadata={"help": "cutmix min/max ratio"})
    mixup_prob: float = field(default=1.0, metadata={"help": "Probability of applying mixup/cutmix"})
    mixup_switch_prob: float = field(default=0.5, metadata={"help": "Switch probability between mixup and cutmix"})
    mixup_mode: str = field(default="batch", metadata={"help": "Mixup mode: batch, pair, or elem"})
    # Finetuning parameters
    finetune: str = field(default="", metadata={"help": "Path to finetune checkpoint"})
    global_pool: bool = field(default=True, metadata={"help": "Use global pooling"})
    teacher: bool = field(default=False, metadata={"help": "Load EMA teacher for fine-tuning"})
    # Dataset and logging parameters
    data_path: str = field(default="/datasets01/imagenet_full_size/061417/", metadata={"help": "Dataset path"})
    nb_classes: int = field(default=1000, metadata={"help": "Number of classes"})
    output_dir: str = field(default="./output_dir", metadata={"help": "Directory to save outputs"})
    log_dir: str = field(default="./output_dir", metadata={"help": "Directory for tensorboard logs"})
    device: str = field(default="cuda", metadata={"help": "Device to use"})
    seed: int = field(default=0, metadata={"help": "Random seed"})
    resume: str = field(default="", metadata={"help": "Resume from checkpoint"})
    experiment: str = field(default="exp", metadata={"help": "Experiment name for logging"})
    start_epoch: int = field(default=0, metadata={"help": "Starting epoch"})
    eval: bool = field(default=False, metadata={"help": "Evaluation only flag"})
    dist_eval: bool = field(default=False, metadata={"help": "Distributed evaluation flag"})
    num_workers: int = field(default=10, metadata={"help": "Number of data loader workers"})
    pin_mem: bool = field(default=True, metadata={"help": "Pin memory in DataLoader"})
    # Distributed training parameters
    world_size: int = field(default=1, metadata={"help": "Number of distributed processes"})
    local_rank: int = field(default=-1, metadata={"help": "Local process rank"})
    dist_on_itp: bool = field(default=False, metadata={"help": "Distributed training flag"})
    dist_url: str = field(default="env://", metadata={"help": "URL for distributed training"})

def main(cfg: Config):
    misc.init_distributed_mode(cfg)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print(OmegaConf.to_yaml(OmegaConf.create(cfg.__dict__)))

    device = torch.device(cfg.device)

    # fix the seed for reproducibility
    seed = cfg.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = build_dataset(is_train=True, args=cfg)
    dataset_val = build_dataset(is_train=False, args=cfg)

    if cfg.distributed:  # cfg.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if cfg.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank,
                shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_mem,
        drop_last=False
    )

    if global_rank == 0 and cfg.log_dir is not None and not cfg.eval:
        log_dir = os.path.join(cfg.log_dir, f"{cfg.model}_{cfg.experiment}")
        os.makedirs(log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=log_dir)
    else:
        log_writer = None

    mixup_fn = None
    mixup_active = cfg.mixup > 0 or cfg.cutmix > 0. or cfg.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=cfg.mixup, cutmix_alpha=cfg.cutmix, cutmix_minmax=cfg.cutmix_minmax,
            prob=cfg.mixup_prob, switch_prob=cfg.mixup_switch_prob, mode=cfg.mixup_mode,
            label_smoothing=cfg.smoothing, num_classes=cfg.nb_classes)

    model = models_vit.__dict__[cfg.model](
        num_classes=cfg.nb_classes,
        drop_path_rate=cfg.drop_path,
        global_pool=cfg.global_pool,
        fix_pos_emb=cfg.fix_pos_emb,
    )

    if cfg.finetune:
        # load pretrained model
        print("Load pre-trained checkpoint from: %s" % cfg.finetune)
        checkpoint = torch.load(cfg.finetune, map_location='cpu')

        if cfg.teacher:
            key = 'ema_state_dict'
        else:
            key = 'state_dict' if 'state_dict' in checkpoint else 'model'
        print("Load checkpoint[{}] for fine-tuning".format(key))
        checkpoint_model = checkpoint[key]

        state_dict = model.state_dict()
        checkpoint_model = {k.replace("module.", ""): v for k, v in checkpoint_model.items()}

        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print('missing keys:', msg.missing_keys)
        print('unexpected keys:', msg.unexpected_keys)

        if cfg.global_pool:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = cfg.batch_size * cfg.accum_iter * misc.get_world_size()
    
    if cfg.lr is None:  # only base_lr is specified
        cfg.lr = cfg.blr * eff_batch_size / 256

    print("base lr: %.2e" % (cfg.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % cfg.lr)

    print("accumulate grad iterations: %d" % cfg.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if cfg.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.gpu])
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp, cfg.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=cfg.layer_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr)
    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif cfg.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=cfg.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    # resume model
    ckpt_path = os.path.join(cfg.output_dir, f"{cfg.model}_{cfg.experiment}_temp.pth")
    if not os.path.isfile(ckpt_path):
        print("Checkpoint not founded in {}, train from random initialization".format(ckpt_path))
    else:
        print("Found checkpoint at {}".format(ckpt_path))
        misc.load_model(args=cfg, ckpt_path=ckpt_path, model_without_ddp=model, optimizer=optimizer,
                        loss_scaler=loss_scaler)

    if cfg.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    print(f"Start training for {cfg.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(cfg.start_epoch, cfg.epochs):
        if cfg.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            cfg.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=cfg
        )

        save_dict = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model": cfg.model,
        }

        ckpt_path = os.path.join(cfg.output_dir, f"{cfg.model}_{cfg.experiment}_temp.pth")
        utils.save_on_master(save_dict, ckpt_path)
        print(f"model_path: {ckpt_path}")

        if cfg.output_dir and ((epoch + 1) % 100 == 0 or epoch + 1 == cfg.epochs):
            ckpt_path = os.path.join(cfg.output_dir,
                                     "{}_{}_{:04d}.pth".format(cfg.model, cfg.experiment,
                                                               epoch))
            utils.save_on_master(save_dict, ckpt_path)

        test_stats = evaluate(data_loader_val, model, device)
        print(f"Pretrained from: {cfg.finetune}")
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

        if cfg.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(
                    cfg.output_dir,
                    "{}_{}_log.txt".format(
                        cfg.model,
                        cfg.experiment
                    )
            ), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


@hydra.main(config_name="config", config_path="conf")
def hydra_main(cfg: Config):
    print(OmegaConf.to_yaml(cfg))
    if cfg.output_dir:
        from pathlib import Path
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    main(cfg)

if __name__ == '__main__':
    hydra_main()
