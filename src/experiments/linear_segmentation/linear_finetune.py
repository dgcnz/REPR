import os
import random
import re
from pathlib import Path
from typing import Tuple

import hydra
import torch
import torch.nn as nn
import torchvision.transforms as T
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms.functional import InterpolationMode
from lightning.pytorch.loggers import WandbLogger
import wandb
import pytorch_lightning as pl

from src.data.VOCdevkit.vocdata import VOCDataModule
from src.data.coco.coco_data_module import CocoDataModule
from src.experiments.utils.neco_utils import PredsmIoU
from src.experiments.utils.linear_finetuning_transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    ToTensor,
    SepTransforms,
)
from src.utils.timm_utils import timm_patch_size, timm_embed_dim
from src.utils.io import find_run_id

# from src.models.vit_clip_exp import vit_base as vit_clip_base
from src.data.cityscapes.cityscapes_data import CityscapesDataModule
from src.data.ade20k.ade20kdata import Ade20kDataModule


def validate_checkpoint(ckpt_path: Path) -> tuple[int, str, dict]:
    """Return the step, run ID and config from a checkpoint.

    :param ckpt_path: Path to the Lightning checkpoint file.
    :returns: Tuple ``(global_step, run_id, config)`` extracted from the file.
    """

    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint {ckpt_path} not found")

    state = torch.load(ckpt_path, map_location="cpu")
    ckpt_step = int(state["global_step"])

    m = re.search(r"step_(\d+)\.ckpt", ckpt_path.name)
    if m and int(m.group(1)) != ckpt_step:
        raise ValueError(
            f"step mismatch: filename step {m.group(1)} != global_step {ckpt_step}"
        )

    output_dir = ckpt_path.parent
    run_id = find_run_id(output_dir / "wandb")
    config_path = output_dir / ".hydra" / "config.yaml"
    resume_cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False)

    return ckpt_step, run_id, resume_cfg


@hydra.main(version_base="1.3", config_path="../../../fabric_configs/experiment/linear_segmentation", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.get("fp32", None) == 'high':
        torch.set_float32_matmul_precision("high")

    if cfg.restart and cfg.ckpt_path:
        step, run_id, resume_cfg = validate_checkpoint(Path(cfg.ckpt_path))
        run = wandb.init(
            project="PART-linear-segmentation",
            group=run_id,
            config=resume_cfg,
            name=f"{run_id}-{step:07d}",
        )
    else:
        run = wandb.init(project="PART-linear-segmentation")
        run_id = run.id
        step = 0

    run_name = run.name
    run.config.update({"tags": cfg.tags}, allow_val_change=True)
    logger = WandbLogger(experiment=run)

    seed_everything(0)
    input_size = cfg.input_size

    data_config = cfg.data
    train_config = cfg

    # Init transforms and train data
    train_transforms = Compose(
        [
            RandomResizedCrop(size=input_size, scale=(0.8, 1.0)),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_image_transforms = T.Compose(
        [
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_target_transforms = T.Compose(
        [
            T.Resize((input_size, input_size), interpolation=InterpolationMode.NEAREST),
            T.ToTensor(),
        ]
    )

    data_dir = data_config.data_dir
    dataset_name = data_config.dataset_name
    if dataset_name == "voc":
        num_classes = 21
        ignore_index = 255
        data_module = VOCDataModule(
            batch_size=train_config.batch_size,
            return_masks=True,
            num_workers=train_config.num_workers,
            train_split="trainaug",
            val_split="val",
            data_dir=data_dir,
            train_image_transform=train_transforms,
            drop_last=True,
            val_image_transform=val_image_transforms,
            val_target_transform=val_target_transforms,
        )
    elif "coco" in dataset_name:
        assert len(dataset_name.split("-")) == 2
        mask_type = dataset_name.split("-")[-1]
        assert mask_type in ["thing", "stuff"]
        if mask_type == "thing":
            num_classes = 12
        else:
            num_classes = 15
        ignore_index = 255
        file_list = os.listdir(os.path.join(data_dir, "images", "train2017"))
        file_list_val = os.listdir(os.path.join(data_dir, "images", "val2017"))
        random.shuffle(file_list_val)
        # sample 10% of train images
        random.shuffle(file_list)
        file_list = file_list[: int(len(file_list) * 0.1)]
        print(f"sampled {len(file_list)} COCO images for training")

        data_module = CocoDataModule(
            batch_size=train_config.batch_size,
            num_workers=train_config.num_workers,
            file_list=file_list,
            data_dir=data_dir,
            file_list_val=file_list_val,
            mask_type=mask_type,
            train_transforms=train_transforms,
            val_transforms=val_image_transforms,
            val_target_transforms=val_target_transforms,
        )
    elif dataset_name == "ade20k":
        num_classes = 151
        ignore_index = 0
        val_transforms = SepTransforms(val_image_transforms, val_target_transforms)
        data_module = Ade20kDataModule(
            data_dir,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            shuffle=False,
            num_workers=train_config.num_workers,
            batch_size=train_config.batch_size,
        )
    elif dataset_name == "cityscapes":
        num_classes = 19
        ignore_index = 255
        val_transforms = SepTransforms(val_image_transforms, val_target_transforms)
        data_module = CityscapesDataModule(
            root=data_dir,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            shuffle=True,
            num_workers=train_config.num_workers,
            batch_size=train_config.batch_size,
        )
    else:
        raise ValueError(f"{dataset_name} not supported")

    # Init backbone and linear head
    backbone = instantiate(cfg.model.net, _convert_="all")
    inference_fn = instantiate(cfg.model.inference_fn, _partial_=True)
    model = LinearFinetune(
        net=backbone,
        inference_fn=inference_fn,
        num_classes=num_classes,
        lr=train_config.lr,
        input_size=input_size,
        val_iters=train_config.val_iters,
        drop_at=train_config.drop_at,
        decay_rate=train_config.decay_rate,
        ignore_index=ignore_index,
    )

    ckpt_path = train_config.ckpt_path if train_config.restart else None

    # Init checkpoint callback storing top 3 heads
    print(f"{train_config.ckpt_dir}, {run_name}")
    checkpoint_dir = os.path.join(train_config.ckpt_dir, run_name)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="miou_val",
        filename="ckp-{epoch:02d}-{miou_val:.4f}",
        save_top_k=3,
        mode="max",
        verbose=True,
    )

    # Init trainer and start training head
    trainer = Trainer(
        num_sanity_val_steps=0,
        logger=logger,
        max_epochs=train_config.max_epochs,
        devices=1,
        accelerator="cuda",
        fast_dev_run=train_config.fast_dev_run,
        log_every_n_steps=50,
        benchmark=True,
        deterministic=False,
        detect_anomaly=False,
        callbacks=[checkpoint_callback],
        precision=train_config.get("precision", None),
    )
    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=ckpt_path,
    )

    if isinstance(logger, WandbLogger):
        logger.experiment.finish()


class LinearFinetune(pl.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        inference_fn: callable,
        num_classes: int,
        lr: float,
        input_size: int,
        val_iters: int,
        drop_at: int,
        decay_rate: float = 0.1,
        ignore_index: int = 255,
    ) -> None:
        """Train a linear segmentation head on frozen features.

        :param net: Backbone model to freeze.
        :param num_classes: Number of segmentation classes.
        :param lr: Learning rate for the head optimizer.
        :param input_size: Input image size in pixels.
        :param val_iters: Maximum validation batches per epoch.
        :param drop_at: Scheduler step interval.
        :param decay_rate: Scheduler decay rate.
        :param ignore_index: Label value to ignore when computing loss.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["net"])
        self.model = net
        self.inference_fn = inference_fn
        in_ch = timm_embed_dim(net)
        self.finetune_head = nn.Conv2d(in_ch, num_classes, 1)

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.miou_metric = PredsmIoU(num_classes, num_classes)
        self.num_classes = num_classes
        self.lr = lr
        self.val_iters = val_iters
        self.input_size = input_size
        self.spatial_res = input_size // timm_patch_size(net)
        self.drop_at = drop_at
        self.ignore_index = ignore_index
        self.decay_rate = decay_rate
        self.train_mask_size = 100
        self.val_mask_size = 100

    def on_after_backward(self):
        # Freeze all layers of backbone
        for param in self.model.parameters():
            param.requires_grad = False

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.finetune_head.parameters(),
            weight_decay=0.0001,
            momentum=0.9,
            lr=self.lr,
        )
        scheduler = StepLR(optimizer, gamma=self.decay_rate, step_size=self.drop_at)
        return [optimizer], [scheduler]

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        imgs, masks = batch
        assert imgs.size(3) == self.input_size
        self.model.eval()

        with torch.no_grad():
            tokens = self.inference_fn(self.model, imgs)
            tokens = nn.functional.interpolate(
                tokens,
                size=(self.train_mask_size, self.train_mask_size),
                mode="bilinear",
            )
        mask_preds = self.finetune_head(tokens)

        masks *= 255
        if self.train_mask_size != self.input_size:
            with torch.no_grad():
                masks = nn.functional.interpolate(
                    masks,
                    size=(self.train_mask_size, self.train_mask_size),
                    mode="nearest",
                )

        loss = self.criterion(mask_preds, masks.long().squeeze())

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        if self.val_iters is None or batch_idx < self.val_iters:
            with torch.no_grad():
                imgs, masks = batch
                tokens = self.inference_fn(self.model, imgs)
                tokens = nn.functional.interpolate(
                    tokens,
                    size=(self.val_mask_size, self.val_mask_size),
                    mode="bilinear",
                )
                mask_preds = self.finetune_head(tokens)

                # downsample masks and preds
                gt = masks * 255
                gt = nn.functional.interpolate(
                    gt, size=(self.val_mask_size, self.val_mask_size), mode="nearest"
                )
                valid = gt != self.ignore_index  # mask to remove object boundary class
                mask_preds = torch.argmax(mask_preds, dim=1).unsqueeze(1)

                # update metric
                self.miou_metric.update(gt[valid], mask_preds[valid])

    def on_validation_epoch_end(self) -> None:
        miou = self.miou_metric.compute(True, many_to_one=False, linear_probe=True)[0]
        self.miou_metric.reset()
        print(miou)
        self.log("miou_val", round(miou, 6))


if __name__ == "__main__":
    main()
