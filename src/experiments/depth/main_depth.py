import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import lightning as L  # Lightning Fabric
import h5py
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from lightning.pytorch.loggers import WandbLogger
from tqdm.auto import tqdm
import wandb
from torchmetrics import MeanMetric
from torchmetrics.regression import MeanSquaredError

from src.utils.timm_utils import timm_embed_dim, timm_patch_size
from src.utils import pylogger

log = pylogger.RankedLogger(__name__)
# --- NYU Labeled Dataset Definition ---
class NYULabeledDataset(Dataset):
    """
    PyTorch Dataset for NYU-Depth V2 labeled subset stored in MATLAB v7.3 .mat (HDF5) format.
    Applies separate transforms for RGB and depth.
    Assumes images shape (N, C, H, W) and depths shape (N, H, W).
    """
    def __init__(self,
                 mat_path: str,
                 indices: list[int],
                 img_transform=None,
                 depth_transform=None):
        super().__init__()
        f = h5py.File(mat_path, 'r')
        self.images = f['images']      # (N, 3, H, W)
        self.depths = f['depths']      # (N, H, W)
        self.indices = indices
        self.img_transform = img_transform
        self.depth_transform = depth_transform
        assert self.images.ndim == 4, f"Expected images.ndim==4, got {self.images.shape}"
        assert self.depths.ndim == 3, f"Expected depths.ndim==3, got {self.depths.shape}"

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = self.indices[idx]
        img = Image.fromarray(self.images[i].transpose(1, 2, 0))  # H,W,C -> C,H,W
        depth = Image.fromarray(self.depths[i])  # H,W

        img = self.img_transform(img) if self.img_transform else T.ToTensor()(img)
        depth = self.depth_transform(depth) if self.depth_transform else T.ToTensor()(depth)
        return img, depth

# --- Main Training Script ---
@hydra.main(
    version_base="1.3",
    config_path="../../../fabric_configs/experiment/depth",
    config_name="config",
)
def main(cfg: DictConfig):
    log.info(OmegaConf.to_yaml(cfg))

    run = wandb.init(project="PART-depth")
    run.config.update({"tags": cfg.get("tags", [])}, allow_val_change=True)
    logger = WandbLogger(experiment=run)

    try:
        # Transforms
        img_transform = T.Compose([
            T.Resize((cfg.train.input_size, cfg.train.input_size)),
            T.ToTensor(),
            T.Normalize(mean=cfg.train.img_mean, std=cfg.train.img_std)
        ])
        depth_transform = T.Compose([
            T.Resize((cfg.train.input_size, cfg.train.input_size), interpolation=InterpolationMode.NEAREST),
            T.ToTensor(),
        ])

        # Data loaders
        train_ds = NYULabeledDataset(
            mat_path=cfg.data.mat_path,
            indices=list(range(cfg.data.train_split)),
            img_transform=img_transform,
            depth_transform=depth_transform
        )
        val_ds = NYULabeledDataset(
            mat_path=cfg.data.mat_path,
            indices=list(range(cfg.data.train_split, cfg.data.total)),
            img_transform=img_transform,
            depth_transform=depth_transform
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            num_workers=cfg.train.num_workers,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=cfg.train.num_workers,
        )

        # Model
        backbone = instantiate(cfg.model.net, _convert_="all")
        inference_fn = instantiate(cfg.model.inference_fn)
        backbone.head = nn.Identity()
        for p in backbone.parameters():
            p.requires_grad = False

        class DepthProbe(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = backbone
                self.inference_fn = inference_fn
                self.patch_size = timm_patch_size(backbone)
                self.emb_dim = timm_embed_dim(backbone)
                self.head = nn.Conv2d(self.emb_dim, 1, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                feats = self.inference_fn(self.backbone, x)
                up = F.interpolate(
                    feats,
                    scale_factor=self.patch_size,
                    mode="bilinear",
                    align_corners=False,
                )
                return self.head(up)

        model = DepthProbe()
        optimizer = AdamW(
            model.head.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )
        scheduler = StepLR(
            optimizer,
            step_size=cfg.train.step_size,
            gamma=cfg.train.gamma,
        )
        criterion = nn.MSELoss()

        fabric = L.Fabric(
            accelerator=cfg.train.accelerator,
            devices=cfg.train.devices,
            precision=cfg.train.precision,
            loggers=logger,
        )
        model, optimizer = fabric.setup(model, optimizer)
        train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)
        train_loss_metric = MeanMetric().to(fabric.device)
        val_rmse_metric = MeanSquaredError(squared=False).to(fabric.device)

    # Training loop counters
        global_step = 0


        for epoch in range(cfg.train.epochs):
            model.train()
            train_loss_metric.reset()
            train_iter = tqdm(
                train_loader,
                desc=f"Train {epoch}",
                disable=not fabric.is_global_zero,
            )
            for imgs, depths in train_iter:
                preds = model(imgs)
                loss = criterion(preds, depths)
                fabric.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                train_loss_metric.update(loss)
                if fabric.is_global_zero:
                    fabric.log_dict({"train/loss": float(train_loss_metric.compute())}, step=global_step)
                global_step += 1
            scheduler.step()

            model.eval()
            val_rmse_metric.reset()
            val_iter = tqdm(
                val_loader,
                desc="Val",
                disable=not fabric.is_global_zero,
            )
            for imgs, depths in val_iter:
                preds = model(imgs)
                val_rmse_metric.update(preds, depths)
            rmse = float(val_rmse_metric.compute())
            log.info(f"Epoch {epoch:02d} RMSE: {rmse:.4f}")
            if fabric.is_global_zero:
                fabric.log_dict({"val/rmse": rmse}, step=epoch)

    finally:
        if wandb.run:
            wandb.finish()

if __name__ == "__main__":
    main()
