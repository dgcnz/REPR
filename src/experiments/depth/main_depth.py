import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import lightning as L  # Lightning Fabric
import timm
import h5py
import numpy as np
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from src.utils.timm_utils import timm_embed_dim, timm_patch_size

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
@hydra.main(version_base=None, config_path=None)
def main(cfg: DictConfig):
    # Print config
    print(OmegaConf.to_yaml(cfg))

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
        mat_path=cfg.dataset.mat_path,
        indices=list(range(cfg.dataset.train_split)),
        img_transform=img_transform,
        depth_transform=depth_transform
    )
    val_ds   = NYULabeledDataset(
        mat_path=cfg.dataset.mat_path,
        indices=list(range(cfg.dataset.train_split, cfg.dataset.total)),
        img_transform=img_transform,
        depth_transform=depth_transform
    )
    train_loader = DataLoader(train_ds,
                              batch_size=cfg.train.batch_size,
                              shuffle=True,
                              num_workers=cfg.train.num_workers)
    val_loader   = DataLoader(val_ds,
                              batch_size=cfg.train.batch_size,
                              shuffle=False,
                              num_workers=cfg.train.num_workers)

    # Model
    backbone = timm.create_model(cfg.model.backbone, pretrained=cfg.model.pretrained)
    backbone.head = nn.Identity()
    for p in backbone.parameters(): p.requires_grad = False

    class DepthProbe(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.patch_size = timm_patch_size(backbone)
            self.emb_dim = timm_embed_dim(backbone)
            self.head = nn.Conv2d(self.emb_dim, 1, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, C, H, W = x.shape
            tokens = self.backbone.forward_features(x)[:, 1:, :]
            n = H // self.patch_size
            feat = tokens.transpose(1, 2).view(B, -1, n, n)
            up = F.interpolate(feat,
                               scale_factor=self.patch_size,
                               mode='bilinear',
                               align_corners=False)
            return self.head(up)

    model = DepthProbe()
    optimizer = AdamW(model.head.parameters(),
                      lr=cfg.train.lr,
                      weight_decay=cfg.train.weight_decay)
    scheduler = StepLR(optimizer,
                       step_size=cfg.train.step_size,
                       gamma=cfg.train.gamma)
    criterion = nn.MSELoss()

    fabric = L.Fabric(accelerator=cfg.train.accelerator,
                      devices=cfg.train.devices,
                      precision=cfg.train.precision)
    model, optimizer = fabric.setup(
        model, optimizer)
    train_loader, val_loader = fabric.setup_dataloaders(
        train_loader, val_loader)

    
    for epoch in range(cfg.train.epochs):
        model.train()
        for imgs, depths in train_loader:
            preds = model(imgs)
            loss = criterion(preds, depths)
            fabric.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()

        model.eval()
        sum_sq = 0.0
        count = 0
        for imgs, depths in val_loader:
            preds = model(imgs)
            err = preds - depths
            sum_sq += (err**2).sum().item()
            count += depths.numel()
        rmse = (sum_sq / count)**0.5
        fabric.print(f"Epoch {epoch:02d} RMSE: {rmse:.4f}")

if __name__=='__main__':
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        'dataset': {'mat_path':'/mnt/sdb1/datasets/nyuv2/nyu_depth_v2_labeled.mat','train_split':795,'total':1449},
        'model': {'backbone':'vit_small_patch16_224','pretrained':False},
        'train': {'batch_size':16,
                  'num_workers':0,
                  'lr':1e-3,
                  'weight_decay':1e-4,
                  'step_size':10,
                  'gamma':0.5,
                  'epochs':10,
                  'accelerator':'gpu',
                  'devices':1,
                  'precision':'16-mixed',
                  'device':'cuda',
                  'input_size':224,
                  'img_mean':[0.485,0.456,0.406],
                  'img_std':[0.229,0.224,0.225]}
    })
    main(cfg)
