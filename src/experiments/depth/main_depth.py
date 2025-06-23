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
from torchmetrics import MeanMetric, Metric, MinMetric
from torchmetrics.regression import MeanSquaredError
from torchmetrics.wrappers.abstract import WrapperMetric

import random
import torchvision.transforms.functional as TF
from src.utils.timm_utils import timm_embed_dim, timm_patch_size
from src.utils import pylogger


log = pylogger.RankedLogger(__name__)

class ScaleInvariantLogRMSE(Metric):
    """
    SILog as printed on the KITTI depth-prediction page:

        d_i  = log(pred_i) - log(gt_i)
        SILog_raw = 1/n * Σ d_i²  -  (1/n * Σ d_i)²

    This implementation:
    - Masks out any GT <= eps (invalid depth)
    - Clamps preds to >= eps to avoid log(≤0)
    - Guards against dividing by zero valid pixels
    """

    full_state_update = False

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.add_state("sum_d2", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_d",  default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total",  default=torch.tensor(0),   dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Flatten predictions and targets
        p = preds.flatten()
        t = target.flatten()

        # Mask out invalid ground-truth (zeros or near-zero)
        valid = t > self.eps
        assert valid.numel() > 0, "No pixels to evaluate in SILog (all GT ≤ eps)"
        p = torch.clamp(p[valid], min=self.eps)
        t = t[valid]

        # Double-check no non-positive values remain
        assert (p > 0).all(), "Predictions contain non-positive values after clamping"
        assert (t > 0).all(), "Targets contain non-positive values after masking"

        # Compute log differences
        d = torch.log(p) - torch.log(t)

        # Accumulate sums
        self.sum_d2 += (d * d).sum()
        self.sum_d  += d.sum()
        self.total  += d.numel()

    def compute(self) -> torch.Tensor:
        # If no valid pixels, return zero
        if self.total == 0:
            return torch.tensor(0.0, device=self.sum_d.device)

        n = self.total.float()
        mean_sq = self.sum_d2 / n
        mean    = self.sum_d  / n
        return mean_sq - mean * mean




class TrainMetrics(WrapperMetric):
    """Wrapper metric for training that consolidates training-specific metrics."""

    def __init__(self, nan_strategy="disable", sync_on_compute=False):
        super().__init__()
        mean_kwargs = {
            "nan_strategy": nan_strategy,
            "sync_on_compute": sync_on_compute,
        }

        self.metrics = nn.ModuleDict(
            {
                "train/loss": MeanMetric(**mean_kwargs),
            }
        )

    def update(self, loss):
        self.metrics["train/loss"].update(loss)

    def reset(self):
        self.metrics["train/loss"].reset()

    def compute(self):
        return {"train/loss": float(self.metrics["train/loss"].compute())}


class ValMetrics(WrapperMetric):
    """Wrapper metric for validation that consolidates validation-specific metrics."""

    def __init__(self, nan_strategy="disable", sync_on_compute=False):
        super().__init__()
        mse_kwargs = {"sync_on_compute": sync_on_compute}

        self.metrics = nn.ModuleDict(
            {
                "val/rmse": MeanSquaredError(squared=False, **mse_kwargs),
                "val/are": AbsoluteRelativeError(),
                "val/silog": ScaleInvariantLogRMSE(),   # raw version
            }
        )

    def update(self, preds, targets):
        self.metrics["val/rmse"].update(preds, targets)
        self.metrics["val/are"].update(preds, targets)
        self.metrics["val/silog"].update(preds, targets)

    def reset(self):
        self.metrics["val/rmse"].reset()
        self.metrics["val/are"].reset()
        self.metrics["val/silog"].reset()

    def compute(self):
        return {
            "val/rmse": float(self.metrics["val/rmse"].compute()),
            "val/are": float(self.metrics["val/are"].compute()),
            "val/silog": float(self.metrics["val/silog"].compute()),
        }


class BestValMetrics(WrapperMetric):
    """Wrapper metric for tracking best validation scores across epochs."""

    def __init__(self, nan_strategy="disable", sync_on_compute=False):
        super().__init__()
        mean_kwargs = {
            "nan_strategy": nan_strategy,
            "sync_on_compute": sync_on_compute,
        }

        self.metrics = nn.ModuleDict(
            {
                "val/rmse_best": MinMetric(**mean_kwargs),
                "val/are_best": MinMetric(**mean_kwargs),
                "val/silog_best": MinMetric(**mean_kwargs),
            }
        )

    def update(self, val_scores_dict):
        """Update best metrics from validation scores dictionary."""
        self.metrics["val/rmse_best"].update(val_scores_dict["val/rmse"])
        self.metrics["val/are_best"].update(val_scores_dict["val/are"])
        self.metrics["val/silog_best"].update(val_scores_dict["val/silog"])

    def compute(self):
        return {
            "val/rmse_best": float(self.metrics["val/rmse_best"].compute()),
            "val/are_best": float(self.metrics["val/are_best"].compute()),
            "val/silog_best": float(self.metrics["val/silog_best"].compute()),
        }


class AbsoluteRelativeError(Metric):
    """Absolute Relative Error metric for depth estimation."""

    def __init__(self):
        super().__init__()
        self.add_state("sum_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Compute absolute relative error: |pred - target| / target
        # Avoid division by zero by adding small epsilon
        eps = 1e-6
        relative_error = torch.abs(preds - target) / (target + eps)
        self.sum_error += relative_error.sum()
        self.total += relative_error.numel()

    def compute(self):
        return self.sum_error / self.total


class PairedRandomCrop:
    def __init__(self, size, scale=(0.8, 1.0), ratio=(3 / 4, 4 / 3)):
        self.size = size
        self.scale, self.ratio = scale, ratio

    def __call__(self, img, depth):
        i, j, h, w = T.RandomResizedCrop.get_params(img, self.scale, self.ratio)
        img = TF.resized_crop(
            img, i, j, h, w, self.size, interpolation=InterpolationMode.BILINEAR
        )
        depth = TF.resized_crop(
            depth, i, j, h, w, self.size, interpolation=InterpolationMode.NEAREST
        )
        return img, depth


class PairedRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, depth):
        if random.random() < self.p:
            img = TF.hflip(img)
            depth = TF.hflip(depth)
        return img, depth


class NYULabeledDataset(Dataset):
    """
    PyTorch Dataset for NYU-Depth V2 labeled subset stored in MATLAB v7.3 .mat (HDF5) format.
    Applies separate transforms for RGB and depth.
    Assumes images shape (N, C, H, W) and depths shape (N, H, W).
    """

    def __init__(
        self,
        mat_path: str,
        indices: list[int],
        img_transform=None,
        depth_transform=None,
        joint_transform=None,
    ):
        super().__init__()
        f = h5py.File(mat_path, "r")
        self.images = f["images"]  # (N, 3, H, W)
        self.depths = f["depths"]  # (N, H, W)
        self.indices = indices
        self.img_transform = img_transform
        self.depth_transform = depth_transform
        self.joint_transform = joint_transform
        assert self.images.ndim == 4, (
            f"Expected images.ndim==4, got {self.images.shape}"
        )
        assert self.depths.ndim == 3, (
            f"Expected depths.ndim==3, got {self.depths.shape}"
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = self.indices[idx]
        img = Image.fromarray(self.images[i].transpose(1, 2, 0))  # H,W,C -> C,H,W
        depth = Image.fromarray(self.depths[i])  # H,W

        if self.joint_transform:
            for t in self.joint_transform:
                img, depth = t(img, depth)
        img = self.img_transform(img) if self.img_transform else T.ToTensor()(img)
        depth = (
            self.depth_transform(depth) if self.depth_transform else T.ToTensor()(depth)
        )

        assert depth.min() >= 0, "Depth has negative values—check your loader!"
        assert (depth > 0).any(), "Depth map is all zeros—did you point at the right file?"
        return img, depth


class DepthProbe(nn.Module):
    def __init__(self, backbone: nn.Module, inference_fn: callable, mask_size: int):
        super().__init__()
        self.backbone = backbone
        self.inference_fn = inference_fn
        self.patch_size = timm_patch_size(backbone)
        self.emb_dim = timm_embed_dim(backbone)
        self.head = nn.Conv2d(self.emb_dim, 1, 1)
        self.mask_size = mask_size

        self.backbone.head = nn.Identity()
        for p in backbone.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.inference_fn(self.backbone, x)
            up = F.interpolate(
                feats,
                size=(self.mask_size, self.mask_size),
                mode="bilinear",
            )
        return self.head(up)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    fabric: L.Fabric,
    metrics: TrainMetrics,
    global_step: int,
    epoch: int = 0,
):
    model.train()
    metrics.reset()
    train_iter = tqdm(
        train_loader,
        desc=f"Train {epoch}",
        disable=not fabric.is_global_zero,
    )
    for imgs, depths in train_iter:
        preds = model(imgs)
        # if depths resolution is different from preds, resize depths
        with torch.no_grad():
            if preds.shape[-2:] != depths.shape[-2:]:
                depths = F.interpolate(
                    depths,
                    size=(model.mask_size, model.mask_size),
                    mode="nearest",
                )
        loss = criterion(preds, depths)
        fabric.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        metrics.update(loss.detach())
        if fabric.is_global_zero:
            fabric.log_dict(metrics.compute(), step=global_step)
        global_step += 1
    scheduler.step()
    return global_step


def run_validation(
    model: nn.Module,
    val_loader: DataLoader,
    val_metrics: ValMetrics,
    best_val_metrics: BestValMetrics,
    fabric: L.Fabric,
    epoch: int,
    global_step: int,
):
    model.eval()
    val_metrics.reset()
    val_iter = tqdm(
        val_loader,
        desc="Val",
        disable=not fabric.is_global_zero,
    )
    for imgs, depths in val_iter:
        with torch.no_grad():
            preds = model(imgs)
            if preds.shape[-2:] != depths.shape[-2:]:
                depths = F.interpolate(
                    depths,
                    size=(model.mask_size, model.mask_size),
                    mode="nearest",
                )
        val_metrics.update(preds, depths)

    # Get current epoch validation scores
    current_scores = val_metrics.compute()
    best_val_metrics.update(current_scores)
    best_scores = best_val_metrics.compute()

    log.info(
        f"Epoch {epoch:02d} RMSE: {current_scores['val/rmse']:.4f}, ARE: {current_scores['val/are']:.4f}"
    )
    log.info(
        f"Best RMSE: {best_scores['val/rmse_best']:.4f}, Best ARE: {best_scores['val/are_best']:.4f}"
    )

    if fabric.is_global_zero:
        fabric.log_dict({**current_scores, **best_scores}, step=global_step)


# --- Main Training Script ---
@hydra.main(
    version_base="1.3",
    config_path="../../../fabric_configs/experiment/depth",
    config_name="config",
)
def main(cfg: DictConfig):
    log.info(OmegaConf.to_yaml(cfg))
    torch.set_float32_matmul_precision(cfg.train.get("fp32", "highest"))

    run = wandb.init(project="PART-depth")
    run.config.update({"tags": cfg.get("tags", [])}, allow_val_change=True)
    logger = WandbLogger(experiment=run)

    joint_transforms = [
        PairedRandomCrop((cfg.train.input_size, cfg.train.input_size)),
        PairedRandomHorizontalFlip(p=0.5),
    ]

    # Image-only photometric + normalization
    img_transform = T.Compose(
        [
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=cfg.train.img_mean, std=cfg.train.img_std),
        ]
    )

    # Depth-only to-Tensor
    depth_transform = T.Compose(
        [
            T.ToTensor(),
        ]
    )

    # Validation only: a deterministic resize → tensor → normalize
    val_img_transform = T.Compose(
        [
            T.Resize((cfg.train.input_size, cfg.train.input_size)),
            T.ToTensor(),
            T.Normalize(mean=cfg.train.img_mean, std=cfg.train.img_std),
        ]
    )
    val_depth_transform = T.Compose(
        [
            T.Resize(
                (cfg.train.input_size, cfg.train.input_size),
                interpolation=InterpolationMode.NEAREST,
            ),
            T.ToTensor(),
        ]
    )

    # Data loaders
    train_ds = NYULabeledDataset(
        mat_path=cfg.data.mat_path,
        indices=list(range(cfg.data.train_split)),
        img_transform=img_transform,
        depth_transform=depth_transform,
        joint_transform=joint_transforms,
    )
    val_ds = NYULabeledDataset(
        mat_path=cfg.data.mat_path,
        indices=list(range(cfg.data.train_split, cfg.data.total)),
        img_transform=val_img_transform,
        depth_transform=val_depth_transform,
        joint_transform=None,
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

    model = DepthProbe(
        backbone=backbone, inference_fn=inference_fn, mask_size=cfg.train.mask_size
    )
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
    for logger in fabric._loggers:
        logger.log_hyperparams(
            OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        )
    fabric.seed_everything(cfg.train.seed, workers=True)
    model, optimizer = fabric.setup(model, optimizer)
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    # Initialize separate metric wrappers
    train_metrics = TrainMetrics().to(fabric.device)
    val_metrics = ValMetrics().to(fabric.device)
    best_val_metrics = BestValMetrics().to(fabric.device)

    # Training loop counters
    global_step = 0

    for epoch in range(cfg.train.epochs):
        global_step = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            fabric=fabric,
            metrics=train_metrics,
            global_step=global_step,
            epoch=epoch,
        )
        run_validation(
            model=model,
            val_loader=val_loader,
            val_metrics=val_metrics,
            best_val_metrics=best_val_metrics,
            fabric=fabric,
            epoch=epoch,
            global_step=global_step,
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        if wandb.run:
            wandb.finish()
