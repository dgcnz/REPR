# FILE: src/experiments/linear_classification/main_linear.py

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from lightning.pytorch.loggers import WandbLogger
import lightning as L  # Lightning Fabric
import wandb
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.aggregation import MaxMetric
from torchmetrics.wrappers import WrapperMetric

from src.utils import pylogger
from src.utils.timm_utils import timm_embed_dim
from src.utils.timm_utils import extract_vit_cls, extract_meanpool_cls, extract_patch_meanpool

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features."""

    def __init__(self, dim: int, num_labels: int) -> None:
        """Initialize the classifier.

        :param dim: Feature dimension of the backbone.
        :param num_labels: Number of target classes.
        """
        super().__init__()
        self.linear = nn.Linear(dim, num_labels)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TrainMetrics(WrapperMetric):
    """Aggregate training metrics."""

    def __init__(
        self,
        num_classes: int,
        nan_strategy: str = "disable",
        sync_on_compute: bool = False,
    ) -> None:
        super().__init__()
        mean_kwargs = {
            "nan_strategy": nan_strategy,
            "sync_on_compute": sync_on_compute,
        }
        acc_kwargs = {
            "num_classes": num_classes,
            "top_k": 1,
            "sync_on_compute": sync_on_compute,
        }

        self.metrics = nn.ModuleDict(
            {
                "train/loss": MeanMetric(**mean_kwargs),
                "train/acc": MulticlassAccuracy(**acc_kwargs),
            }
        )

    def update(self, preds: torch.Tensor, targets: torch.Tensor, loss: torch.Tensor) -> None:
        self.metrics["train/loss"].update(loss)
        self.metrics["train/acc"].update(preds, targets)

    def reset(self) -> None:  # noqa: D401 - keep same signature
        for m in self.metrics.values():
            m.reset()

    def compute(self) -> dict[str, float]:
        return {
            "train/loss": float(self.metrics["train/loss"].compute()),
            "train/acc": float(self.metrics["train/acc"].compute()) * 100.0,
        }


class ValMetrics(WrapperMetric):
    """Aggregate validation metrics."""

    def __init__(
        self,
        num_classes: int,
        nan_strategy: str = "disable",
        sync_on_compute: bool = False,
    ) -> None:
        super().__init__()
        mean_kwargs = {
            "nan_strategy": nan_strategy,
            "sync_on_compute": sync_on_compute,
        }
        acc_kwargs = {
            "num_classes": num_classes,
            "sync_on_compute": sync_on_compute,
        }

        self.metrics = nn.ModuleDict(
            {
                "val/loss": MeanMetric(**mean_kwargs),
                "val/top-1-acc": MulticlassAccuracy(top_k=1, **acc_kwargs),
                "val/top-5-acc": MulticlassAccuracy(top_k=5, **acc_kwargs),
            }
        )

    def update(
        self, preds: torch.Tensor, targets: torch.Tensor, loss: torch.Tensor
    ) -> None:
        self.metrics["val/loss"].update(loss)
        self.metrics["val/top-1-acc"].update(preds, targets)
        self.metrics["val/top-5-acc"].update(preds, targets)

    def reset(self) -> None:  # noqa: D401 - keep same signature
        for m in self.metrics.values():
            m.reset()

    def compute(self) -> dict[str, float]:
        return {
            "val/loss": float(self.metrics["val/loss"].compute()),
            "val/top-1-acc": float(self.metrics["val/top-1-acc"].compute()) * 100.0,
            "val/top-5-acc": float(self.metrics["val/top-5-acc"].compute()) * 100.0,
        }


class BestValMetrics(WrapperMetric):
    """Track best validation metrics across epochs."""

    def __init__(
        self,
        nan_strategy: str = "disable",
        sync_on_compute: bool = False,
    ) -> None:
        super().__init__()
        mean_kwargs = {
            "nan_strategy": nan_strategy,
            "sync_on_compute": sync_on_compute,
        }

        self.metrics = nn.ModuleDict(
            {"val/best-top-1-acc": MaxMetric(**mean_kwargs)}
        )

    def update(self, val_scores: dict[str, float]) -> None:
        self.metrics["val/best-top-1-acc"].update(val_scores["val/top-1-acc"])

    def compute(self) -> dict[str, float]:
        return {
            "val/best-top-1-acc": float(
                self.metrics["val/best-top-1-acc"].compute()
            )
        }


def train_one_epoch(
    backbone: nn.Module,
    classifier: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    inference_fn: callable,
    metrics: TrainMetrics,
    fabric: L.Fabric,
    global_step: int,
    epoch: int = 0,
) -> int:
    """Run one training epoch.

    :param backbone: Frozen feature extractor.
    :param classifier: Linear classifier to train.
    :param loader: Dataloader providing training images and labels.
    :param optimizer: Optimizer for ``classifier`` parameters.
    :param inference_fn: Function extracting features from ``backbone``.
    :param metrics: Metric aggregator updated during training.
    :param fabric: Lightning Fabric handler.
    :param global_step: Current global step.
    :param epoch: Current epoch index.
    :returns: Updated global step after the epoch.
    """
    criterion = nn.CrossEntropyLoss()
    classifier.train()
    metrics.reset()
    train_iter = tqdm(
        loader,
        desc=f"Train {epoch}",
        disable=not fabric.is_global_zero,
    )
    for imgs, target in train_iter:
        with torch.no_grad():
            feats = inference_fn(backbone, imgs)
        output = classifier(feats)
        loss = criterion(output, target)
        fabric.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        metrics.update(output.detach(), target, loss.detach())
        if fabric.is_global_zero:
            fabric.log_dict(metrics.compute(), step=global_step)
        global_step += 1
    return global_step


def run_validation(
    backbone: nn.Module,
    classifier: nn.Module,
    loader: DataLoader,
    val_metrics: ValMetrics,
    best_val_metrics: BestValMetrics,
    inference_fn: callable,
    fabric: L.Fabric,
    epoch: int,
    global_step: int,
) -> dict[str, float]:
    """Run validation and update best metrics.

    :param backbone: Frozen feature extractor.
    :param classifier: Linear classifier to evaluate.
    :param loader: Validation dataloader.
    :param val_metrics: Metric aggregator for current epoch.
    :param best_val_metrics: Tracker for best metrics across epochs.
    :param inference_fn: Feature extraction function.
    :returns: Dictionary with validation and best scores.
    """
    criterion = nn.CrossEntropyLoss()
    classifier.eval()
    val_metrics.reset()
    val_iter = tqdm(
        loader,
        desc="Val",
        disable=not fabric.is_global_zero,
    )
    with torch.no_grad():
        for imgs, target in val_iter:
            feats = inference_fn(backbone, imgs)
            output = classifier(feats)
            loss = criterion(output, target)
            val_metrics.update(output, target, loss)
    current_scores = val_metrics.compute()
    best_val_metrics.update(current_scores)
    best_scores = best_val_metrics.compute()
    log.info(
        "Epoch %d: val loss %.4f top1 %.2f top5 %.2f best %.2f",
        epoch,
        current_scores["val/loss"],
        current_scores["val/top-1-acc"],
        current_scores["val/top-5-acc"],
        best_scores["val/best-top-1-acc"],
    )
    if fabric.is_global_zero:
        fabric.log_dict({**current_scores, **best_scores}, step=global_step)
    return {**current_scores, **best_scores}


@hydra.main(
    version_base="1.3",
    config_path="../../../fabric_configs/experiment/linear_classification",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """Run a linear classification experiment.

    :param cfg: Hydra configuration composed from ``config.yaml``.
    """
    log.info(OmegaConf.to_yaml(cfg))
    run = wandb.init(project="PART-linear-classification")
    run.config.update({"tags": cfg.get("tags", [])}, allow_val_change=True)
    logger = WandbLogger(experiment=run)

    train_ds = instantiate(cfg.data.train)
    val_ds = instantiate(cfg.data.val)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    inference_fn = instantiate(cfg.model.inference_fn)

    backbone = instantiate(cfg.model.net, _convert_="all")
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    feat_dim = timm_embed_dim(backbone) * (
        cfg.n_last_blocks + int(cfg.avgpool_patchtokens)
    )
    classifier = LinearClassifier(feat_dim, cfg.num_labels)

    optimizer = torch.optim.SGD(classifier.parameters(), lr=cfg.train.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.train.epochs)

    fabric = L.Fabric(
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        precision=cfg.train.precision,
        loggers=logger,
    )
    classifier, optimizer = fabric.setup(classifier, optimizer)
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)
    backbone = backbone.to(fabric.device)

    train_metrics = TrainMetrics(cfg.num_labels).to(fabric.device)
    val_metrics = ValMetrics(cfg.num_labels).to(fabric.device)
    best_val_metrics = BestValMetrics().to(fabric.device)
    global_step = 0
    for epoch in range(cfg.train.epochs):
        global_step = train_one_epoch(
            backbone=backbone,
            classifier=classifier,
            loader=train_loader,
            optimizer=optimizer,
            inference_fn=inference_fn,
            metrics=train_metrics,
            fabric=fabric,
            global_step=global_step,
            epoch=epoch,
        )
        scheduler.step()
        if epoch % cfg.train.val_freq == 0 or epoch == cfg.train.epochs - 1:
            run_validation(
                backbone=backbone,
                classifier=classifier,
                loader=val_loader,
                val_metrics=val_metrics,
                best_val_metrics=best_val_metrics,
                inference_fn=inference_fn,
                fabric=fabric,
                epoch=epoch,
                global_step=global_step,
            )

    log.info(
        "Training finished. Best accuracy: %.2f",
        best_val_metrics.compute()["val/best-top-1-acc"],
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        if wandb.run:
            wandb.finish()


# FILE: src/utils/timm_utils.py
import torch
from torch import Tensor
from jaxtyping import Float


def timm_patch_size(model: torch.nn.Module) -> int:
    """Return effective patch size for a timm model.

    :param model: timm model.
    :returns: Patch size mapped to a single token.
    """
    name = model.default_cfg.get("architecture", "")
    if name.startswith("vit_"):
        size = getattr(model.patch_embed, "patch_size", None)
        if isinstance(size, tuple):
            return size[-1]
        if isinstance(size, int):
            return size
        return model.patch_embed.proj.kernel_size[-1]
    elif name.startswith("convnext_"):
        patch = model.stem[0].stride[0]
        for stage in model.stages:
            ds = getattr(stage, "downsample", None)
            if ds is None:
                continue
            for m in ds.modules():
                if isinstance(m, torch.nn.Conv2d):
                    patch *= m.stride[0]
        return patch
    raise ValueError(f"Unsupported timm model: {name}")

def timm_embed_dim(model: torch.nn.Module) -> int:
    """Return embedding dimension for a timm model.
    
    :param model: timm model.
    :returns: Embedding dimension.
    """
    name = model.default_cfg.get("architecture", "")
    if name.startswith("vit_"):
        return model.embed_dim
    elif name.startswith("convnext_"):
        return model.num_features
    raise ValueError(f"Unsupported timm model: {name}")


def forward_features_vit(
    model: torch.nn.Module, imgs: Float[Tensor, "B C H W"]
) -> Float[Tensor, "B D h w"]:
    """Return patch feature map for a timm model.

    :param model: A timm model supporting ``forward_intermediates``.
    :param imgs: Input batch of images.
    :returns: Final feature map in ``NCHW`` format.
    """

    with torch.no_grad():
        feats = model.forward_intermediates(
            imgs,
            indices=[-1],
            return_prefix_tokens=False,
            output_fmt="NCHW",
            intermediates_only=True,
        )[0]
    return feats


def forward_features_convnext(
    model: torch.nn.Module, imgs: Float[Tensor, "B C H W"]
) -> Float[Tensor, "B D h w"]:
    """Return patch feature map for a convnext timm model.
    
    :param model: A timm model supporting ``forward_intermediates``.
    :param imgs: Input batch of images.
    :returns: Final feature map in ``NCHW`` format.
    """ 
    with torch.no_grad():
        feats = model.forward_intermediates(
            imgs,
            indices=[-1],
            output_fmt="NCHW",
            intermediates_only=True,
        )[0]
        feats = model.norm_pre(feats)
    return feats


def extract_vit_cls(
    model: torch.nn.Module,
    imgs: Float[Tensor, "B C H W"],
    n_last_blocks: int,
) -> Float[Tensor, "B d"]:
    """Return concatenated CLS tokens from the last ``n`` blocks.

    :param model: Vision Transformer backbone.
    :param imgs: Batch of input images.
    :param n_last_blocks: Number of blocks to extract features from.
    :returns: Flattened CLS token features.
    """
    with torch.no_grad():
        outs = model.get_intermediate_layers(imgs, n_last_blocks)
    cls_tokens = [x[:, 0] for x in outs]
    return torch.cat(cls_tokens, dim=-1)


def extract_meanpool_cls(
    model: torch.nn.Module,
    imgs: Float[Tensor, "B C H W"],
    n_last_blocks: int,
) -> Float[Tensor, "B d"]:
    """Return CLS tokens plus mean pooled patch tokens from the last layer.

    :param model: Vision Transformer backbone.
    :param imgs: Batch of input images.
    :param n_last_blocks: Number of blocks to extract features from.
    :returns: Flattened features.
    """
    with torch.no_grad():
        outs = model.get_intermediate_layers(imgs, n_last_blocks)
    cls_tokens = torch.cat([x[:, 0] for x in outs], dim=-1)
    prefix = getattr(model, "num_prefix_tokens", 1)
    mean_tokens = torch.mean(outs[-1][:, prefix:], dim=1)
    return torch.cat((cls_tokens, mean_tokens), dim=-1)


def extract_patch_meanpool(
    model: torch.nn.Module,
    imgs: Float[Tensor, "B C H W"],
    n_last_blocks: int,
) -> Float[Tensor, "B d"]:
    """Return mean pooled patch tokens from the last block.

    :param model: Vision Transformer backbone.
    :param imgs: Batch of input images.
    :param n_last_blocks: Ignored but kept for API compatibility.
    :returns: Flattened mean pooled patch tokens.
    """
    with torch.no_grad():
        outs = model.get_intermediate_layers(imgs, n_last_blocks)
    prefix = getattr(model, "num_prefix_tokens", 1)
    return torch.mean(outs[-1][:, prefix:], dim=1)


