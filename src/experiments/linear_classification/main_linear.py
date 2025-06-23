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
from torchmetrics.wrappers.abstract import WrapperMetric

from src.utils import pylogger
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

OmegaConf.register_new_resolver("eval", eval)

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
        "Epoch {}: val loss {:.4f} top1 {:.2f} top5 {:.2f} best {:.2f}".format(
            epoch,
            current_scores['val/loss'],
            current_scores['val/top-1-acc'],
            current_scores['val/top-5-acc'],
            best_scores['val/best-top-1-acc']
        )
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

    classifier = LinearClassifier(cfg.model.feat_dim, cfg.data.num_labels)

    total_batch_size = cfg.train.batch_size * cfg.train.devices
    log.info("Total batch size: {}".format(total_batch_size))
    lr = cfg.train.blr * total_batch_size / 256
    log.info("Using learning rate: {:.6f}".format(lr))
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.train.epochs)

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
    fabric.seed_everything(cfg.train.seed)
    classifier, optimizer = fabric.setup(classifier, optimizer)
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)
    backbone = backbone.to(fabric.device)

    train_metrics = TrainMetrics(cfg.data.num_labels).to(fabric.device)
    val_metrics = ValMetrics(cfg.data.num_labels).to(fabric.device)
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
        "Training finished. Best accuracy: {:.2f}".format(
            best_val_metrics.compute()['val/best-top-1-acc']
        )
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        if wandb.run:
            wandb.finish()
