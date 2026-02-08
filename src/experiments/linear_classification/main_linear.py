import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
import lightning as L  # Lightning Fabric
import logging

from src.utils import pylogger, checkpointer

torch.set_float32_matmul_precision("high")

log = pylogger.RankedLogger(__name__, rank_zero_only=True)
logging.basicConfig(level=logging.INFO)


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


@torch.no_grad()
def _top1_correct(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return number of correct top-1 predictions."""
    pred = output.argmax(dim=1)
    return pred.eq(target).sum()


def train_one_epoch(
    backbone: nn.Module,
    classifier: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    inference_fn: callable,
    fabric: L.Fabric,
    global_step: int,
    steps_per_epoch: int,
    log_freq: int = 10,
    epoch: int = 0,
) -> int:
    """Run one training epoch.

    :param backbone: Frozen feature extractor.
    :param classifier: Linear classifier to train.
    :param loader: Dataloader providing training images and labels.
    :param optimizer: Optimizer for ``classifier`` parameters.
    :param inference_fn: Function extracting features from ``backbone``.
    :param fabric: Lightning Fabric handler.
    :param global_step: Current global step.
    :param steps_per_epoch: Number of training iterations per epoch.
    :param log_freq: Log train metrics every N steps (<=0 disables).
    :param epoch: Current epoch index.
    :returns: Updated global step after the epoch.
    """
    criterion = nn.CrossEntropyLoss()
    classifier.train()

    window_loss_sum = torch.tensor(0.0, device=fabric.device)
    window_correct1 = torch.tensor(0, device=fabric.device, dtype=torch.long)
    window_count = torch.tensor(0, device=fabric.device, dtype=torch.long)

    for itr, (imgs, target) in enumerate(loader, start=1):
        with torch.no_grad(), fabric.autocast():
            feats = inference_fn(backbone, imgs)
        with fabric.autocast():
            output = classifier(feats)
            loss = criterion(output, target)

        fabric.backward(loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        batch_size = int(target.shape[0])
        window_loss_sum += loss.detach() * batch_size
        window_correct1 += _top1_correct(output.detach(), target)
        window_count += batch_size

        global_step += 1
        should_log = (
            fabric.is_global_zero and log_freq > 0 and (global_step % log_freq == 0)
        )
        if should_log:
            num = int(window_count.item())
            loss_avg = float(window_loss_sum.item() / max(1, num))
            acc1 = float(window_correct1.item() * 100.0 / max(1, num))
            lr = float(optimizer.param_groups[0]["lr"])
            log.info(
                "[%d, %5d/%5d] loss: %.4f acc1: %.2f lr: %.2e"
                % (epoch + 1, itr, steps_per_epoch, loss_avg, acc1, lr)
            )
            window_loss_sum.zero_()
            window_correct1.zero_()
            window_count.zero_()

    return global_step


def run_validation(
    backbone: nn.Module,
    classifier: nn.Module,
    loader: DataLoader,
    inference_fn: callable,
    fabric: L.Fabric,
    epoch: int,
) -> dict[str, float]:
    """Run validation and update best metrics.

    :param backbone: Frozen feature extractor.
    :param classifier: Linear classifier to evaluate.
    :param loader: Validation dataloader.
    :param inference_fn: Feature extraction function.
    :returns: Dictionary with validation and best scores.
    """
    criterion = nn.CrossEntropyLoss()
    classifier.eval()

    loss_sum = torch.tensor(0.0, device=fabric.device)
    correct1 = torch.tensor(0, device=fabric.device, dtype=torch.long)
    count = torch.tensor(0, device=fabric.device, dtype=torch.long)
    with torch.no_grad(), fabric.autocast():
        for imgs, target in loader:
            feats = inference_fn(backbone, imgs)
            output = classifier(feats)
            loss = criterion(output, target)
            batch_size = int(target.shape[0])
            loss_sum += loss.detach() * batch_size
            count += batch_size
            correct1 += _top1_correct(output.detach(), target)

    n = int(count.item())
    scores = {
        "val/loss": float(loss_sum.item() / max(1, n)),
        "val/top-1-acc": float(correct1.item() * 100.0 / max(1, n)),
    }
    log.info(
        "Epoch {}: val loss {:.4f} top1 {:.2f}".format(
            epoch,
            scores["val/loss"],
            scores["val/top-1-acc"],
        )
    )
    return scores


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
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    steps_per_epoch = len(train_loader)
    total_steps = int(cfg.train.epochs) * int(steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    fabric = L.Fabric(
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        precision=cfg.train.precision,
    )
    fabric.seed_everything(cfg.train.seed)

    classifier, optimizer = fabric.setup(classifier, optimizer)
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)
    backbone = backbone.to(fabric.device)
    best_val_top1 = float("-inf")
    global_step = 0
    for epoch in range(cfg.train.epochs):
        global_step = train_one_epoch(
            backbone=backbone,
            classifier=classifier,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            inference_fn=inference_fn,
            fabric=fabric,
            global_step=global_step,
            steps_per_epoch=steps_per_epoch,
            log_freq=cfg.train.log_freq,
            epoch=epoch,
        )
        if epoch % cfg.train.val_freq == 0 or epoch == cfg.train.epochs - 1:
            val_scores = run_validation(
                backbone=backbone,
                classifier=classifier,
                loader=val_loader,
                inference_fn=inference_fn,
                fabric=fabric,
                epoch=epoch,
            )
            if (
                cfg.train.get("save_best", True)
                and fabric.is_global_zero
                and val_scores["val/top-1-acc"] > best_val_top1
            ):
                best_val_top1 = val_scores["val/top-1-acc"]
                checkpointer.save_checkpoint(
                    fabric=fabric,
                    model=classifier,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    filepath=str(output_dir / "classifier_best.ckpt"),
                    scheduler=scheduler,
                    val_scores=val_scores,
                    feat_dim=int(cfg.model.feat_dim),
                    num_labels=int(cfg.data.num_labels),
                )

    if cfg.train.get("save_last", True) and fabric.is_global_zero:
        checkpointer.save_checkpoint(
            fabric=fabric,
            model=classifier,
            optimizer=optimizer,
            epoch=cfg.train.epochs - 1,
            global_step=global_step,
            filepath=str(output_dir / "classifier_last.ckpt"),
            scheduler=scheduler,
            feat_dim=int(cfg.model.feat_dim),
            num_labels=int(cfg.data.num_labels),
        )

    log.info("Training finished. Best accuracy: {:.2f}".format(best_val_top1))


if __name__ == "__main__":
    main()
