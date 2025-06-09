import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader


def extract_cls_features(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        feats = model.forward_features(x)
    if feats.dim() == 3:
        feats = feats[:, 0]
    return torch.nn.functional.normalize(feats, dim=1)


def gather_features(model, loader, device):
    feats, labels = [], []
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            x, y = batch[0], batch[1]
        else:
            x, y = batch["image"], batch["label"]
        x = x.to(device)
        y = y.to(device)
        feats.append(extract_cls_features(model, x))
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def knn_predict(train_feats, train_labels, test_feats, k, T, num_classes):
    sim = torch.mm(test_feats, train_feats.t())
    topk_sim, topk_idx = sim.topk(k, dim=1, largest=True, sorted=True)
    topk_labels = train_labels[topk_idx]
    weights = torch.softmax(topk_sim / T, dim=1)
    preds = torch.zeros(test_feats.size(0), num_classes, device=test_feats.device)
    preds.scatter_add_(1, topk_labels, weights)
    return preds


def knn_eval(model, train_loader, val_loader, k_list, temperature, device):
    train_feats, train_labels = gather_features(model, train_loader, device)
    val_feats, val_labels = gather_features(model, val_loader, device)
    num_classes = int(train_labels.max().item() + 1)
    results = {}
    for k in k_list:
        probs = knn_predict(train_feats, train_labels, val_feats, k, temperature, num_classes)
        top1 = (probs.argmax(1) == val_labels).float().mean().item()
        top5 = (
            probs.topk(min(5, num_classes), dim=1).indices.eq(val_labels.unsqueeze(1)).any(dim=1).float().mean().item()
        )
        results[k] = {"top1": top1, "top5": top5}
    return results


@hydra.main(version_base="1.3", config_path="../fabric_configs/experiment/knn", config_name="config")
def main(cfg: DictConfig) -> None:
    device = torch.device(cfg.device)
    model = instantiate(cfg.model, _convert_="all").to(device).eval()

    train_ds = instantiate(cfg.data.train, _convert_="all")
    val_ds = instantiate(cfg.data.val, _convert_="all")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    results = knn_eval(model, train_loader, val_loader, cfg.nb_knn, cfg.temperature, device)
    for k, res in results.items():
        print(f"k={k}: top1={res['top1']*100:.2f}, top5={res['top5']*100:.2f}")


if __name__ == "__main__":
    main()
