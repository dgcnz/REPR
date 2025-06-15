import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True

@torch.compile
def extract_cls_features(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=True):
        with torch.no_grad():
            feats = model.forward_features(x)
        if feats.dim() == 3:
            feats = feats[:, 0]
        return torch.nn.functional.normalize(feats, dim=1)


def gather_features(model, loader, device):
    feats, labels = [], []
    logging.info(f"Gathering features using device: {device}")
    for batch in tqdm(loader, desc="Gathering features"):
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
    logging.info("Starting k-NN evaluation.")
    logging.info("Gathering training features...")
    train_feats, train_labels = gather_features(model, train_loader, device)
    logging.info("Gathering validation features...")
    val_feats, val_labels = gather_features(model, val_loader, device)
    num_classes = int(train_labels.max().item() + 1)
    logging.info(f"Number of classes: {num_classes}")
    results = {}
    for k in k_list:
        logging.info(f"Running k-NN prediction for k={k}")
        probs = knn_predict(train_feats, train_labels, val_feats, k, temperature, num_classes)
        top1 = (probs.argmax(1) == val_labels).float().mean().item()
        top5 = (
            probs.topk(min(5, num_classes), dim=1).indices.eq(val_labels.unsqueeze(1)).any(dim=1).float().mean().item()
        )
        results[k] = {"top1": top1, "top5": top5}
    logging.info("k-NN evaluation finished.")
    return results


@hydra.main(version_base="1.3", config_path="../fabric_configs/experiment/knn", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.info("Starting main function.")
    device = torch.device(cfg.device)
    logging.info(f"Using device: {device}")

    logging.info("Instantiating model...")
    model_cfg = instantiate(cfg.model, _convert_="all")
    model = model_cfg["net"] 
    model = model.to(device).eval()
    logging.info("Model instantiated and moved to device.")

    logging.info("Instantiating training dataset...")
    train_ds = instantiate(cfg.data.train, _convert_="all")
    logging.info("Training dataset instantiated.")
    logging.info("Instantiating validation dataset...")
    val_ds = instantiate(cfg.data.val, _convert_="all")
    logging.info("Validation dataset instantiated.")

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

    logging.info("Starting k-NN evaluation.")
    results = knn_eval(model, train_loader, val_loader, cfg.nb_knn, cfg.temperature, device)
    logging.info("k-NN evaluation completed.")
    for k, res in results.items():
        print(f"k={k}: top1={res['top1']*100:.2f}, top5={res['top5']*100:.2f}")
        logging.info(f"Results for k={k}: top1={res['top1']*100:.2f}, top5={res['top5']*100:.2f}")
    logging.info("Main function finished.")


if __name__ == "__main__":
    main()
