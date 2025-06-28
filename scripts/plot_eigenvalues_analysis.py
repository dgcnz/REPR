import os
import logging
from pathlib import Path
from collections import defaultdict

import torch
import torchvision.datasets as datasets
import timm.data.transforms_factory as tff
import einops
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import hydra

# ────── Logging Setup ──────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ────── Torch/CUDA Tuning ──────
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

# ────── Configuration ──────
DEVICE = "cuda"
IMG_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = min(8, os.cpu_count() or 0)
SUBSAMPLE_RATIO = 2 ** -3
CFG_FOLDER = Path("fabric_configs/experiment/hummingbird/model")
MODELS = [
    "dino_b",
    "cim_b", 
    "droppos_b",
    "partmae_v5_b",
    "partmae_v6_b",
    "mae_b",
    "part_v0",
]

NAMES = {
    "dino_b": "DINO",
    "cim_b": "CIM",
    "droppos_b": "DropPos",
    "partmae_v6_b": "Ours",
    "partmae_v5_b": r"Ours ($\mathcal{L}_{\rm pose}$ only)",
    "mae_b": "MAE", 
    "part_v0": "PART",
}


def subsample(dataset, ratio, random=False):
    logger.debug(f"Subsampling dataset at ratio={ratio}")
    idxs_per_cls = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        idxs_per_cls[label].append(idx)
    picks = []
    for cls, idxs in idxs_per_cls.items():
        k = int(len(idxs) * ratio)
        picks.extend(torch.randperm(len(idxs))[:k].tolist() if random else idxs[:k])
    logger.info(f"  Subsampled {len(picks)}/{len(dataset)} samples")
    return picks


def setup_dataset(path="~/development/datasets/imagenette2-160"):
    logger.info("Setting up dataset…")
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tf = tff.transforms_imagenet_eval(img_size=IMG_SIZE, mean=mean, std=std)
    val_dir = os.path.expanduser(os.path.join(path, "val"))
    ds = datasets.ImageFolder(val_dir, tf)
    logger.info(f"  Original size: {len(ds)}")
    subset = subsample(ds, SUBSAMPLE_RATIO)
    ds = torch.utils.data.Subset(ds, subset)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(NUM_WORKERS > 0),
    )
    logger.info(f"  DataLoader → {len(loader)} batches of size {BATCH_SIZE}")
    return loader


def load_model(name):
    logger.info(f"Loading model '{name}'…")
    cfg = OmegaConf.load(CFG_FOLDER / f"{name}.yaml")
    model = hydra.utils.instantiate({"model": cfg}, _convert_="all")["model"]["net"]
    model = model.eval().to(DEVICE)
    for blk in model.blocks:
        blk.attn.fused_attn = False
    logger.info("  Model ready")
    return model


def clean_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.detach()
    if isinstance(x, (list, tuple)):
        return type(x)(clean_tensor(y) for y in x)
    if isinstance(x, dict):
        return {k: clean_tensor(v) for k, v in x.items()}
    return x


def get_layer(model, path):
    o = model
    for p in path.split("."):
        o = o[int(p)] if p.isdigit() else getattr(o, p)
    return o


class ActivationCache:
    def __init__(self, head="head"):
        self.head = head
        self.cache = {}
        self.hooks = {}

    def _hook(self, name):
        def fn(module, inp, out):
            self.cache[name] = {"in": clean_tensor(inp), "out": clean_tensor(out)}
        return fn

    def hook(self, model):
        logger.info("Registering hooks…")
        names = []
        for i in range(len(model.blocks)):
            names += [f"blocks.{i}", f"blocks.{i}.norm2", f"blocks.{i}.mlp"]
        names.append(self.head)
        for n in names:
            layer = get_layer(model, n)
            self.hooks[n] = layer.register_forward_hook(self._hook(n))
        logger.info(f"  Installed {len(self.hooks)} hooks")

    def clear(self):
        self.cache.clear()

    def unhook(self):
        for h in self.hooks.values():
            h.remove()
        self.hooks.clear()
        logger.info("All hooks removed")

    def get_zs(self):
        """
        Returns:
          [blocks.0 input,
           blocks.0 norm2-in, blocks.0 post-MLP,
           blocks.1 norm2-in, blocks.1 post-MLP,
           …,
           head output]
        """
        zs = [self.cache["blocks.0"]["in"][0]]
        num = len([k for k in self.cache if k.startswith("blocks.") and k.endswith("mlp")])
        for i in range(num):
            z1 = self.cache[f"blocks.{i}.norm2"]["in"][0]
            mlp = self.cache[f"blocks.{i}.mlp"]["out"]
            zs += [z1, z1 + mlp]
        zs.append(self.cache[self.head]["out"])
        return zs


def batch_cov(points):
    # points: (B, D, N)
    B, D, N = points.shape
    mean = points.mean(dim=2, keepdim=True)
    diffs = points - mean
    return torch.bmm(diffs, diffs.transpose(1, 2)) / (N - 1)


class ActivationStats:
    def __init__(self, model):
        self.n = len(model.blocks)
        D = model.blocks[0].attn.proj.weight.shape[0]
        logger.info(f"Init stats: layers={self.n}, dim={D}")

        # image-level
        self.S_img = [torch.zeros(D, D, device=DEVICE) for _ in range(self.n)]
        self.m_img = [torch.zeros(D, device=DEVICE) for _ in range(self.n)]
        self.N_img = [0] * self.n

        # CLS-level
        self.S_cls = [torch.zeros(D, D, device=DEVICE) for _ in range(self.n)]
        self.m_cls = [torch.zeros(D, device=DEVICE) for _ in range(self.n)]
        self.N_cls = [0] * self.n

        # token-level scatter
        self.S_tok = [torch.zeros(D, D, device=DEVICE) for _ in range(self.n)]
        self.m_tok = [torch.zeros(D, device=DEVICE) for _ in range(self.n)]
        self.N_tok = [0] * self.n

    def update(self, zs):
        # post-MLP for block i is at zs[2*i + 2]
        for i in range(self.n):
            z2 = zs[2 * i + 2]      # (B, N, D)
            B, N, D = z2.shape

            # image-level
            img = z2[:, 1:, :].mean(dim=1)  # (B, D)
            self.S_img[i] += img.T @ img
            self.m_img[i] += img.sum(dim=0)
            self.N_img[i] += B

            # CLS-level
            cls = z2[:, 0, :]           # (B, D)
            self.S_cls[i] += cls.T @ cls
            self.m_cls[i] += cls.sum(dim=0)
            self.N_cls[i] += B

            # token-level scatter
            spatial = z2[:, 1:, :].reshape(B * (N - 1), D)
            self.S_tok[i] += spatial.T @ spatial
            self.m_tok[i] += spatial.sum(dim=0)
            self.N_tok[i] += B * (N - 1)

        logger.debug(f" Updated stats: N_img={self.N_img}, N_tok={self.N_tok}")

    def finalize(self):
        logger.info("Finalizing stats…")
        out = {"image": [], "cls": [], "token": []}

        for i in range(self.n):
            # image-level covariance SVD
            Ni = self.N_img[i]
            cov_i = (
                self.S_img[i]
                - (self.m_img[i].unsqueeze(1) @ self.m_img[i].unsqueeze(0)) / Ni
            ) / (Ni - 1)
            s_img = torch.linalg.svdvals(cov_i).log().cpu()
            out["image"].append(s_img)

            # CLS-level
            Nc = self.N_cls[i]
            cov_c = (
                self.S_cls[i]
                - (self.m_cls[i].unsqueeze(1) @ self.m_cls[i].unsqueeze(0)) / Nc
            ) / (Nc - 1)
            s_cls = torch.linalg.svdvals(cov_c).log().cpu()
            out["cls"].append(s_cls)

            # token-level
            Nt = self.N_tok[i]
            cov_t = (
                self.S_tok[i]
                - (self.m_tok[i].unsqueeze(1) @ self.m_tok[i].unsqueeze(0)) / Nt
            ) / (Nt - 1)
            s_tok = torch.linalg.svdvals(cov_t).log().cpu()
            out["token"].append(s_tok)

            logger.info(
                f"Layer {i}: image_N={Ni}, cls_N={Nc}, token_N={Nt}, "
                f"eig_shapes={[len(out[k][-1]) for k in out]}"
            )

        return out


def extract_and_compute_singular_values(model, loader, label):
    logger.info(f"Extract+SVD for '{label}'")
    head = "clf" if label == "PARTv1_ft" else "head"
    cache = ActivationCache(head)
    stats = ActivationStats(model)

    cache.hook(model)
    total = len(loader)
    with torch.no_grad():
        for idx, (xs, _) in enumerate(loader, 1):
            xs = xs.to(DEVICE, non_blocking=True)
            _ = model(xs)
            zs = cache.get_zs()
            stats.update(zs)
            cache.clear()
            if idx % 10 == 0 or idx == total:
                logger.info(f"  Batch {idx}/{total}")
    cache.unhook()

    return stats.finalize()


def plot_svd(data, kind):
    import matplotlib.pyplot as plt

    out = Path("scripts/output/svd")
    out.mkdir(parents=True, exist_ok=True)
    logger.info(f"Plotting {kind}-level SVD")

    # first-component
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for name, svs in data.items():
        first = torch.stack([s[0] for s in svs]).cpu()
        ax.plot(range(len(first)), first, marker="o", label=NAMES[name])
    ax.set(xlabel="Depth", ylabel="Log SV (1st)", title=f"{kind.title()}-level SVD")
    ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(out / f"{kind}_analysis.png", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved → {kind}_analysis.png")

    # full-spectrum
    maxr = {"image": 49, "cls": 384, "token": 196}[kind]
    nm = len(data)
    cols = min(nm, 3); rows = (nm + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), dpi=150)
    # add title
    fig.suptitle(f"{kind.title()}-level SVD Spectrum", fontsize=16, fontweight='bold')
    axes = axes.flatten() if nm > 1 else [axes]
    cmap = plt.cm.viridis
    norm = plt.Normalize(0, len(next(iter(data.values()))) - 1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for ax, (name, svs) in zip(axes, data.items()):
        for d, sv in enumerate(svs):
            ax.plot(
                range(min(len(sv), maxr)),
                sv[:maxr].cpu(),
                color=cmap(norm(d)),
                alpha=0.8
            )
        ax.set(title=NAMES[name], xlabel="Rank", ylabel="Log SV")
        plt.colorbar(sm, ax=ax, label="Depth")
        ax.grid(alpha=0.3)

    for ax in axes[nm:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out / f"{kind}_spectrum.png", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved → {kind}_spectrum.png")


def main():
    logger.info("=== STARTING ANALYSIS ===")
    loader = setup_dataset()
    results = {"image": {}, "cls": {}, "token": {}}

    for m in MODELS:
        mdl = load_model(m)
        sv = extract_and_compute_singular_values(mdl, loader, m)
        for k in sv:
            results[k][m] = sv[k]
        del mdl
        torch.cuda.empty_cache()
        logger.info(f"Cleared cache after '{m}'")

    for k in ["image", "cls", "token"]:
        plot_svd(results[k], k)

    logger.info("=== FINISHED ===")


if __name__ == "__main__":
    main()
