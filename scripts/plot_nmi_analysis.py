import torch
import torch.nn.functional as F
import numpy as np
import math
import os
from pathlib import Path
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import timm.data.transforms_factory as tff
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import hydra
import logging
from timm.utils import AverageMeter
from einops import reduce, repeat

from src.utils.analysis.clmim_hook import ActivationCache
from src.models.components.cl_vs_mim.utils import subsample

logging.basicConfig(level=logging.INFO)

# Configuration
DEVICE = "cuda"
IMG_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 1

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
    "partmae_v6_b": "REPR",
    "partmae_v5_b": r"REPR ($\mathcal{L}_{\rm pose}$ only)",
    "mae_b": "MAE", 
    "part_v0": "PART",
}

# Global model variable
model = None


def load_model(model_name: str):
    """Load a model from its configuration file."""
    global model
    logging.info(f"Loading model: {model_name}")
    assert model_name in MODELS, f"Model {model_name} not found in {MODELS}"
    model_conf = OmegaConf.load(CFG_FOLDER / f"{model_name}.yaml")
    model = (
        hydra.utils.instantiate({"model": model_conf}, _convert_="all")["model"]["net"]
        .eval()
        .to(DEVICE)
    )
    logging.info(f"Model {model_name} loaded.")
    return model


def setup_dataset(imagenet_path="~/development/datasets/imagenette2-160"):
    """Setup ImageNet dataset for testing."""
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])

    transform_test = tff.transforms_imagenet_eval(
        img_size=IMG_SIZE, mean=imagenet_mean, std=imagenet_std,
    )

    test_dir = os.path.join(imagenet_path, 'val')
    dataset_test = datasets.ImageFolder(test_dir, transform_test)
    dataset_test = torch.utils.data.Subset(
        dataset_test, 
        subsample(dataset_test, ratio=math.pow(2, -6))  # use a subsampled batch
    )

    dataset_test = DataLoader(
        dataset_test, 
        num_workers=NUM_WORKERS, 
        batch_size=BATCH_SIZE,
    )
    
    return dataset_test


def calculate_nmi(attn): 
    """Normalized mutual information with a return type of (batch, head)"""
    b, h, q, k = attn.shape
    pq = torch.ones([b, h, q]).to(attn.device)
    pq = F.softmax(pq, dim=-1)
    pq_ext = repeat(pq, "b h q -> b h q k", k=k)
    pk = reduce(attn * pq_ext, "b h q k -> b h k", "sum")
    pk_ext = repeat(pk, "b h k -> b h q k", q=q)
    
    mi = reduce(attn * pq_ext * torch.log(attn / pk_ext), "b h q k -> b h", "sum")
    eq = - reduce(pq * torch.log(pq), "b h q -> b h", "sum")
    ek = - reduce(pk * torch.log(pk), "b h k -> b h", "sum")
    
    nmiv = mi / torch.sqrt(eq * ek)
    
    return nmiv


def get_res_perfect_square(x):
    """Given x = a^2 + b, return b"""
    return x - int(x**0.5)**2


def calculate_nmi_for_model(model, dataset_test, label):
    """Calculate NMI for all layers of a model."""
    encoder_length = len(model.blocks)  # 12 for ViT-B
    nmis = [AverageMeter() for _ in range(encoder_length)]

    for idx, (xs, _) in enumerate(dataset_test):
        xs = xs.to(DEVICE)

        head_name = "clf" if label == "PARTv1_ft" else "head"
        act = ActivationCache(head_name)
        act.hook(model)
        
        with torch.no_grad():
            _ = model(xs)

        attns = act.get_attns() 
        attns = torch.stack(attns)

        b = get_res_perfect_square(attns.shape[-2])
        for i, attn in enumerate([attn for attn in attns if attn is not None]):
            attn = attn[:, :, b:, b:]  # drop cls token
            attn = attn + 1e-8
            attn = attn / attn.sum(dim=-1, keepdim=True)  # normalize
            attn = attn.cpu().float()
            
            nmi = calculate_nmi(attn)
            nmis[i].update(torch.mean(nmi, dim=0))  # average w.r.t. batch
            
        act.unhook()
        if idx > -1:  # Process only one batch for speed
            break
    
    return [torch.mean(nmi.avg) for nmi in nmis]


def plot_nmi_comparison(dataset_test, model_names, output_path="scripts/output/nmi_analysis.png"):
    """Create NMI comparison plot for multiple models."""
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Create figure
    fig, ax = plt.subplots(1, figsize=(10, 6), dpi=150)
    
    ax.set_xlabel("Depth")
    ax.set_ylabel("Normalized MI")
    
    # Plot for each model
    for model_name in model_names:
        model = load_model(model_name)
        nmis = calculate_nmi_for_model(model, dataset_test, model_name)
        display_name = NAMES.get(model_name, model_name)
        ax.plot(range(1, 13), nmis, marker="o", label=display_name)
        
    ax.set_ylim(top=0.6, bottom=0.0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logging.info(f"NMI analysis plot saved to {output_path}")
    plt.close()


def plot_all_models_nmi(dataset_test, model_names, output_path="scripts/output/nmi_all_models.png"):
    """Create NMI plot with all models in the same plot."""
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Create figure
    fig, ax = plt.subplots(1, figsize=(12, 8), dpi=150)
    
    # Plot for each model
    for model_name in model_names:
        model = load_model(model_name)
        nmis = calculate_nmi_for_model(model, dataset_test, model_name)
        display_name = NAMES.get(model_name, model_name)
        ax.plot(range(1, 13), nmis, marker="o", label=display_name, linewidth=2, markersize=6)
    
    ax.set_xlabel("Depth", fontsize=14)
    ax.set_ylabel("Normalized MI", fontsize=14)
    ax.set_title("Normalized Mutual Information - All Models", fontsize=16)
    ax.set_ylim(top=0.6, bottom=0.0)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logging.info(f"All models NMI plot saved to {output_path}")
    plt.close()


def main():
    """Main function to generate NMI analysis plots."""
    # Setup dataset
    logging.info("Setting up dataset...")
    dataset_test = setup_dataset()
    
    # Create output directory if it doesn't exist
    Path("scripts/output").mkdir(exist_ok=True)
    
    # Generate comparison plot for all models (legacy format)
    logging.info("Generating NMI comparison plot...")
    plot_nmi_comparison(dataset_test, MODELS)
    
    # Generate single plot with all models
    logging.info("Generating combined NMI plot with all models...")
    plot_all_models_nmi(dataset_test, MODELS)
    
    logging.info("NMI analysis completed!")


if __name__ == "__main__":
    main()
