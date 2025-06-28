import torch
import numpy as np
import math
import os
from pathlib import Path
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import timm.data.transforms_factory as tff
from torch.utils.data import DataLoader
from timm.models.vision_transformer import VisionTransformer
from omegaconf import OmegaConf
import hydra
import logging
from timm.utils import AverageMeter
from einops import rearrange

from src.utils.analysis.clmim_hook import ActivationCache
from src.models.components.cl_vs_mim.utils import subsample

logging.basicConfig(level=logging.INFO)

# Configuration
DEVICE = "cuda"
IMG_SIZE = 224
BATCH_SIZE = 128
NUM_WORKERS = 8
# DATA_SUBSAMPLE_RATIO = math.pow(2, -6)
DATA_SUBSAMPLE_RATIO = math.pow(2, -4)

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

# Global model variable (timm's VisionTransformer)
model: VisionTransformer = None


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
        subsample(dataset_test, ratio=DATA_SUBSAMPLE_RATIO)  # use a subsampled batch
    )

    dataset_test = DataLoader(
        dataset_test, 
        num_workers=NUM_WORKERS, 
        batch_size=BATCH_SIZE,
    )
    
    return dataset_test


def fourier(x):
    """2D Fourier transform"""
    f = torch.fft.fft2(x)
    f = f.abs() + 1e-6
    f = f.log()
    return f


def shift(x):  
    """Shift Fourier transformed feature map"""
    b, c, h, w = x.shape
    return torch.roll(x, shifts=(int(h/2), int(w/2)), dims=(2, 3))


def get_fourier_latents(latents):
    """Fourier transform feature maps"""
    fourier_latents = []
    for latent in latents:  # `latents` is a list of hidden feature maps in latent spaces
        latent = latent.cpu()
        b, n, c = latent.shape
        h, w = int(math.sqrt(n)), int(math.sqrt(n))
        latent = rearrange(latent, "b (h w) c -> b c h w", h=h, w=w)
        
        latent = fourier(latent)
        latent = shift(latent).mean(dim=(0, 1))
        latent = latent.diag()[int(h/2):]  # only use the half-diagonal components
        latent = latent - latent[0]  # visualize 'relative' log amplitudes 
                                     # (i.e., low-freq amp - high freq amp)
        fourier_latents.append(latent)

    return fourier_latents


def calculate_fourier_analysis(model, dataset_test, label):
    """Calculate Fourier analysis for all layers of a model."""
    head_name = "clf" if label == "PARTv1_ft" else "head"
    act = ActivationCache(head_name)
    act.hook(model)
    fourier_latents = AverageMeter()
    
    for i, (xs, ys) in enumerate(dataset_test):
        with torch.no_grad():
            xs = xs.to(DEVICE)
            _ = model(xs)

        zs = act.get_zs()
        zs = zs[:-1]  # skips model output

        # if model has cls token remove it
        if hasattr(model, 'cls_token') and model.cls_token is not None:
            latents = [z[:, 1:, :] for z in zs]
        else:
            latents = [z for z in zs]
        
        _fourier_latents = torch.stack(get_fourier_latents(latents))
        fourier_latents.update(_fourier_latents)

        if i > -1:  # Process only one batch for speed
            break

    act.unhook()
    return fourier_latents.avg


def plot_fourier_analysis_comparison(dataset_test, model_names, output_path="scripts/output/fourier_analysis.png"):
    """Create Fourier analysis comparison plot for multiple models."""
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
    
    for model_name in model_names:
        model = load_model(model_name)
        fourier_latents = calculate_fourier_analysis(model, dataset_test, model_name)
        display_name = NAMES.get(model_name, model_name)
        
        # Plot the last component of every other layer (matching original behavior)
        ax.plot(range(13), fourier_latents[:, -1][::2], marker="o", label=display_name)

    ax.set_xlabel("Depth")
    ax.set_ylabel("$\\Delta$ Log amplitude")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logging.info(f"Fourier analysis comparison plot saved to {output_path}")
    plt.close()


def plot_all_models_fourier_analysis(dataset_test, model_names, output_path="scripts/output/fourier_analysis_all_models.png"):
    """Create Fourier analysis plot with all models in the same plot."""
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Create figure
    fig, ax = plt.subplots(1, figsize=(12, 8), dpi=150)
    
    # Plot for each model
    for model_name in model_names:
        model = load_model(model_name)
        fourier_latents = calculate_fourier_analysis(model, dataset_test, model_name)
        display_name = NAMES.get(model_name, model_name)
        
        # Plot the last component of every other layer (matching original behavior)
        ax.plot(range(13), fourier_latents[:, -1][::2], marker="o", label=display_name, 
                linewidth=2, markersize=6)
    
    ax.set_xlabel("Depth", fontsize=14)
    ax.set_ylabel("$\\Delta$ Log amplitude", fontsize=14)
    ax.set_title("Fourier Analysis - All Models", fontsize=16)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logging.info(f"All models Fourier analysis plot saved to {output_path}")
    plt.close()


def plot_fourier_heatmap(dataset_test, model_names, output_path="scripts/output/fourier_heatmap.png"):
    """Create a heatmap showing Fourier components across models and depths."""
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Collect data for all models
    all_fourier_data = []
    model_labels = []
    
    for model_name in model_names:
        model = load_model(model_name)
        fourier_latents = calculate_fourier_analysis(model, dataset_test, model_name)
        display_name = NAMES.get(model_name, model_name)
        
        # Take every other layer and just the last frequency component (matching line plots)
        fourier_row = fourier_latents[:, -1][::2]  # Shape: (7,) for 7 layers (every other)
        all_fourier_data.append(fourier_row)
        model_labels.append(display_name)
    
    # Convert to numpy array for plotting
    fourier_matrix = torch.stack(all_fourier_data).numpy()  # Shape: (n_models, 7)
    
    # Create heatmap
    fig, ax = plt.subplots(1, figsize=(10, 8), dpi=150)
    
    im = ax.imshow(fourier_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    
    # Set labels
    ax.set_xlabel("Depth", fontsize=14)
    ax.set_ylabel("Model", fontsize=14)
    ax.set_title("Fourier Analysis Heatmap - Delta Log Amplitude by Model and Depth", fontsize=16)
    
    # Set ticks
    ax.set_yticks(range(len(model_labels)))
    ax.set_yticklabels(model_labels)
    ax.set_xticks(range(fourier_matrix.shape[1]))
    ax.set_xticklabels([f"{i}" for i in range(0, fourier_matrix.shape[1] * 2, 2)])  # 0, 2, 4, 6, 8, 10, 12
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('$\\Delta$ Log amplitude', fontsize=12)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logging.info(f"Fourier analysis heatmap saved to {output_path}")
    plt.close()


def plot_fourier_heatmap_full(dataset_test, model_names, output_path="scripts/output/fourier_heatmap_full.png"):
    """Create a comprehensive heatmap showing all Fourier components across all layers for each model."""
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(len(model_names), 1, figsize=(16, 3 * len(model_names)), dpi=150)
    if len(model_names) == 1:
        axes = [axes]
    
    for idx, model_name in enumerate(model_names):
        model = load_model(model_name)
        fourier_latents = calculate_fourier_analysis(model, dataset_test, model_name)
        display_name = NAMES.get(model_name, model_name)
        
        # Use all layers and all frequency components
        fourier_matrix = fourier_latents.numpy()  # Shape: (layers, frequency_components)
        
        # Create heatmap for this model
        im = axes[idx].imshow(fourier_matrix.T, aspect='auto', cmap='viridis', interpolation='nearest')
        
        # Set labels
        axes[idx].set_xlabel("Layer", fontsize=12)
        axes[idx].set_ylabel("Frequency Component", fontsize=12)
        axes[idx].set_title(f"Fourier Analysis - {display_name}", fontsize=14)
        
        # Set ticks
        axes[idx].set_xticks(range(0, fourier_matrix.shape[0], 2))
        axes[idx].set_xticklabels([f"{i}" for i in range(0, fourier_matrix.shape[0], 2)])
        
        # Add colorbar for each subplot
        cbar = plt.colorbar(im, ax=axes[idx])
        cbar.set_label('$\\Delta$ Log amplitude', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logging.info(f"Full Fourier analysis heatmap saved to {output_path}")
    plt.close()


def main():
    """Main function to generate Fourier analysis plots."""
    # Setup dataset
    logging.info("Setting up dataset...")
    dataset_test = setup_dataset("/mnt/sdb1/datasets/imagenette2")
    
    # Create output directory if it doesn't exist
    Path("scripts/output").mkdir(exist_ok=True)
    
    # Generate comparison plot for all models (legacy format)
    logging.info("Generating Fourier analysis comparison plot...")
    plot_fourier_analysis_comparison(dataset_test, MODELS)
    
    # Generate single plot with all models
    logging.info("Generating combined Fourier analysis plot with all models...")
    plot_all_models_fourier_analysis(dataset_test, MODELS)
    
    # Generate heatmap visualization
    logging.info("Generating Fourier analysis heatmap...")
    plot_fourier_heatmap(dataset_test, MODELS)
    
    # Optionally generate full heatmap (commented out by default due to size)
    # logging.info("Generating full Fourier analysis heatmap...")
    # plot_fourier_heatmap_full(dataset_test, MODELS)
    
    logging.info("Fourier analysis completed!")


if __name__ == "__main__":
    main()
