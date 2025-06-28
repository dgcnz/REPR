import torch
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
import einops

from src.utils.analysis.clmim_hook import ActivationCache
from src.models.components.cl_vs_mim.utils import subsample

logging.basicConfig(level=logging.INFO)

# set high precision for matmul 32
torch.set_float32_matmul_precision('high')

# Configuration
DEVICE = "cuda"
IMG_SIZE = 224
BATCH_SIZE = 768
NUM_WORKERS = 8

# Spectrum plotting limits
MAX_RANK_TOKEN = 196  # Token-level spectrum collapses around 196
MAX_RANK_IMAGE = 49   # Image-level spectrum collapses around 50
MAX_RANK_CLS = 768     # CLS-level spectrum likely similar to image-level
SUBSAMPLE_RATIO = math.pow(2, -1)

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
        subsample(dataset_test, ratio=SUBSAMPLE_RATIO)  # use a subsampled batch
    )
    print(f"SUBSAMPLED {SUBSAMPLE_RATIO} ratio, using {len(dataset_test)} samples")

    dataset_test = DataLoader(
        dataset_test, 
        num_workers=NUM_WORKERS, 
        batch_size=BATCH_SIZE,
        drop_last=True,  
    )
    
    return dataset_test


def get_imglvl_sings(latents):
    """Get image-level singular values.
    
    Args:
        latents: List of tensors with shape (batch, n, c)
    
    Returns:
        List of singular values for each layer
    """
    latents = [latent[:, 1:, :] for latent in latents]  # drop CLS
    latents = [einops.reduce(latent, "b n c -> b c", "mean") for latent in latents]  # img lvl reprs
    
    sinvals = []
    for i, latent in enumerate(latents):
        latent = einops.rearrange(latent, "b c -> c b")
        cov = torch.cov(latent)
        u, s, v = torch.svd(cov)
        s = s.log()
        sinvals.append(s)
        
    return sinvals


def get_tknlvl_sings(latents):
    """Get token-level singular values.
    
    Args:
        latents: List of tensors with shape (batch, n, c)
    
    Returns:
        Tensor of averaged singular values with shape (depth, c)
    """
    latents = [latent[:, 1:, :] for latent in latents]  # drop CLS
    sinvals = [[] for _ in range(len(latents))]

    for i, latent in enumerate(latents):
        for j, _latent in enumerate(latent):
            _latent = einops.rearrange(_latent, "n c -> c n")  # tkn lvl reprs
            cov = torch.cov(_latent)
            u, s, v = torch.svd(cov)
            s = s.log()
            sinvals[i].append(s)
    
    # Use einops reduce as in the original pseudocode
    sinvals = einops.reduce(sinvals, "d m c -> d c", "mean")
    return sinvals


def extract_model_features(model, dataset_test, label):
    """Extract features from all layers of a model once.
    
    Args:
        model: The model to analyze
        dataset_test: Dataset for analysis
        label: Model label
    
    Returns:
        List of tensors, each with shape (total_batch, n_tokens, features)
    """
    head_name = "clf" if label == "PARTv1_ft" else "head"
    act = ActivationCache(head_name)
    act.hook(model)
    all_batch_features = []
    
    for i, (xs, ys) in enumerate(dataset_test):
        with torch.no_grad():
            xs = xs.to(DEVICE)
            _ = model(xs)

        zs = act.get_zs()
        zs = zs[:-1]  # skips model output
        
        # Each element in zs has shape (batch, n_tokens, features)
        all_batch_features.append(zs)
        
        if i >= 0:  # Process only one batch for speed (change to > 0 to process more)
            break

    act.unhook()
    
    # Concatenate all batches across the batch dimension for each layer
    # all_batch_features is a list of lists: [[layer0_batch0, layer1_batch0, ...], [layer0_batch1, layer1_batch1, ...], ...]
    if len(all_batch_features) == 1:
        # Only one batch, return as is
        return all_batch_features[0]
    else:
        # Multiple batches, concatenate across batch dimension
        n_layers = len(all_batch_features[0])
        concatenated_features = []
        for layer_idx in range(n_layers):
            layer_batches = [batch[layer_idx] for batch in all_batch_features]
            concatenated_layer = torch.cat(layer_batches, dim=0)  # Concatenate along batch dimension
            concatenated_features.append(concatenated_layer)
        return concatenated_features


def compute_singular_values_from_features(features, analysis_type="image"):
    """Compute singular values from pre-extracted features.
    
    Args:
        features: List of tensors, each with shape (batch, n_tokens, features)
        analysis_type: Either "image", "token", or "cls" level analysis
    
    Returns:
        Singular values tensor
    """
    if analysis_type == "image":
        return get_imglvl_sings(features)
    elif analysis_type == "token":
        return get_tknlvl_sings(features)
    elif analysis_type == "cls":
        return get_clslvl_sings(features)
    else:
        raise ValueError(f"Unknown analysis_type: {analysis_type}. Must be 'image', 'token', or 'cls'")


def get_clslvl_sings(latents):
    """Get CLS-level singular values.
    
    Args:
        latents: List of tensors with shape (batch, n_tokens, c)
    
    Returns:
        List of singular values for each layer using CLS token only
    """
    # Use only CLS token (first token) as image-level representation
    cls_latents = [latent[:, 0, :] for latent in latents]  # Extract CLS token: (batch, c)
    
    sinvals = []
    for i, latent in enumerate(cls_latents):
        latent = einops.rearrange(latent, "b c -> c b")
        cov = torch.cov(latent)
        u, s, v = torch.svd(cov)
        s = s.log()
        sinvals.append(s)
        
    return sinvals


def plot_singular_values_comparison(dataset_test, model_names, analysis_type="image", 
                                  output_path=None):
    """Create singular values comparison plot for multiple models."""
    if output_path is None:
        output_path = f"scripts/output/svd/{analysis_type}_analysis.png"
    
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)

    for model_name in model_names:
        model = load_model(model_name)
        # Extract features once
        features = extract_model_features(model, dataset_test, model_name)
        # Compute singular values from features
        singular_values = compute_singular_values_from_features(features, analysis_type)
        display_name = NAMES.get(model_name, model_name)
        
        # Plot the first singular value for each layer
        if analysis_type in ["image", "cls"]:
            # singular_values is a list of tensors
            first_singular_per_layer = torch.stack([sv[0] for sv in singular_values])
            ax.plot(range(len(first_singular_per_layer)), first_singular_per_layer.cpu(), 
                   marker="o", label=display_name)
        else:  # token level
            # singular_values shape: (depth, c)
            ax.plot(range(len(singular_values)), singular_values[:, 0].cpu(), 
                   marker="o", label=display_name)

    ax.set_xlabel("Depth")
    ax.set_ylabel("Log Singular Value (First Component)")
    ax.set_title(f"Singular Value Analysis - {analysis_type.title()} Level")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logging.info(f"Singular values {analysis_type} analysis plot saved to {output_path}")
    plt.close()


def plot_all_models_singular_values(dataset_test, model_names, analysis_type="image", 
                                   output_path=None):
    """Create singular values plot with all models in the same plot."""
    if output_path is None:
        output_path = f"scripts/output/svd/{analysis_type}_all_models.png"
    
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Create figure
    fig, ax = plt.subplots(1, figsize=(12, 8), dpi=150)
    
    # Plot for each model
    for model_name in model_names:
        model = load_model(model_name)
        # Extract features once
        features = extract_model_features(model, dataset_test, model_name)
        # Compute singular values from features
        singular_values = compute_singular_values_from_features(features, analysis_type)
        display_name = NAMES.get(model_name, model_name)
        
        # Plot the first singular value for each layer
        if analysis_type in ["image", "cls"]:
            # singular_values is a list of tensors
            first_singular_per_layer = torch.stack([sv[0] for sv in singular_values])
            ax.plot(range(len(first_singular_per_layer)), first_singular_per_layer.cpu(), 
                   marker="o", label=display_name, linewidth=2, markersize=6)
        else:  # token level
            # singular_values shape: (depth, c)
            ax.plot(range(len(singular_values)), singular_values[:, 0].cpu(), 
                   marker="o", label=display_name, linewidth=2, markersize=6)
    
    ax.set_xlabel("Depth", fontsize=14)
    ax.set_ylabel("Log Singular Value (First Component)", fontsize=14)
    ax.set_title(f"Singular Value Analysis - {analysis_type.title()} Level - All Models", fontsize=16)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logging.info(f"All models singular values {analysis_type} analysis plot saved to {output_path}")
    plt.close()


def plot_singular_values_heatmap(dataset_test, model_names, analysis_type="image", 
                                output_path=None):
    """Create a heatmap showing singular values across models and depths."""
    if output_path is None:
        output_path = f"scripts/output/svd/{analysis_type}_heatmap.png"
    
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Collect data for all models
    all_singular_data = []
    model_labels = []
    
    for model_name in model_names:
        model = load_model(model_name)
        # Extract features once
        features = extract_model_features(model, dataset_test, model_name)
        # Compute singular values from features
        singular_values = compute_singular_values_from_features(features, analysis_type)
        display_name = NAMES.get(model_name, model_name)
        
        # Use first singular value for each layer
        if analysis_type in ["image", "cls"]:
            # singular_values is a list of tensors
            first_singular_per_layer = torch.stack([sv[0] for sv in singular_values])
            singular_row = first_singular_per_layer.cpu()  # Shape: (depth,)
        else:  # token level
            # singular_values shape: (depth, c)
            singular_row = singular_values[:, 0].cpu()  # Shape: (depth,)
        
        all_singular_data.append(singular_row)
        model_labels.append(display_name)
    
    # Convert to numpy array for plotting
    singular_matrix = torch.stack(all_singular_data).numpy()  # Shape: (n_models, depth)
    
    # Create heatmap
    fig, ax = plt.subplots(1, figsize=(10, 8), dpi=150)
    
    im = ax.imshow(singular_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    
    # Set labels
    ax.set_xlabel("Depth", fontsize=14)
    ax.set_ylabel("Model", fontsize=14)
    ax.set_title(f"Singular Values Heatmap - {analysis_type.title()} Level (First Component)", fontsize=16)
    
    # Set ticks
    ax.set_yticks(range(len(model_labels)))
    ax.set_yticklabels(model_labels)
    ax.set_xticks(range(singular_matrix.shape[1]))
    ax.set_xticklabels([f"{i}" for i in range(singular_matrix.shape[1])])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Log Singular Value', fontsize=12)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logging.info(f"Singular values {analysis_type} heatmap saved to {output_path}")
    plt.close()


def plot_singular_value_spectrum(dataset_test, model_names, analysis_type="image", 
                                 output_path=None):
    """Generate singular value spectrum plots like in the attachment."""
    if output_path is None:
        output_path = f"scripts/output/svd/{analysis_type}_spectrum.png"
    
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Set max rank based on analysis type
    if analysis_type == "image":
        max_rank = MAX_RANK_IMAGE
    elif analysis_type == "token":
        max_rank = MAX_RANK_TOKEN
    elif analysis_type == "cls":
        max_rank = MAX_RANK_CLS
    else:
        max_rank = 100  # fallback
    
    # Calculate number of subplots needed
    n_models = len(model_names)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), dpi=150)
    if n_models == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    # Create colormap for depth
    cmap = plt.cm.get_cmap('viridis')
    
    # Plot for each model
    for i, model_name in enumerate(model_names):
        model = load_model(model_name)
        # Extract features once
        features = extract_model_features(model, dataset_test, model_name)
        # Compute singular values from features
        singular_values = compute_singular_values_from_features(features, analysis_type)
        display_name = NAMES.get(model_name, model_name)
        
        ax = axes[i] if n_models > 1 else axes[0]
        
        # Plot full spectrum for each layer (depth) - use every other layer for ViT blocks
        if analysis_type in ["image", "cls"]:
            # singular_values is a list of tensors
            # Use every other layer to match ViT block outputs (skip intermediate activations)
            layer_indices = range(0, len(singular_values), 2)
            n_plot_layers = len(layer_indices)
            for i, layer_idx in enumerate(layer_indices):
                color = cmap(i / (n_plot_layers - 1))
                layer_values = singular_values[layer_idx].cpu()
                
                # Plot only up to max_rank
                rank_indices = range(min(len(layer_values), max_rank))
                ax.plot(rank_indices, layer_values[:max_rank], color=color, alpha=0.8, linewidth=1.5)
        else:  # token level
            # singular_values shape: (depth, c)
            # Use every other layer to match ViT block outputs (skip intermediate activations)
            layer_indices = range(0, len(singular_values), 2)
            n_plot_layers = len(layer_indices)
            for i, layer_idx in enumerate(layer_indices):
                color = cmap(i / (n_plot_layers - 1))
                layer_values = singular_values[layer_idx].cpu()
                
                # Plot only up to max_rank
                rank_indices = range(min(len(layer_values), max_rank))
                ax.plot(rank_indices, layer_values[:max_rank], color=color, alpha=0.8, linewidth=1.5)
        
        ax.set_xlabel("Rank index")
        ax.set_ylabel("Δ Log singular value")
        ax.set_title(f"{display_name}")
        ax.grid(True, alpha=0.3)
        
        # Add colorbar for depth - use actual plotted layers count
        if analysis_type in ["image", "cls"]:
            n_plot_layers = len(range(0, len(singular_values), 2))
        else:
            n_plot_layers = len(range(0, len(singular_values), 2))
        norm = plt.Normalize(vmin=0, vmax=n_plot_layers-1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Depth (ViT blocks)')

    # Hide unused subplots
    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logging.info(f"Singular values {analysis_type} spectrum plot saved to {output_path} (max rank: {max_rank})")
    plt.close()


def plot_singular_values_from_cache(model_singular_values, analysis_type):
    """Generate plots from pre-computed singular values."""
    
    # Main analysis plot
    output_path = f"scripts/output/svd/{analysis_type}_analysis.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
    
    for model_name, singular_values in model_singular_values.items():
        display_name = NAMES.get(model_name, model_name)
        
        if analysis_type in ["image", "cls"]:
            # singular_values is a list of tensors
            first_singular_per_layer = torch.stack([sv[0] for sv in singular_values])
            # Use every other layer to match ViT block outputs (skip intermediate activations)
            ax.plot(range(len(first_singular_per_layer[::2])), first_singular_per_layer[::2].cpu(), 
                   marker="o", label=display_name)
        else:  # token level
            # singular_values shape: (depth, c)  
            # Use every other layer to match ViT block outputs (skip intermediate activations)
            ax.plot(range(len(singular_values[::2])), singular_values[::2, 0].cpu(), 
                   marker="o", label=display_name)

    ax.set_xlabel("Depth")
    ax.set_ylabel("Log Singular Value (First Component)")
    ax.set_title(f"Singular Value Analysis - {analysis_type.title()} Level")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logging.info(f"Singular values {analysis_type} analysis plot saved to {output_path}")
    plt.close()
    
    # Spectrum plot
    output_path = f"scripts/output/svd/{analysis_type}_spectrum.png"
    n_models = len(model_singular_values)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    # Set max rank based on analysis type
    if analysis_type == "image":
        max_rank = MAX_RANK_IMAGE
    elif analysis_type == "token":
        max_rank = MAX_RANK_TOKEN
    elif analysis_type == "cls":
        max_rank = MAX_RANK_CLS
    else:
        max_rank = 100  # fallback
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), dpi=150)
    if n_models == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    cmap = plt.cm.get_cmap('viridis')
    
    for i, (model_name, singular_values) in enumerate(model_singular_values.items()):
        display_name = NAMES.get(model_name, model_name)
        ax = axes[i] if n_models > 1 else axes[0]
        
        if analysis_type in ["image", "cls"]:
            n_layers = len(singular_values)
            for layer_idx in range(n_layers):
                color = cmap(layer_idx / (n_layers - 1))
                layer_values = singular_values[layer_idx].cpu()
                # Plot only up to max_rank
                rank_indices = range(min(len(layer_values), max_rank))
                ax.plot(rank_indices, layer_values[:max_rank], color=color, alpha=0.8, linewidth=1.5)
        else:
            n_layers = len(singular_values)
            for layer_idx in range(n_layers):
                color = cmap(layer_idx / (n_layers - 1))
                layer_values = singular_values[layer_idx].cpu()
                # Plot only up to max_rank
                rank_indices = range(min(len(layer_values), max_rank))
                ax.plot(rank_indices, layer_values[:max_rank], color=color, alpha=0.8, linewidth=1.5)
        
        ax.set_xlabel("Rank index")
        ax.set_ylabel("Δ Log singular value")
        ax.set_title(f"{display_name}")
        ax.grid(True, alpha=0.3)
        
        n_layers = len(singular_values)
        norm = plt.Normalize(vmin=0, vmax=n_layers-1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Depth')

    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logging.info(f"Singular values {analysis_type} spectrum plot saved to {output_path} (max rank: {max_rank})")
    plt.close()


def main():
    """Main function to generate singular value analysis plots efficiently."""
    # Setup dataset
    logging.info("Setting up dataset...")
    dataset_test = setup_dataset()
    
    # Create output directory if it doesn't exist
    Path("scripts/output").mkdir(exist_ok=True)
    
    # Extract features once for all models (this is the expensive part)
    logging.info("Extracting features from all models...")
    model_features = {}
    for model_name in MODELS:
        logging.info(f"Extracting features from {model_name}...")
        model = load_model(model_name)
        features = extract_model_features(model, dataset_test, model_name)
        model_features[model_name] = features
        logging.info(f"Features extracted for {model_name}: {len(features)} layers, each with shape {features[0].shape}")
    
    # Now compute all analysis types from the cached features
    logging.info("Computing singular value analyses from cached features...")
    model_singular_values = {}
    
    for analysis_type in ["image", "token", "cls"]:
        model_singular_values[analysis_type] = {}
        for model_name in MODELS:
            logging.info(f"Computing {analysis_type}-level analysis for {model_name}...")
            singular_values = compute_singular_values_from_features(
                model_features[model_name], analysis_type
            )
            model_singular_values[analysis_type][model_name] = singular_values
    
    # Generate all plots from the cached results
    logging.info("Generating plots from cached results...")
    for analysis_type in ["image", "token", "cls"]:
        logging.info(f"Generating {analysis_type}-level plots...")
        
        # Use the optimized plotting functions that work with pre-computed data
        plot_singular_values_from_cache(model_singular_values[analysis_type], analysis_type)
    
    logging.info("Singular value analysis completed efficiently!")


if __name__ == "__main__":
    main()
