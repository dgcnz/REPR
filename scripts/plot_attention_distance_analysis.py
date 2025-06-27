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
from timm.utils import AverageMeter

from src.utils.analysis.clmim_hook import ActivationCache
from src.models.components.cl_vs_mim.utils import subsample

logging.basicConfig(level=logging.INFO)

# Configuration
DEVICE = "cuda"
IMG_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 1
PATCH_SIZE = 16

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


def compute_distance_matrix(patch_size, num_patches, length):
    """Helper function to compute distance matrix."""
    distance_matrix = np.zeros((num_patches, num_patches))
    for i in range(num_patches):
        for j in range(num_patches):
            if i == j:  # zero distance
                continue

            xi, yi = (int(i/length)), (i % length)
            xj, yj = (int(j/length)), (j % length)
            distance_matrix[i, j] = patch_size * np.linalg.norm([xi - xj, yi - yj])

    return distance_matrix


def calculate_mean_attention_dist(patch_size, attention_weights):
    """The attention_weights shape = (batch, num_heads, num_patches, num_patches)"""
    
    num_patches = attention_weights.shape[-1]
    length = int(np.sqrt(num_patches))
    assert (length**2 == num_patches), ("Num patches is not perfect square")

    distance_matrix = compute_distance_matrix(patch_size, num_patches, length)
    h, w = distance_matrix.shape

    distance_matrix = distance_matrix.reshape((1, 1, h, w))
    # The attention_weights along the last axis adds to 1
    # this is due to the fact that they are softmax of the raw logits
    # summation of the (attention_weights * distance_matrix)
    # should result in an average distance per token
    mean_distances = attention_weights * distance_matrix
    mean_distances = np.sum(mean_distances, axis=-1)  # sum along last axis to get average distance per token
    mean_distances = np.mean(mean_distances, axis=-1)  # now average across all the tokens

    return torch.tensor(mean_distances)


def get_res_perfect_square(x):
    """Given x = a^2 + b, return b"""
    return x - int(x**0.5)**2


def calculate_distances(model, label: str, dataset_test):
    """Calculate attention distances for all layers of a model."""
    encoder_length = len(model.blocks)
    distances = [AverageMeter() for _ in range(encoder_length)]

    for idx, (xs, _) in enumerate(dataset_test):
        xs = xs.to(DEVICE)
        head_name = "clf" if label == "PARTv1_ft" else "head"
        act = ActivationCache(head_name)
        act.hook(model)
        
        with torch.no_grad():
            _ = model(xs)

        attns = act.get_attns()
        # stack attns
        attns = torch.stack(attns)
        
        b = get_res_perfect_square(attns.shape[-2])
            
        for i, attn in enumerate([attn for attn in attns if attn is not None]):    
            # remove the first k patches so that the distance matrix is square
            attn = attn[:, :, b:, b:]

            attn = attn + 1e-32
            attn = attn / attn.sum(dim=-1, keepdim=True)
            attn = attn.cpu().float().detach().numpy()

            distance = calculate_mean_attention_dist(patch_size=PATCH_SIZE, attention_weights=attn)
            distances[i].update(torch.mean(distance, dim=0))
            
        act.unhook()
        if idx > -1:  # Process only one batch for speed
            break
    
    return [torch.mean(distance.avg) for distance in distances]


def plot_attention_distance_comparison(dataset_test, model_names, output_path="scripts/output/attention_distance_analysis.png"):
    """Create attention distance comparison plot for multiple models."""
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
    x = range(1, 13)
    
    for model_name in model_names:
        model = load_model(model_name)
        # Calculate distances for all models
        distances = calculate_distances(model, model_name, dataset_test)
        display_name = NAMES.get(model_name, model_name)

        # Plot curve
        ax.plot(x, distances, label=display_name, marker='o')

    ax.set_xlabel("Depth")
    ax.set_ylabel("Attention distance (px)")
    ax.set_ylim(top=140, bottom=0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logging.info(f"Attention distance analysis plot saved to {output_path}")
    plt.close()


def plot_all_models_attention_distance(dataset_test, model_names, output_path="scripts/output/attention_distance_all_models.png"):
    """Create attention distance plot with all models in the same plot."""
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=150)
    x = range(1, 13)
    
    for model_name in model_names:
        model = load_model(model_name)
        # Calculate distances for all models
        distances = calculate_distances(model, model_name, dataset_test)
        display_name = NAMES.get(model_name, model_name)

        # Plot curve
        ax.plot(x, distances, label=display_name, marker='o', linewidth=2, markersize=6)

    ax.set_xlabel("Depth", fontsize=14)
    ax.set_ylabel("Attention distance (px)", fontsize=14)
    ax.set_title("Attention Distance - All Models", fontsize=16)
    ax.set_ylim(top=140, bottom=0)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logging.info(f"All models attention distance plot saved to {output_path}")
    plt.close()


def main():
    """Main function to generate attention distance analysis plots."""
    # Setup dataset
    logging.info("Setting up dataset...")
    dataset_test = setup_dataset()
    
    # Create output directory if it doesn't exist
    Path("scripts/output").mkdir(exist_ok=True)
    
    # Generate comparison plot for all models (legacy format)
    logging.info("Generating attention distance comparison plot...")
    plot_attention_distance_comparison(dataset_test, MODELS)
    
    # Generate single plot with all models
    logging.info("Generating combined attention distance plot with all models...")
    plot_all_models_attention_distance(dataset_test, MODELS)
    
    logging.info("Attention distance analysis completed!")


if __name__ == "__main__":
    main()
