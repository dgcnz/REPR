import torch
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import torchvision.transforms as T
from omegaconf import OmegaConf
import hydra
import logging

logging.basicConfig(level=logging.INFO)

# Configuration
PATCH_SIZE = 16  # must match model.patch_size
IMG_SIZE = 224  # must match model.img_size
DEVICE = "cuda"

# Preprocessing: convert to tensor, normalize with ImageNet stats
transform = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

CFG_FOLDER = Path("fabric_configs/experiment/hummingbird/model")
MODELS = [
    "dino_b",
    "cim_b",
    "droppos_b",
    "partmae_v5_b",
    "partmae_v6_ep074_b",
    "partmae_v6_ep099_b",
    "partmae_v6_ep149_b",
    "partmae_v6_b",  # 199
    "mae_b",
    "part_v0",
]
NAMES = {
    "dino_b": "DINO",
    "cim_b": "CIM",
    "droppos_b": "DropPos",
    "partmae_v6_b": "Ours",
    "partmae_v5_b": r"Ours ($\mathcal{L}_{\rm pose}$ only)",
    "partmae_v6_ep099_b": "Ours (Epoch 99)",
    "partmae_v6_ep074_b": "Ours (Epoch 74)",
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
    return f"Model '{model_name}' loaded successfully."


def compute_cls_token_affinities(pil_imgs: list[Image.Image], temperature: float):
    """
    Compute affinities between CLS token and all patch tokens for a batch of images.

    Args:
        pil_imgs: List of input PIL images
        temperature: Temperature for softmax normalization

    Returns:
        affinities: numpy array of shape [B, N] where B is batch size and N is number of patch tokens
    """
    # Preprocess images
    x = torch.stack([transform(img) for img in pil_imgs]).to(DEVICE)  # [B, C, H, W]

    with torch.no_grad():
        feats = model.forward_features(x)  # [B, N+1, D] where N+1 includes CLS token
        # l2 normalize features
        feats = torch.nn.functional.normalize(feats, dim=-1)

        # Extract CLS token and patch tokens
        cls_token = feats[:, 0, :]  # [B, D] - CLS token is typically the first token
        patch_tokens = feats[
            :, model.num_prefix_tokens :, :
        ]  # [B, N, D] - patch tokens

        # Compute similarities between CLS token and all patch tokens
        # cls_token: [B, D], patch_tokens: [B, N, D]
        affinities = torch.bmm(
            cls_token.unsqueeze(1), patch_tokens.transpose(1, 2)
        )  # [B, 1, N]
        affinities = affinities.squeeze(1)  # [B, N]

    # Apply softmax with temperature
    if temperature > 0:
        affinities = torch.nn.functional.softmax(affinities / temperature, dim=1)

    return affinities.cpu().numpy()


def create_cls_affinity_heatmap(pil_img: Image.Image, affinities: np.ndarray):
    """
    Create a heatmap visualization of CLS token affinities overlaid on the original image.

    Args:
        pil_img: Original PIL image
        affinities: Affinity values for each patch [N]

    Returns:
        PIL Image with heatmap overlay
    """
    N = len(affinities)
    grid_size = int(np.sqrt(N))

    # Reshape affinities to spatial grid
    heat = affinities.reshape(grid_size, grid_size)

    # Normalize to [0, 1]
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

    # Apply colormap
    cmap = plt.get_cmap("jet")
    heat_rgba = (cmap(heat) * 255).astype(np.uint8)
    heat_img = Image.fromarray(heat_rgba).convert("RGBA")
    heat_img = heat_img.resize(pil_img.size, resample=Image.BILINEAR)

    # Blend with original image
    base = pil_img.convert("RGBA")
    overlay = Image.blend(base, heat_img, alpha=0.6)
    return overlay.convert("RGB")


def plot_cls_affinities_grid(image_paths, model_names, temperatures):
    """
    Create a grid plot showing CLS token affinities for multiple images and models.
    Processes images in batches for each model to improve performance.

    Args:
        image_paths: List of paths to input images
        model_names: List of model names to compare
        temperatures: Dict mapping model names to temperature values
    """
    num_images = len(image_paths)
    num_models = len(model_names)

    # Create subplot grid: rows = images, cols = models + 1 (original)
    fig, axs = plt.subplots(
        num_images,
        num_models + 1,
        figsize=((num_models + 1) * 3, num_images * 3),
        squeeze=False,
    )

    # Set column titles
    axs[0, 0].set_title("Original", fontsize=18, pad=10)
    for model_idx, model_name in enumerate(model_names):
        display_name = NAMES.get(model_name, model_name)
        axs[0, model_idx + 1].set_title(display_name, fontsize=18, pad=10)

    # Load and display original images
    pil_imgs = []
    for img_idx, img_path in enumerate(image_paths):
        logging.info(f"Loading image {img_idx + 1}/{num_images}: {Path(img_path).name}")
        pil_img = (
            Image.open(img_path)
            .convert("RGB")
            .resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        )
        pil_imgs.append(pil_img)

        axs[img_idx, 0].imshow(pil_img)
        axs[img_idx, 0].axis("off")
        # Set row titles (image names) on the first column
        axs[img_idx, 0].set_ylabel(
            Path(img_path).stem, rotation=90, va="center", fontsize=10
        )

    # Process each model for all images
    for model_idx, model_name in enumerate(model_names):
        logging.info(f"Processing model: {model_name} for all images")
        load_model(model_name)

        # Get temperature for this model
        temp = temperatures.get(model_name, 0.1)

        # Compute CLS token affinities for the batch of images
        affinities_batch = compute_cls_token_affinities(pil_imgs, temp)

        # Create heatmap for each image
        for img_idx, pil_img in enumerate(pil_imgs):
            affinities = affinities_batch[img_idx]
            heatmap_img = create_cls_affinity_heatmap(pil_img, affinities)

            # Place in grid
            col_idx = model_idx + 1
            axs[img_idx, col_idx].imshow(heatmap_img)
            axs[img_idx, col_idx].axis("off")

    # Adjust layout
    # plt.subplots_adjust(wspace=0.02, hspace=0.1)
    plt.subplots_adjust(wspace=0.01, hspace=0.05)

    # Save plot
    output_path = "scripts/output/cls_token_affinities_grid_batch.png"
    plt.savefig(output_path, dpi=500, bbox_inches="tight", pad_inches=0.1)
    logging.info(f"CLS token affinities grid saved to {output_path}")
    plt.close()


def plot_single_cls_affinity(image_path, model_name, temperature=0.1):
    """
    Create a simple side-by-side plot for a single image and model.

    Args:
        image_path: Path to input image
        model_name: Name of the model to use
        temperature: Temperature for softmax normalization
    """
    logging.info(f"Processing image: {Path(image_path).name} with model: {model_name}")

    # Load model and image
    load_model(model_name)
    pil_img = (
        Image.open(image_path)
        .convert("RGB")
        .resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    )

    # Compute CLS token affinities
    affinities = compute_cls_token_affinities([pil_img], temperature)[0]

    # Create heatmap overlay
    heatmap_img = create_cls_affinity_heatmap(pil_img, affinities)

    # Create side-by-side plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(pil_img)
    ax1.set_title("Original Image")
    ax1.axis("off")

    ax2.imshow(heatmap_img)
    display_name = NAMES.get(model_name, model_name)
    ax2.set_title(f"CLS Token Affinities - {display_name}")
    ax2.axis("off")

    plt.tight_layout()

    # Save plot
    img_name = Path(image_path).stem
    output_path = f"scripts/output/cls_affinity/{img_name}/{model_name}_batch.png"
    Path(f"scripts/output/cls_affinity/{img_name}").mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logging.info(f"CLS token affinity plot saved to {output_path}")
    plt.close()


# Example configurations
# SAMPLE_IMAGES = [
#     "scripts/inputs/dog.jpg",
#     "scripts/inputs/plane.jpg",
#     "scripts/inputs/polarbears.jpg",
#     "scripts/inputs/zebras.jpg",
#     "scripts/inputs/dogs.jpg",
# ]

SAMPLE_IMAGES = [
    "scripts/inputs/polarbears.jpg",
    "scripts/inputs/threepeople.jpg",
    # "scripts/inputs/threepeople2.jpg",
    "scripts/inputs/plane_corner.jpg",
    "scripts/inputs/zebras.jpg",
    # "scripts/inputs/wiiplaying.jpg"
]

SAMPLE_MODELS = [
    "dino_b",
    "mae_b",
    "cim_b",
    "droppos_b",
    "part_v0",
    "partmae_v5_b",
    "partmae_v6_b",
]

# Temperature settings optimized for CLS token visualization
CLS_TEMPERATURES = {
    "dino_b": 0.1,
    "mae_b": 0.2,
    "cim_b": 0.1,
    "droppos_b": 0.075,
    "part_v0": 0.2,
    "partmae_v5_b": 0.075,
    "partmae_v6_b": 0.0275,
    "partmae_v6_ep074_b": 0.1,
    "partmae_v6_ep099_b": 0.1,
}


def main():
    """
    Main function to generate CLS token affinity visualizations.
    """
    # Create output directory if it doesn't exist
    Path("scripts/output").mkdir(exist_ok=True)

    # Example 1: Single image, single model
    logging.info("Generating single CLS affinity plot...")
    plot_single_cls_affinity("artifacts/samoyed.jpg", "partmae_v6_b", temperature=0.1)

    # Example 2: Multiple images, multiple models grid
    logging.info("Generating CLS affinities grid...")
    plot_cls_affinities_grid(SAMPLE_IMAGES, SAMPLE_MODELS, CLS_TEMPERATURES)

    logging.info("CLS token affinity visualization completed!")


if __name__ == "__main__":
    main()
