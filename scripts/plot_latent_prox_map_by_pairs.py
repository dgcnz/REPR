import torch
import numpy as np
from PIL import Image, ImageDraw
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
    "partmae_v6_b", # 199
    "mae_b",
    "part_v0",
]
NAMES = {
    "dino_b": "DINO",
    "cim_b": "CIM",
    "droppos_b": "DropPos",
    "partmae_v6_b": "Ours",
    "partmae_v6_ep099_b": "Ours (Epoch 99)",
    "partmae_v6_ep074_b": "Ours (Epoch 74)",
    "partmae_v5_b": r"Ours ($\mathcal{L}_{\rm pose}$ only)",
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


def compute_sim(pilA: Image.Image, pilB: Image.Image, temperature: float):
    # Preprocess and batch
    xA = transform(pilA).to(DEVICE)
    xB = transform(pilB).to(DEVICE)
    x = torch.stack([xA, xB], dim=0)  # [2, C, H, W]

    with torch.no_grad():
        feats = model.forward_features(x)
        # l2 normalize features
        # remove cls token
        feats = feats[:, model.num_prefix_tokens :, :]  # [B, N, D]
        feats = torch.nn.functional.normalize(feats, dim=-1)
        # B N D

    # we're interested in seeing the distance (cosine) between patches in images A and B
    # not inside the same image.
    # so the final similarity matrix should be: N N
    # sim[i, j] = similarity between patch i in image A and patch j in image B
    featA, featB = feats[0], feats[1]  # each is [N, D]
    sim = featA @ featB.t()

    # Apply softmax with temperature
    if temperature > 0:
        sim = torch.nn.functional.softmax(sim / temperature, dim=-1)

    return sim.cpu().numpy()  # convert to numpy for further processing


def overlay_distance_heatmap(
    pilB: Image.Image, sim: np.ndarray, N: int, patch_idx_A: int
):
    grid_size = int(np.sqrt(N))
    sim = sim[patch_idx_A]  # get distances from patch A to all patches in B
    heat = sim.reshape(grid_size, grid_size)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

    cmap = plt.get_cmap("jet")
    heat_rgba = (cmap(heat) * 255).astype(np.uint8)
    heat_img = Image.fromarray(heat_rgba).convert("RGBA")
    heat_img = heat_img.resize(pilB.size, resample=Image.BILINEAR)

    base = pilB.convert("RGBA")
    overlay = Image.blend(base, heat_img, alpha=0.5)
    return overlay.convert("RGB")


def highlight_patch(image, patch_idx, patch_size, grid_color="red", line_width=3):
    """Draws a rectangle around a specified patch on the image."""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    grid_width = image.width // patch_size
    
    row = patch_idx // grid_width
    col = patch_idx % grid_width
    
    x1 = col * patch_size
    y1 = row * patch_size
    x2 = x1 + patch_size
    y2 = y1 + patch_size
    
    draw.rectangle([x1, y1, x2, y2], outline=grid_color, width=line_width)
    return img_copy


def generate_compact_grid_by_pairs(plot_data):
    """
    Generates a compact grid plot organized by image pairs.
    Layout: Each row is an image pair, first column shows Image A with highlighted patch,
    subsequent columns show Image B with heatmap overlay for each model.
    """
    logging.info("Generating compact grid plot organized by pairs")
    
    # Extract unique image pairs from the plot data
    # Assuming all models have the same image pairs
    first_model_data = next(iter(plot_data.values()))
    image_pairs = [(img_a_path, img_b_path, patch_idx, temp) for img_a_path, img_b_path, patch_idx, temp in first_model_data]
    
    num_pairs = len(image_pairs)
    num_models = len(plot_data)
    
    # Create subplot: rows = image pairs, cols = 1 (Image A) + num_models (Image B with overlays)
    fig, axs = plt.subplots(num_pairs, num_models + 1, figsize=((num_models + 1) * 2, num_pairs * 2), squeeze=False)
    
    model_names = list(plot_data.keys())
    
    # Multiple rows case
    axs[0, 0].set_title("Reference", fontsize=10, pad=5)
    for model_idx, model_name in enumerate(model_names):
        display_name = NAMES.get(model_name, model_name)
        axs[0, model_idx + 1].set_title(display_name, fontsize=10, pad=5)
    
    for pair_idx, (img_a_path, img_b_path, patch_idx, temp) in enumerate(image_pairs):
        logging.info(f"Processing image pair {pair_idx + 1}/{num_pairs}: {Path(img_a_path).name} -> {Path(img_b_path).name}")
        
        # Load and preprocess images
        pilA = Image.open(img_a_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        pilB = Image.open(img_b_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        
        # Column 0: Image A with highlighted patch (same for all models)
        img_a_highlighted = highlight_patch(pilA, patch_idx, PATCH_SIZE)
        axs[pair_idx, 0].imshow(img_a_highlighted)
        axs[pair_idx, 0].axis("off")
        
        # Columns 1 to num_models: Image B with heatmap overlay for each model
        for model_idx, model_name in enumerate(model_names):
            logging.info(f"Processing model: {model_name}")
            load_model(model_name)
            
            # Get the temperature for this model and image pair
            # Find the corresponding entry in this model's data
            model_data = plot_data[model_name]
            model_temp = None
            for model_img_a_path, model_img_b_path, model_patch_idx, model_temp_val in model_data:
                if (model_img_a_path == img_a_path and 
                    model_img_b_path == img_b_path and 
                    model_patch_idx == patch_idx):
                    model_temp = model_temp_val
                    break
            
            if model_temp is None:
                logging.warning(f"No temperature found for model {model_name} and pair {img_a_path} -> {img_b_path}")
                model_temp = temp  # fallback to the default temperature
            
            # Compute similarity and create overlay
            sim = compute_sim(pilA, pilB, model_temp)
            N = sim.shape[0]
            overlay = overlay_distance_heatmap(pilB, sim, N, patch_idx)
            
            # Place in grid
            col_idx = model_idx + 1
            axs[pair_idx, col_idx].imshow(overlay)
            axs[pair_idx, col_idx].axis("off")
    
    # Remove spacing between subplots but keep some space for titles
    plt.subplots_adjust(wspace=0, hspace=0.1)
    
    output_path = "scripts/output/latent_similarities/compact_grid_by_pairs.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    logging.info(f"Compact grid plot (by pairs) saved to {output_path}")
    plt.close()


# Example plot data - organized by texture bias experiment
TEXTURE_BIAS = {
    "dino_b": [
        ("scripts/inputs/samoyed_2.jpg", "scripts/inputs/retriever_snow.jpg", 52, 0.2),
    ],
    "mae_b": [
        ("scripts/inputs/samoyed_2.jpg", "scripts/inputs/retriever_snow.jpg", 52, 0.3),
    ],
    "droppos_b": [
        ("scripts/inputs/samoyed_2.jpg", "scripts/inputs/retriever_snow.jpg", 52, 0.15),
    ],
    "part_v0": [
        ("scripts/inputs/samoyed_2.jpg", "scripts/inputs/retriever_snow.jpg", 52, 0.05),
    ],
    "cim_b": [
        ("scripts/inputs/samoyed_2.jpg", "scripts/inputs/retriever_snow.jpg", 52, 0.15),
    ],
    "partmae_v5_b": [
        ("scripts/inputs/samoyed_2.jpg", "scripts/inputs/retriever_snow.jpg", 52, 0.2),
    ],
    "partmae_v6_b": [
        ("scripts/inputs/samoyed_2.jpg", "scripts/inputs/retriever_snow.jpg", 52, 0.02),
    ],
}

POSITIONAL_BIAS = {
    "dino_b": [
        ("artifacts/samoyed.jpg", "artifacts/samoyed_rot.jpg", 75, 0.7),
        ("scripts/inputs/peoplerestaurant.jpg", "scripts/inputs/familystanding.jpg", 80, 0.25),
        ("scripts/inputs/whitelaptop.jpg", "scripts/inputs/greylaptop.jpg", 80, 0.05),
    ],
    "droppos_b": [
        ("artifacts/samoyed.jpg", "artifacts/samoyed_rot.jpg", 75, 0.15),
        ("scripts/inputs/peoplerestaurant.jpg", "scripts/inputs/familystanding.jpg", 80, 0.2),
        ("scripts/inputs/whitelaptop.jpg", "scripts/inputs/greylaptop.jpg", 80, 0.2),
    ],
    "part_v0": [
        ("artifacts/samoyed.jpg", "artifacts/samoyed_rot.jpg", 75, 0.15),
        ("scripts/inputs/peoplerestaurant.jpg", "scripts/inputs/familystanding.jpg", 80, 0.1),
        ("scripts/inputs/whitelaptop.jpg", "scripts/inputs/greylaptop.jpg", 80, 0.1),
    ],
    "partmae_v6_ep074_b": [
        ("artifacts/samoyed.jpg", "artifacts/samoyed_rot.jpg", 75, 0.025),
        ("scripts/inputs/peoplerestaurant.jpg", "scripts/inputs/familystanding.jpg", 80, 0.03),
        ("scripts/inputs/whitelaptop.jpg", "scripts/inputs/greylaptop.jpg", 80, 0.035),
    ],
    "partmae_v6_ep099_b": [
        ("artifacts/samoyed.jpg", "artifacts/samoyed_rot.jpg", 75, 0.025),
        ("scripts/inputs/peoplerestaurant.jpg", "scripts/inputs/familystanding.jpg", 80, 0.03),
        ("scripts/inputs/whitelaptop.jpg", "scripts/inputs/greylaptop.jpg", 80, 0.035),
    ],
    "partmae_v6_b": [
        ("artifacts/samoyed.jpg", "artifacts/samoyed_rot.jpg", 75, 0.025),
        ("scripts/inputs/peoplerestaurant.jpg", "scripts/inputs/familystanding.jpg", 80, 0.03),
        ("scripts/inputs/whitelaptop.jpg", "scripts/inputs/greylaptop.jpg", 80, 0.035),
    ],
}


def main():
    """
    Generates and saves a compact grid plot organized by image pairs.
    Each row represents an image pair, and each column (after the first) represents a different model.
    """
    # plot_data = TEXTURE_BIAS
    plot_data = TEXTURE_BIAS
    
    # Generate compact grid plot organized by pairs
    generate_compact_grid_by_pairs(plot_data)
    
    logging.info("Compact grid plot (by pairs) generated successfully!")


if __name__ == "__main__":
    main()
