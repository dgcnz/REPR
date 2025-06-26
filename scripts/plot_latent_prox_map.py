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
MODEL = "partmae_v6_b"

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


def generate_individual_model_plot(model_name, model_plot_data):
    """
    Generates a plot for a single model showing latent similarities.
    """
    logging.info(f"Generating individual plot for model: {model_name}")
    load_model(model_name)
    
    num_rows = len(model_plot_data)
    fig, axs = plt.subplots(num_rows, 3, figsize=(12, num_rows * 4))
    
    # If only one row, axs is 1D array, so we need to handle this
    if num_rows == 1:
        axs = np.array([axs])
    
    for i, (img_a_path, img_b_path, patch_idx, temp) in enumerate(model_plot_data):
        pilA = Image.open(img_a_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        pilB = Image.open(img_b_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

        # Column 0: Image A with highlighted patch
        img_a_highlighted = highlight_patch(pilA, patch_idx, PATCH_SIZE)
        axs[i, 0].imshow(img_a_highlighted)
        axs[i, 0].set_title(f"Image A: {Path(img_a_path).name}\nPatch: {patch_idx}")
        axs[i, 0].axis("off")

        # Column 1: Image B
        axs[i, 1].imshow(pilB)
        axs[i, 1].set_title(f"Image B: {Path(img_b_path).name}")
        axs[i, 1].axis("off")

        # Column 2: Similarity heatmap overlay on Image B
        sim = compute_sim(pilA, pilB, temp)
        N = sim.shape[0]
        overlay = overlay_distance_heatmap(pilB, sim, N, patch_idx)
        axs[i, 2].imshow(overlay)
        axs[i, 2].set_title(f"Similarity Overlay (T: {temp})")
        axs[i, 2].axis("off")

    plt.suptitle(f"Latent Similarities - {model_name}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = f"scripts/output/latent_similarities_{model_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logging.info(f"Individual plot for {model_name} saved to {output_path}")
    plt.close()


def generate_combined_model_plot(plot_data):
    """
    Generates a combined plot showing latent similarities for all models.
    """
    logging.info("Generating combined plot for all models")
    
    # Calculate total rows: sum of all image pairs across all models
    total_rows = sum(len(model_data) for model_data in plot_data.values())
    
    # Create subplot: 5 columns (Model name, Image A, Image B, Overlay, spacer)
    fig, axs = plt.subplots(total_rows, 5, figsize=(20, total_rows * 3))
    # If only one row, axs is 1D array, so we need to handle this
    if total_rows == 1:
        axs = np.array([axs])

    current_row = 0
    
    for model_name, model_plot_data in plot_data.items():
        logging.info(f"Processing model: {model_name}")
        load_model(model_name)
        
        for img_a_path, img_b_path, patch_idx, temp in model_plot_data:
            pilA = Image.open(img_a_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            pilB = Image.open(img_b_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

            # Column 0: Model name
            axs[current_row, 0].text(0.5, 0.5, model_name, ha='center', va='center', 
                                    fontsize=12, fontweight='bold', rotation=90)
            axs[current_row, 0].axis("off")

            # Column 1: Image A with highlighted patch
            img_a_highlighted = highlight_patch(pilA, patch_idx, PATCH_SIZE)
            axs[current_row, 1].imshow(img_a_highlighted)
            axs[current_row, 1].set_title(f"Image A: {Path(img_a_path).name}\nPatch: {patch_idx}")
            axs[current_row, 1].axis("off")

            # Column 2: Image B
            axs[current_row, 2].imshow(pilB)
            axs[current_row, 2].set_title(f"Image B: {Path(img_b_path).name}")
            axs[current_row, 2].axis("off")

            # Column 3: Similarity heatmap overlay on Image B
            sim = compute_sim(pilA, pilB, temp)
            N = sim.shape[0]
            overlay = overlay_distance_heatmap(pilB, sim, N, patch_idx)
            axs[current_row, 3].imshow(overlay)
            axs[current_row, 3].set_title(f"Similarity Overlay (T: {temp})")
            axs[current_row, 3].axis("off")

            # Column 4: Spacer
            axs[current_row, 4].axis("off")

            current_row += 1

    plt.suptitle("Latent Similarities - Model Comparison", fontsize=16, fontweight='bold')
    plt.tight_layout()
    output_path = "scripts/output/latent_similarities_multi_model_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logging.info(f"Combined plot saved to {output_path}")
    plt.close()


def generate_compact_grid_plot(plot_data):
    """
    Generates a compact grid plot with no titles or metadata, just the images.
    Layout: Each row is a model, each set of 2 columns is an image pair (A with highlighted patch, B with overlay).
    """
    logging.info("Generating compact grid plot")
    
    num_models = len(plot_data)
    # Assume all models have the same number of image pairs
    num_pairs = len(next(iter(plot_data.values())))
    
    # Create subplot: rows = models, cols = pairs * 2 (A, B with Overlay for each pair)
    fig, axs = plt.subplots(num_models, num_pairs * 2, figsize=(num_pairs * 4, num_models * 2))
    
    # Handle single row case
    if num_models == 1:
        axs = np.array([axs])
    # Handle single column case
    if num_pairs == 1:
        axs = axs.reshape(num_models, -1)
    
    for row, (model_name, model_plot_data) in enumerate(plot_data.items()):
        logging.info(f"Processing model: {model_name}")
        load_model(model_name)
        
        for pair_idx, (img_a_path, img_b_path, patch_idx, temp) in enumerate(model_plot_data):
            pilA = Image.open(img_a_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            pilB = Image.open(img_b_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            
            col_base = pair_idx * 2
            
            # Column 0 of this pair: Image A with highlighted patch
            img_a_highlighted = highlight_patch(pilA, patch_idx, PATCH_SIZE)
            axs[row, col_base].imshow(img_a_highlighted)
            axs[row, col_base].axis("off")
            
            # Column 1 of this pair: Image B with similarity heatmap overlay
            sim = compute_sim(pilA, pilB, temp)
            N = sim.shape[0]
            overlay = overlay_distance_heatmap(pilB, sim, N, patch_idx)
            axs[row, col_base + 1].imshow(overlay)
            axs[row, col_base + 1].axis("off")
    
    # Remove all spacing between subplots for maximum compactness
    plt.subplots_adjust(wspace=0, hspace=0)
    
    output_path = "scripts/output/latent_similarities_compact_grid.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    logging.info(f"Compact grid plot saved to {output_path}")
    plt.close()

TEXTURE_BIAS =  {
        "dino_b": [
            ("scripts/inputs/samoyed_2.jpg", "scripts/inputs/retriever_snow.jpg", 52, 0.2),
        ],
        "droppos_b": [
            ("scripts/inputs/samoyed_2.jpg", "scripts/inputs/retriever_snow.jpg", 52, 0.2),
        ],
        "part_v0": [
            ("scripts/inputs/samoyed_2.jpg", "scripts/inputs/retriever_snow.jpg", 52, 0.05),
        ],
        "cim_b": [
            ("scripts/inputs/samoyed_2.jpg", "scripts/inputs/retriever_snow.jpg", 52, 0.15),
        ],
        "mae_b": [
            ("scripts/inputs/samoyed_2.jpg", "scripts/inputs/retriever_snow.jpg", 52, 0.3),
        ],
        "partmae_v6_b": [
            ("scripts/inputs/samoyed_2.jpg", "scripts/inputs/retriever_snow.jpg", 52, 0.02),
        ],
    }



def main():
    """
    Generates and saves plots showing latent similarities for multiple models and image pairs.
    Creates both individual plots for each model and a combined comparison plot.
    """
    # Dictionary of model names to their respective plot data
    # Each model_plot_data is a list of tuples: (img_A_path, img_B_path, patch_idx_in_A, temperature)
    # plot_data = {
    #     "dino_b": [
    #         ("artifacts/samoyed.jpg", "artifacts/samoyed_rot.jpg", 75, 0.9),
    #         ("scripts/inputs/peoplerestaurant.jpg", "scripts/inputs/familystanding.jpg", 80, 0.9),
    #         ("scripts/inputs/whitelaptop.jpg", "scripts/inputs/greylaptop.jpg", 80, 0.025),
    #     ],
    #     "droppos_b": [
    #         ("artifacts/samoyed.jpg", "artifacts/samoyed_rot.jpg", 75, 0.1),
    #         ("scripts/inputs/peoplerestaurant.jpg", "scripts/inputs/familystanding.jpg", 80, 0.2),
    #         ("scripts/inputs/whitelaptop.jpg", "scripts/inputs/greylaptop.jpg", 80, 0.15),
    #     ],
    #     # "partmae_v5_b": [
    #     #     ("artifacts/samoyed.jpg", "artifacts/samoyed_rot.jpg", 75, 0.02),
    #     #     ("scripts/inputs/peoplerestaurant.jpg", "scripts/inputs/familystanding.jpg", 80, 0.1),
    #     #     ("scripts/inputs/whitelaptop.jpg", "scripts/inputs/greylaptop.jpg", 80, 0.1),
    #     # ],
    #     "partmae_v6_ep074_b": [
    #         ("artifacts/samoyed.jpg", "artifacts/samoyed_rot.jpg", 75, 0.02),
    #         ("scripts/inputs/peoplerestaurant.jpg", "scripts/inputs/familystanding.jpg", 80, 0.02),
    #         ("scripts/inputs/whitelaptop.jpg", "scripts/inputs/greylaptop.jpg", 80, 0.02),
    #     ],
    #     "partmae_v6_ep099_b": [
    #         ("artifacts/samoyed.jpg", "artifacts/samoyed_rot.jpg", 75, 0.02),
    #         ("scripts/inputs/peoplerestaurant.jpg", "scripts/inputs/familystanding.jpg", 80, 0.02),
    #         ("scripts/inputs/whitelaptop.jpg", "scripts/inputs/greylaptop.jpg", 80, 0.02),
    #     ],
    #     # "partmae_v6_ep149_b": [
    #     #     ("artifacts/samoyed.jpg", "artifacts/samoyed_rot.jpg", 75, 0.02),
    #     #     ("scripts/inputs/peoplerestaurant.jpg", "scripts/inputs/familystanding.jpg", 80, 0.02),
    #     #     ("scripts/inputs/whitelaptop.jpg", "scripts/inputs/greylaptop.jpg", 80, 0.02),
    #     # ],
    #     "partmae_v6_b": [
    #         ("artifacts/samoyed.jpg", "artifacts/samoyed_rot.jpg", 75, 0.02),
    #         ("scripts/inputs/peoplerestaurant.jpg", "scripts/inputs/familystanding.jpg", 80, 0.02),
    #         ("scripts/inputs/whitelaptop.jpg", "scripts/inputs/greylaptop.jpg", 80, 0.02),
    #     ],
    # }
    plot_data = TEXTURE_BIAS

    # Generate individual plots for each model
    for model_name, model_plot_data in plot_data.items():
        generate_individual_model_plot(model_name, model_plot_data)
    
    # Generate combined plot with all models
    generate_combined_model_plot(plot_data)
    
    # Generate compact grid plot without titles/metadata
    generate_compact_grid_plot(plot_data)
    
    logging.info("All plots generated successfully!")

if __name__ == "__main__":
    main()
