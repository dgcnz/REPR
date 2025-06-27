import torch
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import torchvision.transforms as T
from omegaconf import OmegaConf
import hydra
import logging
import types

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
    "partmae_v6_b": "REPR",
    "partmae_v5_b": r"REPR ($\mathcal{L}_{\rm pose}$ only)",
    "partmae_v6_ep099_b": "REPR (Epoch 99)",
    "partmae_v6_ep074_b": "REPR (Epoch 74)",
    "mae_b": "MAE",
    "part_v0": "PART",
}

# Global model variable
model = None
OUTPUT_DIR = Path("scripts/output/cls_attention_averaged")

def load_model(model_name: str, temperature: float = 1.0):
    """Load a model from its configuration file and patch it to get attention."""
    global model
    logging.info(f"Loading model: {model_name}")
    assert model_name in MODELS, f"Model {model_name} not found in {MODELS}"
    model_conf = OmegaConf.load(CFG_FOLDER / f"{model_name}.yaml")
    model = (
        hydra.utils.instantiate({"model": model_conf}, _convert_="all")["model"]["net"]
        .eval()
        .to(DEVICE)
    )

    # Monkey-patch the forward method of the last attention block to get attention maps
    def new_attn_forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if temperature > 0:
            attn = torch.nn.functional.softmax(attn / temperature, dim=-1)
        
        # Store attention map
        self.attn_map = attn
        
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    model.blocks[-1].attn.forward = types.MethodType(new_attn_forward, model.blocks[-1].attn)
    
    logging.info(f"Model {model_name} loaded and patched.")
    return f"Model '{model_name}' loaded successfully."


def compute_cls_token_attention(pil_imgs: list[Image.Image]):
    """
    Compute attention from CLS token to all patch tokens for each head.
    This function assumes the model has been patched by load_model to store attention maps.
    
    Args:
        pil_imgs: List of input PIL images
        
    Returns:
        attentions: numpy array of shape [B, H, N] where B is batch size, H is number of heads,
                    and N is number of patch tokens.
    """
    # Preprocess images
    x = torch.stack([transform(img) for img in pil_imgs]).to(DEVICE)  # [B, C, H, W]

    with torch.no_grad():
        # This forward pass will trigger the patched attention forward method
        _ = model.forward_features(x)
        
        # Retrieve the stored attention map from the last block
        attentions = model.blocks[-1].attn.attn_map  # [B, H, N+1, N+1]

    # Extract attention from CLS token to patch tokens
    cls_attentions = attentions[:, :, 0, model.num_prefix_tokens:]  # [B, H, N]

    return cls_attentions.cpu().numpy()


def create_attention_heatmap(pil_img: Image.Image, attention: np.ndarray):
    """
    Create a heatmap visualization of attention values overlaid on the original image.
    
    Args:
        pil_img: Original PIL image
        attention: Attention values for each patch [N]
        
    Returns:
        PIL Image with heatmap overlay
    """
    N = len(attention)
    grid_size = int(np.sqrt(N))
    
    # Reshape attention to spatial grid
    heat = attention.reshape(grid_size, grid_size)
    
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


def compute_averaged_attention(image_path, model_name, temperature: float = 1.0):
    """
    Compute the averaged CLS token attention for a single model.
    
    Args:
        image_path: Path to input image
        model_name: Name of the model to use
        temperature: Temperature for softmax normalization
        
    Returns:
        averaged_attention: numpy array of shape [N] with averaged attention values
    """
    logging.info(f"Computing attention for model: {model_name}")
    
    # Load model and image
    load_model(model_name, temperature=temperature)
    pil_img = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    
    # Compute CLS token attentions for all heads
    # The function returns [B, H, N], and we have B=1
    attentions_all_heads = compute_cls_token_attention([pil_img])[0]  # [H, N]
    
    # Average attention across all heads
    averaged_attention = attentions_all_heads.mean(axis=0) # [N]
    
    return averaged_attention


def plot_attention_grid(image_paths, model_names, temperatures):
    """
    Create a grid plot showing CLS token attention for multiple images and models.
    Each row shows one image with all model attention overlays.
    
    Args:
        image_paths: List of paths to input images
        model_names: List of model names to compare
        temperatures: Dict mapping model names to temperature values
    """
    num_images = len(image_paths)
    num_models = len(model_names)
    
    # Create subplot grid: rows = images, cols = models + 1 (original)
    fig, axs = plt.subplots(num_images, num_models + 1, 
                           figsize=((num_models + 1) * 3, num_images * 3), 
                           squeeze=False)
    
    # Set column titles
    axs[0, 0].set_title("Original", fontsize=18, pad=10)
    for model_idx, model_name in enumerate(model_names):
        display_name = NAMES.get(model_name, model_name)
        axs[0, model_idx + 1].set_title(display_name, fontsize=18, pad=10)

    # Process each image
    for img_idx, img_path in enumerate(image_paths):
        logging.info(f"Processing image {img_idx + 1}/{num_images}: {Path(img_path).name}")
        
        # Load and display original image
        pil_img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        axs[img_idx, 0].imshow(pil_img)
        axs[img_idx, 0].axis("off")
        # Set row titles (image names) on the first column
        axs[img_idx, 0].set_ylabel(Path(img_path).stem, rotation=90, va='center', fontsize=10)

        # Process each model for this image
        for model_idx, model_name in enumerate(model_names):
            ax = axs[img_idx, model_idx + 1]
            temperature = temperatures[model_name]
            
            # Compute averaged attention for this model and image
            averaged_attention = compute_averaged_attention(img_path, model_name, temperature)
            
            # Create heatmap overlay
            heatmap_img = create_attention_heatmap(pil_img, averaged_attention)
            
            ax.imshow(heatmap_img)
            ax.axis("off")
    
    # Adjust layout for compact display
    # plt.subplots_adjust(wspace=0.02, hspace=0.1)
    plt.subplots_adjust(wspace=0.01, hspace=0.05)
    
    # Save plot
    output_path = OUTPUT_DIR / "attention_grid_all_images.png"
    plt.savefig(output_path, dpi=500, bbox_inches='tight', pad_inches=0.1)
    logging.info(f"Attention grid plot saved to {output_path}")
    plt.close()

IMG_MODEL_TEMPERATURES = {
    "dino_b": 1.75,
    "mae_b": 0.8,
    "cim_b": 0.6,
    "droppos_b": 1.5,
    "part_v0": 1.3,
    "partmae_v6_b": 1.1,
    "partmae_v5_b": 0.8,
    "partmae_v6_ep074_b": 1.1,
    "partmae_v6_ep099_b": 1.1,
}

# Sample images for grid visualization
SAMPLE_IMAGES = [
    "scripts/inputs/polarbears.jpg",
    "scripts/inputs/threepeople.jpg",
    # "scripts/inputs/threepeople2.jpg",
    "scripts/inputs/plane_corner.jpg",
    "scripts/inputs/zebras.jpg",
    # "scripts/inputs/wiiplaying.jpg"
]

def main():
    """
    Main function to generate CLS token attention visualizations.
    """
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(exist_ok=True)
    SELECTED_MODELS = [
        "dino_b",
        "mae_b",
        "cim_b",
        "droppos_b",
        "part_v0",
        "partmae_v5_b",
        # "partmae_v6_ep074_b",
        # "partmae_v6_ep099_b",
        "partmae_v6_b",
    ]

    # Generate attention grid with multiple images and models
    logging.info("Generating attention grid with multiple images and models...")
    plot_attention_grid(SAMPLE_IMAGES, SELECTED_MODELS, IMG_MODEL_TEMPERATURES)
    
    logging.info("Attention grid visualization completed!")


if __name__ == "__main__":
    main()
