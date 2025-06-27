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
    "partmae_v6_b": "Ours",
    "partmae_v6_ep099_b": "Ours (Epoch 99)",
    "partmae_v6_ep074_b": "Ours (Epoch 74)",
    "mae_b": "MAE",
    "part_v0": "PART",
}

# Global model variable
model = None
OUTPUT_DIR = Path("scripts/output/cls_attention")

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


def plot_attention_per_head(image_path, model_name, temperature: float = 1.0):
    """
    Creates a grid plot showing CLS token attention for each head of a single image.
    
    Args:
        image_path: Path to input image
        model_name: Name of the model to use
        temperature: Temperature for softmax normalization
    """
    logging.info(f"Processing image: {Path(image_path).name} with model: {model_name}")
    
    # Load model and image
    load_model(model_name, temperature=temperature)
    pil_img = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    
    # Compute CLS token attentions for all heads
    # The function returns [B, H, N], and we have B=1
    attentions_all_heads = compute_cls_token_attention([pil_img])[0]  # [H, N]
    
    num_heads = attentions_all_heads.shape[0]
    
    # Create subplot grid, aiming for a square-like layout
    grid_cols = int(np.ceil(np.sqrt(num_heads)))
    grid_rows = int(np.ceil(num_heads / grid_cols))
    
    fig, axs = plt.subplots(grid_rows, grid_cols, 
                           figsize=(grid_cols * 3, grid_rows * 3), 
                           squeeze=False)
    axs = axs.flatten()
    
    fig.suptitle(f"CLS Token Attention per Head - {NAMES.get(model_name, model_name)}", fontsize=16)
    
    for head_idx in range(num_heads):
        ax = axs[head_idx]
        attention_one_head = attentions_all_heads[head_idx]  # [N]
        
        heatmap_img = create_attention_heatmap(pil_img, attention_one_head)
        
        ax.imshow(heatmap_img)
        ax.set_title(f"Head {head_idx + 1}")
        ax.axis("off")
        
    # Hide any unused subplots
    for i in range(num_heads, len(axs)):
        axs[i].axis("off")
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save plot
    img_name = Path(image_path).stem
    output_path = OUTPUT_DIR/f"{img_name}_{model_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logging.info(f"CLS token attention plot saved to {output_path}")
    plt.close()


def main():
    """
    Main function to generate CLS token attention visualizations.
    """
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(exist_ok=True)
    SELECTED_MODELS = [
        "dino_b",
        "mae_b",
        "part_v0",
        "cim_b",
        "droppos_b",
        "partmae_v5_b",
        "partmae_v6_b",
    ]

    CLS_ATTENTION_TEMPERATURES = {
        "dino_b": 0.99,
        "mae_b": 0.8,
        "cim_b": 0.6,
        "droppos_b": 0.8,
        "part_v0": 0.7,
        "partmae_v6_b": 0.7,
        "partmae_v5_b": 0.7,
        "partmae_v6_ep074_b": 0.7,
        "partmae_v6_ep099_b": 0.7,
    }
    IMG = "scripts/inputs/wiiplaying.jpg"
    
    # Example: Generate attention plots for a single image with a specific model
    logging.info("Generating CLS attention plot for all heads...")
    # Using dino_b as an example ViT model
    for model_name in SELECTED_MODELS:
        temperature = CLS_ATTENTION_TEMPERATURES[model_name]
        plot_attention_per_head(IMG, model_name, temperature=temperature)
    
    logging.info("CLS token attention visualization completed!")


if __name__ == "__main__":
    main()
