import logging
from collections import defaultdict
from gradio_patch_selection import PatchSelector
from io import BytesIO

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

logging.basicConfig(level=logging.INFO)


# -------------------------
# HookedViT - Simplified Version
# -------------------------
class HookedViT(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model.eval()
        self.cache = defaultdict(dict)
        self.hooks = {}
        self.n_blocks = len(self.model.blocks)
        self.hook()

    def forward(self, x):
        self.cache.clear()  # Reset cache before each forward pass
        with torch.no_grad():
            out = self.model(x)
        return {
            "attns": self.get_attns(),
            "out": out
        }
    
    def _hook_fn(self, block_idx):
        def hook_fn(module, input, output):
            # Store attention outputs directly
            self.cache[block_idx] = output.detach().cpu()
        return hook_fn
        
    def hook(self):
        # Set up hooks only for attention modules
        for i in range(self.n_blocks):
            # Disable fused attention to access attention maps
            self.model.blocks[i].attn.fused_attn = False
            # Hook the attention drop module to capture attention maps
            self.hooks[i] = self.model.blocks[i].attn.attn_drop.register_forward_hook(
                self._hook_fn(i)
            )
    
    def get_attns(self):
        # Collect attention maps from all blocks
        attns = []
        for idx in range(self.n_blocks):
            if idx in self.cache:
                attns.append(self.cache[idx])
        return torch.stack(attns) if attns else None


# -------------------------
# Utility: Model Loader & Attention Visualization
# -------------------------
def load_hooked_vit(model_choice: str):
    model = timm.create_model(model_choice, pretrained=True)
    hooked_vit = HookedViT(model)
    transform = T.Compose(
        [
            T.Resize(model.default_cfg["input_size"][1]),
            T.CenterCrop(model.default_cfg["input_size"][1]),
            T.ToTensor(),
            T.Normalize(mean=model.default_cfg["mean"], std=model.default_cfg["std"]),
        ]
    )
    return hooked_vit, transform

def visualize_attention(
    model_choice: str, annotation_data: dict, block_index: int, is_cls_token: bool = False
) -> Image.Image:
    """
    Visualizes attention map for selected token (patch or CLS) in a ViT model
    
    Parameters:
    - model_choice: The model to use 
    - annotation_data: Dictionary with image data and patch index
    - block_index: Transformer block to visualize
    - is_cls_token: Whether to visualize CLS token attention
    """
    if annotation_data is None or "image" not in annotation_data:
        return None
    
    # Get the image from the annotation data
    image = annotation_data["image"]  # numpy array
    
    # Determine which token index to use
    if is_cls_token:
        # For CLS token, use index 0
        patch_index = 0
    else:
        # For regular patches
        if "patchIndex" not in annotation_data:
            return None
        patch_index = annotation_data["patchIndex"]
        patch_index = patch_index + 1  # Adjust for CLS token
    
    # Create PIL image from numpy array
    original_img = Image.fromarray(image)
    
    # Load model and get attention maps
    hooked_vit, transform = load_hooked_vit(model_choice)
    input_tensor = transform(original_img).unsqueeze(0)
    result = hooked_vit(input_tensor)
    
    # Access attention for selected block
    attn_stack = result["attns"]
    attn = attn_stack[block_index].squeeze(0)
    total_tokens = attn.shape[-1]
    
    # Validate token index
    if patch_index < 0 or patch_index >= total_tokens:
        return None
    
    # Extract attention weights from selected token to all other tokens (excluding CLS)
    token_attn = attn[:, patch_index, 1:]
    avg_attn = token_attn.mean(dim=0).numpy()  # Average across attention heads
    
    # Reshape to square grid
    num_patches = avg_attn.shape[0]
    grid_size = int(np.sqrt(num_patches))
    attn_grid = avg_attn.reshape(grid_size, grid_size)
    
    # Upsample attention map to image dimensions
    attn_tensor = torch.tensor(attn_grid).unsqueeze(0).unsqueeze(0)
    upsampled = F.interpolate(
        attn_tensor, size=original_img.size[::-1], mode="bilinear", align_corners=False
    )
    upsampled = upsampled.squeeze().numpy()
    
    # Normalize for visualization
    upsampled = (upsampled - upsampled.min()) / (
        upsampled.max() - upsampled.min() + 1e-8
    )
    
    # Create visualization with title
    title = f"{'[CLS] Token' if is_cls_token else f'Patch {patch_index-1}'} Attention Map (Block {block_index})"
    
    plt.figure(figsize=(6, 6))
    plt.imshow(original_img)
    plt.imshow(upsampled, cmap="jet", alpha=0.5)
    plt.title(title)
    plt.axis("off")
    
    # Save image to buffer
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    vis_img = Image.open(buf).convert("RGB")
    plt.close()
    
    return vis_img

# Example annotation for initial display
example_annotation = {
    "image": "https://gradio-builds.s3.amazonaws.com/demo-files/base.png",
    "patchIndex": 42,  # Example patch index
}

# Model choices
model_choices = ["vit_base_patch16_224.dino", "vit_base_patch14_dinov2.lvd142m", "vit_base_patch16_224.mae"]

with gr.Blocks() as demo:
    gr.Markdown("# HookedViT Attention Map Visualizer with Patch Selector")
    gr.Markdown(
        "Select a model, upload an image using the interface below, choose a transformer block, and then click on a patch in the grid to see the attention map. Use the [CLS] button to visualize the CLS token's attention."
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                choices=model_choices, label="Model", value=model_choices[0]
            )
            block_slider = gr.Slider(
                minimum=0, maximum=11, step=1, value=0, label="Transformer Block Index"
            )
            with gr.Row():
                visualize_button = gr.Button("Visualize Attention")
                cls_button = gr.Button("[CLS] Token Attention", variant="primary")
            
        with gr.Column(scale=2):
            output_img = gr.Image(type="pil", label="Attention Overlay")
    
    with gr.Row():
        with gr.Column():
            patch_selector = PatchSelector(
                example_annotation,
                img_size=224,  # Fixed image size for ViT models
                patch_size=16,  # Standard patch size for ViT
                show_grid=True,
                grid_color="rgba(200, 200, 200, 0.5)"
            )
            
        with gr.Column():    
            output_text = gr.Textbox(label="Selected Patch Info")

        # Show the patch index when a patch is selected
        patch_selector.patch_select(
            lambda event: f"Selected patch index: {event['patchIndex']}",
            patch_selector, 
            output_text
        )
    
    # Define a function to visualize CLS token attention
    def visualize_cls_attention(model_choice, annotation_data, block_index):
        return visualize_attention(model_choice, annotation_data, block_index, is_cls_token=True)
    
    # Visualize attention when button is clicked
    visualize_button.click(
        fn=visualize_attention,
        inputs=[model_dropdown, patch_selector, block_slider],
        outputs=output_img,
    )
    
    # Visualize CLS token attention when CLS button is clicked
    cls_button.click(
        fn=visualize_cls_attention,
        inputs=[model_dropdown, patch_selector, block_slider],
        outputs=output_img,
    )
    
    # Optionally, automatically visualize attention when a patch is selected
    patch_selector.patch_select(
        fn=visualize_attention,
        inputs=[model_dropdown, patch_selector, block_slider],
        outputs=output_img
    )

if __name__ == "__main__":
    demo.launch()