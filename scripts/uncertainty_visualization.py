#!/usr/bin/env python
# coding: utf-8

"""
Self-contained script for generating uncertainty distribution visualizations
for multiple images using a trained PART model.

This script creates a multi-row visualization where each row shows:
1. Original image (canonical view)
2. Ground truth reconstruction  
3. Model prediction reconstruction
4. Uncertainty distributions (Laplace distributions showing position uncertainty)

The uncertainty distributions visualization shows overlaid Laplace distributions
at predicted patch positions, where brighter areas indicate higher uncertainty
in patch position predictions.

Usage:
    python uncertainty_visualization.py

Requirements:
    - A trained PART model checkpoint at the specified path
    - Images in the artifacts/ directory
    - All dependencies listed in the imports section
"""

# Standard library imports
import itertools
import math
from pathlib import Path

# Third-party imports
import hydra
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils._pytree as pytree
from jaxtyping import Float
from omegaconf import OmegaConf
from PIL import Image
from torch import Tensor
from torch.utils.data import default_collate

# Local imports
from src.utils.visualization.reconstruction_v5_gt import reconstruction_gt

# Configuration
ROOT = Path("./")
torch.set_float32_matmul_precision("high")

# Output directory for saving figures
OUTPUT_DIR = ROOT / "scripts/output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Register resolver if not exists
if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval)

# Model configuration
V = 12  # Number of views
gV = 2  # Global views
lV = V - gV  # Local views

# Paths
FOLDER = ROOT / Path("outputs/2025-06-22/19-16-53")
CKPT_PATH = FOLDER / "epoch_0199.ckpt"


def clean_model_io(batch: tuple, out: dict, device="cuda"):
    """
    Clean and organize model inputs and outputs for visualization and analysis.
    
    Args:
        batch: A tuple containing model inputs (global images, global params, local images, local params)
        out: Model output dictionary
        device: Device to move tensors to (default: "cuda")
        
    Returns:
        io: Dictionary containing organized model inputs and outputs
    """
    # Initialize output dictionary
    io = dict()
    
    # Extract shapes from model output
    io["x"] = [list(itertools.chain.from_iterable(items)) for items in zip(*batch[0])]
    io["params"] = [list(itertools.chain.from_iterable(items)) for items in zip(*batch[1])]
    io["canonical_params"] = [[param[0:4] for param in batch_params] for batch_params in io["params"]][0]
    io["crop_params"] = [[param[4:8] for param in batch_params] for batch_params in io["params"]]
    
    # Include all output values
    io.update({name: out[name] for name in out.keys()})
    
    # Move all tensors to the specified device
    io = pytree.tree_map_only(
        Tensor,
        lambda t: t.detach().to(device),
        io
    )
    return io


def create_laplace_distribution_heatmaps(
    patch_positions: Float[Tensor, "M 2"],
    dispersions: Float[Tensor, "M 4"],  # [dy, dx, dlogh, dlogw]
    canonical_size: int,
    patch_size: int,
    alpha: float = 0.7,
) -> Float[Tensor, "canonical_size canonical_size"]:
    """
    Create overlaid Laplace distribution heatmaps at each predicted patch position.
    This shows the raw uncertainty distributions without confidence transformation.
    
    Args:
        patch_positions: Predicted patch positions in canonical space [M, 2]
        dispersions: Per-patch dispersions [M, 4] (in pixel units)
        canonical_size: Size of canonical image
        patch_size: Size of patches
        alpha: Blending factor for overlapping distributions
        
    Returns:
        combined_heatmap: Combined Laplace distributions heatmap
    """
    device = patch_positions.device
    combined_heatmap = torch.zeros((canonical_size, canonical_size), device=device)
    
    # Create coordinate grids
    y_coords = torch.arange(canonical_size, device=device).float()
    x_coords = torch.arange(canonical_size, device=device).float()
    Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    for pos, disp in zip(patch_positions, dispersions):
        mu_y, mu_x = pos[0], pos[1]
        b_y, b_x = disp[0], disp[1]  # Laplace scale parameters (already in pixel units)
        
        # Ensure reasonable scale parameters
        b_y = torch.clamp(b_y, min=0.5, max=canonical_size/2)
        b_x = torch.clamp(b_x, min=0.5, max=canonical_size/2)
        
        # Only compute distribution if patch is reasonably within extended bounds
        if (mu_y >= -2*patch_size and mu_y <= canonical_size + 2*patch_size and 
            mu_x >= -2*patch_size and mu_x <= canonical_size + 2*patch_size):
            
            # Compute Laplace distribution: (1/(4*b_y*b_x)) * exp(-|y-mu_y|/b_y - |x-mu_x|/b_x)
            laplace_dist = (1.0 / (4 * b_y * b_x)) * torch.exp(
                -torch.abs(Y - mu_y) / b_y - torch.abs(X - mu_x) / b_x
            )
            
            # Normalize to [0,1] to prevent any single distribution from dominating
            if laplace_dist.max() > 0:
                laplace_dist = laplace_dist / laplace_dist.max()
            
            # Add to combined heatmap with additive blending
            combined_heatmap += alpha * laplace_dist
    
    # Normalize the final combined heatmap
    if combined_heatmap.max() > 0:
        combined_heatmap = combined_heatmap / combined_heatmap.max()
    
    return combined_heatmap


def paste_patch(
    crop: Float[Tensor, "C h w"],
    pos: Float[Tensor, "2"],
    pos_canonical: Float[Tensor, "2"],
    patch_size_canonical: Float[Tensor, "2"],
    canvas: Float[Tensor, "C H W"],
    count_map: Float[Tensor, "1 H W"],
    patch_size: int,
    canonical_size: int,
    disp: Float[Tensor, "4"] = None,
):
    """
    Extract a patch from a crop at pos and paste it onto a canvas at pos_canonical with appropriate rescaling.

    Args:
        crop: Source image crop of shape [C, h, w]
        pos: Patch position in crop coordinates [y, x]
        pos_canonical: Target position in canonical coordinates [y, x]
        patch_size_canonical: Size of patch in canonical space [height, width]
        canvas: Target canvas to paste onto [C, H, W]
        count_map: Counter for averaging overlapping patches [1, H, W]
        patch_size: Size of patch in crop space
        canonical_size: Size of the canonical image
        disp: Per token dispersion (as in Laplace scale) for each transformation parameter.
    """
    crop_h, crop_w = crop.shape[1:3]

    # Convert to integer coordinates for the canonical position
    y_canonical, x_canonical = int(round(pos_canonical[0].item())), int(
        round(pos_canonical[1].item())
    )

    # Get integer patch size for the canonical space
    patch_h_canonical, patch_w_canonical = patch_size_canonical.round().int()

    # Ensure the patch fits within the canonical canvas
    y_canonical = max(0, min(canonical_size - patch_h_canonical, y_canonical))
    x_canonical = max(0, min(canonical_size - patch_w_canonical, x_canonical))

    # Get source patch coordinates, ensuring they're within the crop boundaries
    y_crop, x_crop = int(round(pos[0].item())), int(round(pos[1].item()))
    y_crop = max(0, min(crop_h - patch_size, y_crop))
    x_crop = max(0, min(crop_w - patch_size, x_crop))

    # Extract the patch from the source crop
    patch = crop[
        :, y_crop : y_crop + patch_size, x_crop : x_crop + patch_size
    ].unsqueeze(0)

    # Resize the patch to the canonical size
    patch_resized = F.interpolate(
        patch,
        size=(patch_h_canonical, patch_w_canonical),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    # Add the patch to the canvas and update the count map
    canvas[
        :,
        y_canonical : y_canonical + patch_h_canonical,
        x_canonical : x_canonical + patch_w_canonical,
    ] += patch_resized
    count_map[
        :,
        y_canonical : y_canonical + patch_h_canonical,
        x_canonical : x_canonical + patch_w_canonical,
    ] += 1


@torch.no_grad
def reconstruction_with_uncertainty_distributions(
    x: list[Float[Tensor, "C gH gW"] | Float[Tensor, "C lH lW"]],
    patch_positions_nopos: Float[Tensor, "M 2"],
    num_tokens: list[int],
    crop_params: list[Float[Tensor, "4"]],
    patch_size: int,
    canonical_img_size: int,
    max_scale_ratio: float,
    pred_dT: Float[Tensor, "M M 4"],
    disp_T: Float[Tensor, "M 4"],  # NOTE: This contains LOG-dispersions
) -> tuple[
    Float[Tensor, "C canonical_img_size canonical_img_size"],  # reconstructed image
    Float[Tensor, "canonical_img_size canonical_img_size"],    # uncertainty map
]:
    """
    Reconstruct image with uncertainty distribution visualization.
    
    Args:
        disp_T: Per-token log-dispersions [M, 4] - NOTE: these are in log-space!
    
    Returns:
        reconstructed_img: Reconstructed canonical image
        uncertainty_map: Uncertainty distribution visualization
    """
    device = x[0].device
    C = x[0].shape[0]

    # Undo normalization
    dT = pred_dT[..., :2] * canonical_img_size
    dS = pred_dT[..., 2:] * math.log(max_scale_ratio)

    # Choose anchor
    T_anchor = (
        crop_params[0][:2]
        + (patch_positions_nopos[0] / x[0].shape[1]) * crop_params[0][2:4]
    )
    S_anchor = torch.log((patch_size * crop_params[0][2:4] / x[0].shape[1]))

    T_global = dT[:, 0] + T_anchor
    S_global = dS[:, 0] + S_anchor

    T_global_grouped = torch.split(T_global, num_tokens)
    S_global_grouped = torch.split(S_global, num_tokens)
    patch_positions_nopos_grouped = torch.split(patch_positions_nopos, num_tokens)
    disp_T_grouped = torch.split(disp_T, num_tokens)

    # Reconstruct the canonical image
    canvas = torch.zeros((C, canonical_img_size, canonical_img_size), device=device)
    count_map = torch.zeros((1, canonical_img_size, canonical_img_size), device=device)

    for crop, patch_positions, canonical_pos, log_size, disp in zip(
        x,
        patch_positions_nopos_grouped,
        T_global_grouped,
        S_global_grouped,
        disp_T_grouped,
    ):
        N = patch_positions.shape[0]
        for i in range(N):
            paste_patch(
                crop=crop,
                pos=patch_positions[i].float(),
                pos_canonical=canonical_pos[i],
                patch_size_canonical=torch.exp(log_size[i]),
                canvas=canvas,
                count_map=count_map,
                patch_size=patch_size,
                canonical_size=canonical_img_size,
                disp=disp[i]
            )

    count_map[count_map == 0] = 1
    reconstructed_img = canvas / count_map

    # Generate uncertainty distribution visualization
    actual_dispersions = torch.exp(disp_T)  # Convert from log-space
    disp_T_pixels = actual_dispersions.clone()
    disp_T_pixels[:, :2] *= canonical_img_size  # dy, dx to pixels
    disp_T_pixels[:, 2:] *= math.log(max_scale_ratio)  # log-scale factors
    
    uncertainty_map = create_laplace_distribution_heatmaps(
        T_global, disp_T_pixels, canonical_img_size, patch_size
    )

    return reconstructed_img, uncertainty_map


def load_model_and_config():
    """Load configuration and model with proper checkpoint handling."""
    cfg = OmegaConf.load(FOLDER / ".hydra/config.yaml")
    
    # Handle uncertainty mode configuration
    if "predict_uncertainty" in cfg["model"]:
        predict_uncertainty = cfg["model"].pop("predict_uncertainty")
        cfg["model"]["uncertainty_mode"] = "additive" if predict_uncertainty else "none"
    elif "uncertainty_mode" not in cfg["model"]:
        raise ValueError("Uncertainty mode not specified in the config.")
    
    print(f"Uncertainty mode: {cfg['model']['uncertainty_mode']}")
    
    # Load and process checkpoint
    ckpt = torch.load(CKPT_PATH, map_location="cuda")
    state_dict = ckpt["model"]
    
    # Handle legacy parameter names
    legacy_mappings = {
        "pose_head.mu.weight": "pose_head.mu_proj.weight",
        "pose_head.logvar.weight": "pose_head.disp_proj.weight",
        "pose_head.logvar.bias": "pose_head.disp_proj.bias",
    }
    
    for old_key, new_key in legacy_mappings.items():
        if old_key in state_dict:
            state_dict[new_key] = state_dict.pop(old_key)
    
    # Handle gate dimension
    if "pose_head.gate_proj.weight" in state_dict:
        if "gate_dim" not in cfg["model"]:
            cfg["model"]["gate_dim"] = state_dict["pose_head.gate_proj.weight"].shape[0]
            print(f"Inferred gate dimension: {cfg['model']['gate_dim']}")
        assert cfg["model"]["gate_dim"] == state_dict["pose_head.gate_proj.weight"].shape[0]
    
    # Initialize missing parameters
    if "pose_head.gate_mult" not in state_dict:
        state_dict["pose_head.gate_mult"] = torch.zeros(1)
    
    # Create model
    model_config = {
        "_target_": "src.models.components.partmae_v6.PARTMaskedAutoEncoderViT",
        "num_views": V,
        "mask_ratio": 0.75 if V == 12 else 0,
        "pos_mask_ratio": 0.75,
    }
    
    if V not in [2, 12]:
        raise ValueError(f"Unsupported number of views: {V}")
    
    model = hydra.utils.instantiate(cfg["model"], **model_config)
    model.load_state_dict(state_dict, strict=True)
    
    print(f"Loaded model from epoch {ckpt['epoch']}, global step {ckpt['global_step']}")
    return model, cfg


def process_single_image(model, train_transform, img_path):
    """Process a single image and return the required outputs."""
    img = Image.open(img_path)
    
    # Create batch with the same image repeated 4 times (as in original code)
    batch = default_collate([train_transform(img)] * 4)
    
    # Run inference
    with torch.no_grad():
        out = model(*batch)
    
    io = clean_model_io(batch, out, 'cuda')
    
    # Generate reconstructions
    gt_reconstruction = reconstruction_gt(
        x=io["x"][0],
        patch_positions_nopos=io["patch_positions_nopos"][0],
        num_tokens=model._Ms,
        crop_params=io["crop_params"][0],
        patch_size=model.patch_size,
        canonical_img_size=model.canonical_img_size,
    )
    
    pred_reconstruction, uncertainty_map = reconstruction_with_uncertainty_distributions(
        x=io["x"][0],
        patch_positions_nopos=io["patch_positions_nopos"][0],
        num_tokens=model._Ms,
        crop_params=io["crop_params"][0],
        patch_size=model.patch_size,
        canonical_img_size=model.canonical_img_size,
        max_scale_ratio=model.max_scale_ratio,
        pred_dT=io["pred_dT"][0],
        disp_T=io["disp_T"][0],
    )
    
    # Get canonical image
    canonical_img = train_transform.recreate_canonical(
        img, io["canonical_params"][0]
    )
    
    return canonical_img, gt_reconstruction, pred_reconstruction, uncertainty_map


def plot_multiple_images_with_uncertainty(
    image_paths: list[Path],
    model,
    train_transform,
    save_path: Path = None,
    figsize_per_row: tuple = (16, 4),
):
    """
    Plot multiple images with uncertainty distributions in a grid layout.
    
    Args:
        image_paths: List of paths to images to process
        model: Trained model
        train_transform: Data transformation pipeline
        save_path: Path to save the figure
        figsize_per_row: Figure size per row (width, height)
    """
    n_images = len(image_paths)
    
    # Create figure with 4 columns (original, gt, pred, uncertainty) and n_images rows
    fig, axes = plt.subplots(
        n_images, 4, 
        figsize=(figsize_per_row[0], figsize_per_row[1] * n_images)
    )
    
    # Handle case where there's only one image (axes won't be 2D)
    if n_images == 1:
        axes = axes.reshape(1, -1)
    
    for i, img_path in enumerate(image_paths):
        print(f"Processing image {i+1}/{n_images}: {img_path.name}")
        
        try:
            canonical_img, gt_reconstruction, pred_reconstruction, uncertainty_map = process_single_image(
                model, train_transform, img_path
            )
            
            # Original image
            axes[i, 0].imshow(canonical_img)
            axes[i, 0].set_title("Original" if i == 0 else "", fontsize=16)
            axes[i, 0].axis("off")
            
            # GT reconstruction
            gt_img = gt_reconstruction.permute(1, 2, 0).cpu()
            gt_img = torch.clamp(gt_img, 0, 1)  # Clamp to valid range
            axes[i, 1].imshow(gt_img)
            axes[i, 1].set_title("Ground Truth Reconstruction" if i == 0 else "", fontsize=16)
            axes[i, 1].axis("off")
            
            # Predicted reconstruction  
            pred_img = pred_reconstruction.permute(1, 2, 0).cpu()
            pred_img = torch.clamp(pred_img, 0, 1)  # Clamp to valid range
            axes[i, 2].imshow(pred_img)
            axes[i, 2].set_title("Predicted Reconstruction" if i == 0 else "", fontsize=16)
            axes[i, 2].axis("off")
            
            # Uncertainty distributions
            axes[i, 3].imshow(uncertainty_map.cpu(), cmap='plasma', alpha=0.8)
            axes[i, 3].set_title("Uncertainty Distribution" if i == 0 else "", fontsize=16)
            axes[i, 3].axis("off")
            
            # Add image name as y-label
            axes[i, 0].set_ylabel(img_path.stem, rotation=0, ha='right', va='center')
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            # Fill with blank plots in case of error
            for j in range(4):
                axes[i, j].text(0.5, 0.5, f"Error: {e}", ha='center', va='center', transform=axes[i, j].transAxes)
                axes[i, j].axis("off")
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    
    return fig, axes


def main():
    """Main execution function."""
    # Load model and configuration
    model, cfg = load_model_and_config()
    
    # Prepare transform
    train_transform = hydra.utils.instantiate(
        cfg["data"]["transform"], 
        distort_color=False, 
        n_local_crops=lV
    )
    
    # Define image paths to process - modify this list to add/remove images
    image_paths = [
        # ROOT / "artifacts/samoyed.jpg",
        ROOT / "artifacts/polarbears.jpg", 
        ROOT / "artifacts/dog.jpg",
        # ROOT / "scripts/inputs/skiing.jpg",
        ROOT / "scripts/inputs/plane.jpg",
        # ROOT / "artifacts/people.jpg",
        ROOT / "artifacts/labrador.jpg",
    ]
    
    # Filter to only existing images
    existing_images = [path for path in image_paths if path.exists()]
    
    if not existing_images:
        print("No images found! Please check the image paths.")
        print("Available images in artifacts/:")
        artifacts_dir = ROOT / "artifacts"
        if artifacts_dir.exists():
            for file in artifacts_dir.glob("*.jpg"):
                print(f"  - {file}")
        return
    
    print(f"Found {len(existing_images)} images to process:")
    for path in existing_images:
        print(f"  - {path}")
    
    # Generate visualization
    save_path = OUTPUT_DIR / "uncertainty_distributions_multiple.png"
    fig, axes = plot_multiple_images_with_uncertainty(
        existing_images,
        model,
        train_transform,
        save_path=save_path,
        figsize_per_row=(16, 4)
    )
    
    # Close the figure to free memory
    plt.close(fig)
    
    print(f"\nVisualization complete! Saved to: {save_path}")


if __name__ == "__main__":
    main()
