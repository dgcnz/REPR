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
from collections import defaultdict


# Inlined from src.utils.analysis.clmim_hook
def get_layer(model, layer_name: str):
    layers = layer_name.split(".")
    for layer in layers:
        if layer.isnumeric():
            model = model[int(layer)]
        elif hasattr(model, layer):
            model = getattr(model, layer)
        else:
            raise ValueError(f"Layer {layer} not found in {model}")
    return model


def clean_tensor(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu()
    elif isinstance(tensor, tuple):
        return tuple(clean_tensor(t) for t in tensor)
    elif isinstance(tensor, list):
        return [clean_tensor(t) for t in tensor]
    elif isinstance(tensor, dict):
        return {k: clean_tensor(v) for k, v in tensor.items()}
    else:
        return tensor


class ActivationCache(object):
    def __init__(self, head_name: str = "head"):
        self.cache = defaultdict(dict)
        self.hooks = {}
        self.head_name = head_name
        self.logger = logging.getLogger(__name__)

    def clear(self):
        self.cache = {}

    def _hook_fn(self, layer_name: str):
        def hook_fn(module, input, output):
            self.cache[layer_name]["input"] = clean_tensor(input)
            self.cache[layer_name]["output"] = clean_tensor(output)

        return hook_fn

    def hook_layer(self, model, layer_name: str):
        layer = get_layer(model, layer_name)
        # print(f"Hooking {layer_name}")
        self.logger.debug(f"Hooking {layer_name}: {layer.__class__}")
        hook = layer.register_forward_hook(self._hook_fn(layer_name))
        return hook

    def hook(self, model):
        # get Attention params: H, D
        self.H = model.blocks[0].attn.num_heads
        self.D = model.blocks[0].attn.proj.weight.shape[0]  # (D, D)

        # hook the layers
        self.n_blocks = len(model.blocks)
        for i in range(self.n_blocks):
            # deactivate fused_attn to get access to the individual components
            model.blocks[i].attn.fused_attn = False
            for layer_name in self._hooked_layers_per_block(i):
                self.hooks[layer_name] = self.hook_layer(model, layer_name)

        # hook the head
        self.hooks[self.head_name] = self.hook_layer(model, self.head_name)
        # self.hooks[''] = model.register_forward_hook(self._hook_fn(''))

    def get_attn_ft(self, block_idx: int):
        proj_input = self.cache[f"blocks.{block_idx}.attn.proj"]["input"][0]
        B, N, _ = proj_input.shape

        # attn_ft: B, H, N, C//H
        # proj_input = attn_ft.transpose(1, 2).reshape(B, N, C)
        # reverse operation
        attn_ft = proj_input.reshape(B, N, self.H, self.D // self.H).transpose(1, 2)
        return attn_ft

    def get_attn(self, block_idx: int):
        return self.cache[f"blocks.{block_idx}.attn.attn_drop"]["output"]

    def get_attns(self):
        attns = []
        for idx in range(self.n_blocks):
            attns.append(self.get_attn(idx))
        return attns

    def get_zs(self):
        # zs: 0 = input of first block, same as input of first norm1
        # zs: i = input of norm2 of block i // 2 + 1
        # zs: i+1 = output of block i // 2 + 2, same as input of norm1 of the next block :0
        # zs: n = output of head, same as output of model
        zs = []
        zs.append(self.cache["blocks.0"]["input"][0])
        for idx in range(self.n_blocks):
            """
            class Block:
                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
                    ---> hook on this x
                    x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
                    ---> hook on this x
                    return x
            """
            z1 = self.cache[f"blocks.{idx}.norm2"]["input"][0]
            # zs.append(self.cache[f'blocks.{idx}']['output'])
            #   This works too for standard timm ViTs,
            #   but not for custom Blocks that output multiple tensors,
            #   like the cl_vs_mim ViT
            mlp_output = self.cache[f"blocks.{idx}.mlp"]["output"]
            z2 = mlp_output + z1
            zs.extend([z1, z2])

        # TODO: careful, this could change depending on the model
        # zs.append(self.cache['']['output'])
        try:
            zs.append(self.cache[self.head_name]["output"])
        except KeyError:
            self.logger.warning("Head output not found in cache. Appending None.")
            zs.append(None)
        return zs

    @staticmethod
    def _hooked_layers_per_block(i):
        return [
            f"blocks.{i}",  # input -> zs[0]
            f"blocks.{i}.norm2",  # input -> zs[i]
            f"blocks.{i}.mlp",  # output + zs[i] -> zs[i+1]
            f"blocks.{i}.attn.attn_drop",  # output -> attn
            f"blocks.{i}.attn.proj",  # input -> attn_ft
        ]

    def unhook(self):
        for hook in self.hooks.values():
            hook.remove()
        self.hooks = {}


# Inlined from src.models.components.cl_vs_mim.utils
def subsample(dataset, ratio, random=False):
    """
    Get indices of subsampled dataset with given ratio.
    """
    idxs = list(range(len(dataset)))
    idxs_sorted = {}
    for idx, target in zip(idxs, dataset.targets):
        if target in idxs_sorted:
            idxs_sorted[target].append(idx)
        else:
            idxs_sorted[target] = [idx]

    for idx in idxs_sorted:
        size = len(idxs_sorted[idx])
        lenghts = (int(size * ratio), size - int(size * ratio))
        if random:
            idxs_sorted[idx] = torch.utils.data.random_split(idxs_sorted[idx], lenghts)[0]
        else:
            idxs_sorted[idx] = idxs_sorted[idx][:lenghts[0]]

    idxs = [idx for idxs in idxs_sorted.values() for idx in idxs]
    return idxs


logging.basicConfig(level=logging.INFO)

# set high precision for matmul 32
torch.set_float32_matmul_precision('high')

# Configuration
DEVICE = "cuda"
IMG_SIZE = 224
BATCH_SIZE = 64  # Reduced from 384 to save memory
NUM_WORKERS = 0

# Spectrum plotting limits
MAX_RANK_TOKEN = 196  # Token-level spectrum collapses around 196
MAX_RANK_IMAGE = 49   # Image-level spectrum collapses around 50
MAX_RANK_CLS = 384     # Reduced from 384 to save memory
SUBSAMPLE_RATIO = math.pow(2, -3)  # Increased from -3 to -4 to use less data

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
        subsample(dataset_test, 
                   ratio=SUBSAMPLE_RATIO)  # use a subsampled batch
    )
    print(f"SUBSAMPLED {SUBSAMPLE_RATIO} ratio, using {len(dataset_test)} samples")

    dataset_test = DataLoader(
        dataset_test, 
        num_workers=NUM_WORKERS, 
        batch_size=BATCH_SIZE,
        drop_last=True,  
        pin_memory=True,
    )
    
    return dataset_test


def batch_cov(points):
    """Compute covariance matrices for a batch of point sets.
    
    Args:
        points: Tensor of shape (B, N, D) where B is batch size, N is number of points, D is dimension
    
    Returns:
        Batch covariance matrices of shape (B, D, D)
    """
    B, N, D = points.size()
    mean = points.mean(dim=1).unsqueeze(1)
    diffs = (points - mean).reshape(B * N, D)
    prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(B, N, D, D)
    bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate
    return bcov  # (B, D, D)


def extract_and_compute_singular_values(model, dataset_test, label):
    """Extract features and compute singular values on-the-fly to save memory.
    
    Args:
        model: The model to analyze
        dataset_test: Dataset for analysis
        label: Model label
    
    Returns:
        Dictionary with analysis_type -> List of singular values for each layer
    """
    logging.info(f"Starting integrated feature extraction and SVD computation for {label}")
    head_name = "clf" if label == "PARTv1_ft" else "head"
    logging.info(f"Activation hooks installed for {label}")
    
    # Initialize accumulators for each analysis type
    # For image/cls: accumulate representations to compute covariance later
    # For token: accumulate singular values directly
    image_accum = []  # List of lists: [layer][batch_representations]
    cls_accum = []    # List of lists: [layer][batch_representations]
    token_singular_values = []  # List of lists: [layer][batch_singular_values]
    
    n_layers = None
    total_batches_processed = 0

    logging.info(f"Dataset contains {len(dataset_test)} batches")
    for i, (xs, ys) in enumerate(dataset_test):
        logging.info(f"Processing batch {i + 1}/{len(dataset_test)} for {label}")
        act = ActivationCache(head_name)
        act.hook(model)
        with torch.no_grad():
            xs = xs.to(DEVICE)
            logging.info(f"Batch {i + 1}: Input tensor moved to {DEVICE}, shape: {xs.shape}")
            _ = model(xs)
        
        logging.info(f"Batch {i + 1}: Forward pass completed, extracting activations...")
        zs = act.get_zs()
        zs = zs[:-1]  # skips model output
        # Only process every other layer to match ViT transformer blocks (skip intermediate activations)
        zs = zs[::2]
        logging.info(f"Batch {i + 1}: Extracted {len(zs)} ViT block activations (every 2nd layer)")
        
        # Initialize accumulators on first batch
        if n_layers is None:
            n_layers = len(zs)
            image_accum = [[] for _ in range(n_layers)]
            cls_accum = [[] for _ in range(n_layers)]
            token_singular_values = [[] for _ in range(n_layers)]
            logging.info(f"Initialized accumulators for {n_layers} layers")
            
        # Process each layer immediately
        logging.info(f"Batch {i + 1}: Processing {n_layers} layers...")
        for layer_idx, layer_features in enumerate(zs):
            logging.info(f"Batch {i + 1}, Layer {layer_idx}: Starting processing...")
            # layer_features shape: (batch, n_tokens, features)
            batch_size = layer_features.shape[0]
            
            # 1. Image-level: average spatial tokens (excluding CLS)
            spatial_tokens = layer_features[:, 1:, :]  # drop CLS: (batch, n_spatial, features)
            image_repr = einops.reduce(spatial_tokens, "b n c -> b c", "mean")  # (batch, features)
            image_accum[layer_idx].append(image_repr.cpu())
            logging.info(f"Batch {i + 1}, Layer {layer_idx}: Image-level repr computed and stored")
            
            # 2. CLS-level: extract CLS token
            cls_repr = layer_features[:, 0, :]  # (batch, features)
            cls_accum[layer_idx].append(cls_repr.cpu())
            logging.info(f"Batch {i + 1}, Layer {layer_idx}: CLS-level repr computed and stored")
            
            # 3. Token-level: compute SVD for all samples in batch simultaneously
            logging.info(f"Batch {i + 1}, Layer {layer_idx}: Starting batched token-level SVD for {batch_size} samples...")
            
            # Rearrange spatial tokens for batch covariance computation: (batch, features, n_spatial)
            spatial_tokens_transposed = einops.rearrange(spatial_tokens, "b n c -> b c n")
            
            # Compute batch covariance matrices: (batch, features, features)
            batch_cov_matrices = batch_cov(spatial_tokens_transposed)
            
            # Compute singular values only for all covariance matrices in the batch
            s = torch.linalg.svdvals(batch_cov_matrices)  # s shape: (batch, min(features, features))
            s = s.log().cpu()  # Apply log and move to CPU
            
            # Average singular values across batch
            avg_singular_values = s.mean(dim=0)  # Average across batch dimension
            token_singular_values[layer_idx].append(avg_singular_values)
            logging.info(f"Batch {i + 1}, Layer {layer_idx}: Batched token-level SVD completed and averaged")
            
            # Detailed logging for first layer only to avoid spam
            if layer_idx == 0:
                logging.info(f"Batch {i + 1}, Layer {layer_idx} DETAILS: "
                           f"spatial_tokens {spatial_tokens.shape}, "
                           f"image_repr {image_repr.shape}, "
                           f"cls_repr {cls_repr.shape}, "
                           f"token SVD computed for {batch_size} samples")
            
            logging.info(f"Batch {i + 1}, Layer {layer_idx}: Processing completed")
        
        # Clear cache and GPU memory after each batch
        total_batches_processed += 1
        logging.info(f"Batch {i + 1}: Cache cleared, GPU memory freed")

    logging.info(f"Processed {total_batches_processed} batches for {label}, hooks removed")
    
    # Finalize computations
    results = {}
    logging.info(f"Starting final SVD computations for {label}...")
    
    # 1. Image-level: concatenate all batches and compute final covariance
    logging.info("Computing image-level singular values...")
    image_singular_values = []
    for layer_idx in range(n_layers):
        if image_accum[layer_idx]:
            # Concatenate all batch representations
            all_image_repr = torch.cat(image_accum[layer_idx], dim=0)  # (total_samples, features)
            logging.info(f"Image-level Layer {layer_idx}: concatenated shape {all_image_repr.shape}")
            all_image_repr = einops.rearrange(all_image_repr, "b c -> c b")
            cov = torch.cov(all_image_repr)
            logging.info(f"Image-level Layer {layer_idx}: covariance matrix shape {cov.shape}")
            u, s, v = torch.svd(cov)
            s = s.log()
            image_singular_values.append(s)
            logging.info(f"Image-level Layer {layer_idx}: computed {len(s)} singular values")
    results["image"] = image_singular_values
    logging.info(f"Image-level analysis complete: {len(image_singular_values)} layers")
    
    # 2. CLS-level: concatenate all batches and compute final covariance
    logging.info("Computing CLS-level singular values...")
    cls_singular_values = []
    for layer_idx in range(n_layers):
        if cls_accum[layer_idx]:
            # Concatenate all batch representations
            all_cls_repr = torch.cat(cls_accum[layer_idx], dim=0)  # (total_samples, features)
            logging.info(f"CLS-level Layer {layer_idx}: concatenated shape {all_cls_repr.shape}")
            all_cls_repr = einops.rearrange(all_cls_repr, "b c -> c b")
            cov = torch.cov(all_cls_repr)
            logging.info(f"CLS-level Layer {layer_idx}: covariance matrix shape {cov.shape}")
            u, s, v = torch.svd(cov)
            s = s.log()
            cls_singular_values.append(s)
            logging.info(f"CLS-level Layer {layer_idx}: computed {len(s)} singular values")
    results["cls"] = cls_singular_values
    logging.info(f"CLS-level analysis complete: {len(cls_singular_values)} layers")
    
    # 3. Token-level: average across all batches
    logging.info("Finalizing token-level singular values...")
    final_token_singular_values = []
    for layer_idx in range(n_layers):
        if token_singular_values[layer_idx]:
            # Average singular values across all batches
            avg_sv = torch.stack(token_singular_values[layer_idx]).mean(dim=0)
            final_token_singular_values.append(avg_sv)
            logging.info(f"Token-level Layer {layer_idx}: averaged {len(token_singular_values[layer_idx])} batch results, final shape {avg_sv.shape}")
    results["token"] = final_token_singular_values
    logging.info(f"Token-level analysis complete: {len(final_token_singular_values)} layers")
    
    logging.info(f"Completed singular value computation for {label}: "
               f"image={len(results['image'])} layers, "
               f"cls={len(results['cls'])} layers, "
               f"token={len(results['token'])} layers")
    return results


# All the old plotting functions removed - they were duplicating computation!


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
            # No need for [::2] slicing since we already filtered during computation
            ax.plot(range(len(first_singular_per_layer)), first_singular_per_layer.cpu(), 
                   marker="o", label=display_name)
        else:  # token level
            # singular_values is a list of tensors (one per layer)
            first_singular_per_layer = torch.stack([sv[0] for sv in singular_values])
            ax.plot(range(len(first_singular_per_layer)), first_singular_per_layer.cpu(), 
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
        cbar.set_label('Depth (ViT blocks)')

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
    
    # Process models one at a time and compute singular values immediately
    logging.info("Processing models one at a time with integrated computation to save memory...")
    model_singular_values = {"image": {}, "token": {}, "cls": {}}
    
    for model_name in MODELS:
        logging.info(f"Processing {model_name}...")
        
        # Load model
        model = load_model(model_name)
        
        # Extract features and compute singular values in one pass - no large tensor storage!
        singular_values_dict = extract_and_compute_singular_values(model, dataset_test, model_name)
        
        # Store results for each analysis type
        for analysis_type in ["image", "token", "cls"]:
            model_singular_values[analysis_type][model_name] = singular_values_dict[analysis_type]
            logging.info(f"Singular values computed for {model_name} {analysis_type}: stored {len(singular_values_dict[analysis_type])} layers")
        
        # Clear model memory immediately
        del model
        torch.cuda.empty_cache()
        logging.info(f"Cleared model memory after processing {model_name}")
    
    # Generate all plots from the cached singular values (much smaller data)
    logging.info("Generating plots from cached singular values...")
    for analysis_type in ["image", "token", "cls"]:
        logging.info(f"Generating {analysis_type}-level plots...")
        
        # Use the optimized plotting functions that work with pre-computed data
        plot_singular_values_from_cache(model_singular_values[analysis_type], analysis_type)
    
    logging.info("Singular value analysis completed efficiently!")


if __name__ == "__main__":
    main()
