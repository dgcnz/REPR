import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

def _create_sampled_grid_flattened(patch_size: int, H: int, W: int, ys: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    """
    Create full indices for each patch specified by top-left corner coordinates (ys, xs).
    Returns indices of shape [B, H * W] that can be used with torch.gather.
    """
    B, nH, nW = ys.size(0), H // patch_size, W // patch_size
    zs = ys * W + xs  # base indices for each patch
    dx = dy = torch.arange(patch_size, device=ys.device)
    d = W * dy.unsqueeze(-1) + dx  # [patch_size, patch_size]
    indices = zs[:, :, None, None] + d  # [B, nH*nW, patch_size, patch_size]
    indices = indices.view(B, nH, nW, patch_size, patch_size)
    indices = indices.permute(0, 1, 3, 2, 4).reshape(B, H * W)
    return indices

def _sample_offgrid(B: int, H: int, W: int, patch_size: int, device: torch.device = torch.device("cpu")) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample random patches off-grid by generating top-left coordinates.
    Returns ys and xs of shape [B, nH*nW] where nH = H//patch_size and nW = W//patch_size.
    """
    nW = W // patch_size
    nH = H // patch_size
    xs = torch.randint(0, W - patch_size, (B, nH * nW), device=device)
    ys = torch.randint(0, H - patch_size, (B, nH * nW), device=device)
    return ys, xs

def sample_and_stitch(img: torch.Tensor, patch_size: int, mode: str = "offgrid") -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample patches from an image and stitch them back together.
    Returns:
        new_img: Restitched image of shape [B, C, H, W]
        patch_positions: tensor of shape [B, N, 2] containing (y, x) coordinates of each patch.
    """
    B, C, H, W = img.size()
    if mode == "offgrid":
        ys, xs = _sample_offgrid(B, H, W, patch_size, device=img.device)
    else:
        raise NotImplementedError("Only offgrid mode is implemented")
    patch_positions = torch.stack([ys, xs], dim=-1)
    indices = _create_sampled_grid_flattened(patch_size, H, W, ys, xs)
    indices = indices.unsqueeze(1).expand(-1, C, -1)
    new_img = img.flatten(-2).gather(2, indices).unflatten(-1, (H, W))
    return new_img, patch_positions

@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("C", [3])
@pytest.mark.parametrize("H, W", [(224, 224)])
@pytest.mark.parametrize("patch_size", [16])
def test_patch_embedding_equivalence(B, C, H, W, patch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_patches = (H // patch_size) * (W // patch_size)  # e.g. 196 for 224x224
    torch.manual_seed(42)
    
    # Create a random input image
    img = torch.randn(B, C, H, W, device=device)
    
    # --- Pipeline 1: Restitch then Conv2d Patch Embedding ---
    stitched_img, patch_positions = sample_and_stitch(img, patch_size, mode="offgrid")
    D = 768  # embedding dimension
    conv_patch_embed = nn.Conv2d(C, D, kernel_size=patch_size, stride=patch_size).to(device)
    conv_out = conv_patch_embed(stitched_img)  # [B, D, H/patch_size, W/patch_size]
    conv_out = conv_out.flatten(2).transpose(1, 2)  # [B, num_patches, D]
    
    # --- Pipeline 2: Gather Extraction then Flatten + MLP ---
    ys = patch_positions[..., 0]  # [B, num_patches]
    xs = patch_positions[..., 1]  # [B, num_patches]
    base_indices = ys * W + xs  # [B, num_patches]
    
    # Precompute patch delta indices for a patch of size patch_size x patch_size.
    dy = torch.arange(patch_size, device=device).unsqueeze(1)
    dx = torch.arange(patch_size, device=device).unsqueeze(0)
    patch_delta = (dy * W + dx).view(-1)  # [patch_size*patch_size]
    
    # Compute indices for all pixels in each patch: [B, num_patches, patch_size*patch_size]
    patch_indices = base_indices.unsqueeze(-1) + patch_delta.unsqueeze(0).unsqueeze(0)
    
    # Account for channel dimension: add offset for each channel.
    channel_offsets = torch.arange(C, device=device) * (H * W)  # [C]
    patch_indices = patch_indices.unsqueeze(2) + channel_offsets.view(1, 1, C, 1)
    patch_indices = patch_indices.view(B, num_patches, C * patch_size * patch_size)
    
    # Flatten the image: [B, C*H*W]
    img_flat = img.view(B, C * H * W)
    # Expand img_flat along a new dimension to match patch_indices shape: [B, num_patches, C*H*W]
    img_flat_expanded = img_flat.unsqueeze(1).expand(B, num_patches, -1)
    patches_flat = torch.gather(img_flat_expanded, 2, patch_indices)  # [B, num_patches, C*patch_size*patch_size]
    
    # Set up a linear projection with weights matching conv2d's kernel.
    with torch.no_grad():
        linear_weight = conv_patch_embed.weight.view(D, -1).clone()
        linear_bias = conv_patch_embed.bias.clone() if conv_patch_embed.bias is not None else None
    mlp_out = F.linear(patches_flat, linear_weight, linear_bias)  # [B, num_patches, D]
    
    diff = (conv_out - mlp_out).abs().max().item()
    print("Max difference between conv2d and flatten+MLP outputs:", diff)
    assert torch.allclose(conv_out, mlp_out, atol=1e-6), f"Max difference: {diff}"

if __name__ == "__main__":
    pytest.main([__file__])
