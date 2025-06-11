import torch
import math
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


def get_1d_sincos_pos_embed(
    positions: Float[Tensor, " N"], embed_dim: int
) -> torch.Tensor:
    assert embed_dim % 2 == 0, "Embedding dimension must be even."
    device = positions.device
    half_dim = embed_dim // 2
    dim_range = torch.arange(half_dim, dtype=torch.float32, device=device)
    # Same as before: frequencies based on embed_dim
    div_term = torch.exp(-math.log(10000.0) * (2 * dim_range / embed_dim))
    angles = positions.unsqueeze(1) * div_term.unsqueeze(0)  # shape (N, half_dim)
    emb_sin = torch.sin(angles)  # shape (N, half_dim)
    emb_cos = torch.cos(angles)  # shape (N, half_dim)
    # Concatenate sin first, then cos (not interleaved)
    pos_embed = torch.cat([emb_sin, emb_cos], dim=1)  # shape (N, embed_dim)
    return pos_embed


def get_2d_sincos_pos_embed(
    coords: Float[torch.Tensor, "N 2"], embed_dim: int
) -> Float[torch.Tensor, "N D"]:
    # coords: (N, 2) -> returns (N, embed_dim)
    assert embed_dim % 2 == 0, "Total embedding dimension must be even."
    embed_dim_half = embed_dim // 2
    y_embed = get_1d_sincos_pos_embed(coords[:, 0], embed_dim_half)
    x_embed = get_1d_sincos_pos_embed(coords[:, 1], embed_dim_half)
    return torch.cat([x_embed, y_embed], dim=1)


def get_canonical_coords(
    grid_size: tuple, patch_size: int, device: str = "cpu"
) -> Float[torch.Tensor, "N 2"]:
    """
    Returns canonical coordinates in pixel space.
    (0, 0), (0, P), (0, 2P), ..., (0, (W-1)P)
    (P, 0), (P, P), (P, 2P), ..., (P, (W-1)P)
    ...
    ((H-1)P, 0), ((H-1)P, P), ..., ((H-1)P, (W-1)P)

    Note: To use this function with sin-cos positional embeddings,
        divide the output by patch_size.
    """
    kwargs = {"dtype": torch.float32, "device": device}
    y_coords = torch.arange(grid_size[0], **kwargs) * patch_size
    x_coords = torch.arange(grid_size[1], **kwargs) * patch_size
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    grid_positions = torch.stack([grid_y, grid_x], dim=-1).view(-1, 2)
    return grid_positions


def get_canonical_pos_embed(
    embed_dim: int,
    grid_size: tuple = (14, 14),
    patch_size: int = 16,
    device: str = "cpu",
    # ) -> Float[torch.Tensor, "1 D H W"]:
) -> Float[torch.Tensor, "1 N D"]:
    # Returns canonical embeddings of shape (1, embed_dim, H, W)
    grid_positions = get_canonical_coords(grid_size, patch_size, device) / patch_size
    pos_embed = get_2d_sincos_pos_embed(grid_positions, embed_dim)
    # pos_embed = pos_embed.view(*grid_size, embed_dim).permute(2, 0, 1).unsqueeze(0)
    pos_embed = pos_embed.unsqueeze(0)
    return pos_embed

def get_sincos_pos_embed(
    embed_dim: int,
    grid_size: tuple = (14, 14),
    patch_size: int = 16,
    device: str = "cpu",
) -> Float[torch.Tensor, "1 N+1 D"]:
    """
    Returns sin-cos positional embeddings of shape (1, embed_dim, H, W).
    """
    pos_embed = get_canonical_pos_embed(embed_dim, grid_size, patch_size, device)
    cls_pos_embed = torch.zeros(1, 1, embed_dim, device=device)
    return torch.cat([cls_pos_embed, pos_embed], dim=1)  # [1, N+1, D]


def interpolate_grid_sample(
    grid: torch.Tensor,   # [D, H, W]
    coords: torch.Tensor  # [N, 2], (y, x) in pixel coords
) -> torch.Tensor:       # returns [N, D]
    """
    Reference using torch.grid_sample with align_corners=True.
    """
    D, H, W = grid.shape
    # prep as (1,C,H,W)
    inp = grid.unsqueeze(0)  # [1, D, H, W]

    # normalize coords to [-1,1]
    y, x = coords.unbind(-1)
    x = x.div(W - 1).mul(2).sub(1)
    y = y.div(H - 1).mul(2).sub(1)
    # grid_sample expects (N, out_H, out_W, 2)
    samp = torch.stack([x, y], dim=-1).view(1, coords.size(0), 1, 2)
    out = F.grid_sample(inp, samp, mode="bilinear", align_corners=True)
    # out: [1, D, N, 1] → [N, D]
    return out[0, :, :, 0].permute(1, 0)               # → [N, D]



def interpolate_pos_embed(
    pos_embed: Float[torch.Tensor, "1 D H W"],
    offgrid_coords: Float[torch.Tensor, "N 2"],
    patch_size: int,
) -> Float[torch.Tensor, "N D"]:
    # ...existing interpolation logic...
    _, _, H, W = pos_embed.shape
    max_x = patch_size * (W - 1)
    max_y = patch_size * (H - 1)
    norm_coords = offgrid_coords.clone()
    norm_coords[:, 0] = norm_coords[:, 0] / max_x * 2 - 1
    norm_coords[:, 1] = norm_coords[:, 1] / max_y * 2 - 1
    grid = norm_coords.view(1, -1, 1, 2)
    sampled = F.grid_sample(pos_embed, grid, align_corners=True)
    sampled = sampled.squeeze(-1).squeeze(0).transpose(0, 1)
    return sampled
