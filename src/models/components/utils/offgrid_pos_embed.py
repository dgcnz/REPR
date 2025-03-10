import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int  # type: ignore
from torch import Tensor


def get_1d_sincos_pos_embed(
    positions: Float[Tensor, "N"], embed_dim: int
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
    grid_positions = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)
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


class PosEmbedDynamicDiff(nn.Module):  # renamed from PosEmbedOption1Diff
    """
    Computes 2D sinusoidal positional embeddings on-the-fly for each batch element
    using different coordinates (heterogeneous batch).
    """

    def __init__(
        self,
        embed_dim: int,
    ):
        super().__init__()
        assert embed_dim % 2 == 0, "Embedding dimension must be even."
        self.embed_dim = embed_dim

    def forward(
        self, x: Float[torch.Tensor, "B N D"], coords: Float[torch.Tensor, "B N 2"]
    ) -> Float[torch.Tensor, "B N D"]:
        """
        Args:
            x      : Tensor of shape (B, N, embed_dim)
            coords : Tensor of shape (B, N, 2) containing positional coordinates.
        Returns:
            Tensor of shape (B, N, embed_dim) after adding positional embeddings.
        """
        B, N, _ = coords.shape
        # NOTE: needs to be divided by patch_size
        pos_embed = get_2d_sincos_pos_embed(coords.flatten(0, 1), self.embed_dim)
        pos_embed = pos_embed.unflatten(0, (B, N))
        return x + pos_embed


class PosEmbedPrecomputedDiff(nn.Module):
    """
    Optimized dynamic positional embedding implementation for the case where
    off-grid positions are given as integer indices in pixel space.
    It precomputes a high-resolution sinusoidal table (e.g. resolution=224) and,
    in forward, directly indexes into this table using advanced indexing rather than interpolation.
    """

    def __init__(self, embed_dim: int, resolution: int = 224):
        super().__init__()
        assert embed_dim % 2 == 0, "Embedding dimension must be even."
        self.embed_dim = embed_dim
        self.resolution = resolution
        # Precompute the full table.
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(resolution, dtype=torch.float32),
                torch.arange(resolution, dtype=torch.float32),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(
            -1, 2
        )  # shape: (resolution*resolution, 2)
        pos_embed = get_2d_sincos_pos_embed(
            coords, embed_dim
        )  # shape: (resolution*resolution, embed_dim)
        # Reshape to (1, embed_dim, resolution, resolution)
        pos_table = (
            pos_embed.view(resolution, resolution, embed_dim)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        self.register_buffer("pos_table", pos_table)

    def forward(self, x: torch.Tensor, offgrid_coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, N, embed_dim) to which positional embeddings are added.
            offgrid_coords: Tensor of shape (B, N, 2) with integer pixel indices in [0, resolution-1].
        Returns:
            Tensor of shape (B, N, embed_dim)
        """
        B, N, _ = offgrid_coords.shape
        # Ensure indices are of type long.
        # offgrid_coords = offgrid_coords.int()
        # Offgrid_coords are assumed to be in order: (x, y)
        x_idx = offgrid_coords[..., 0]  # (B, N)
        y_idx = offgrid_coords[..., 1]  # (B, N)
        # Expand pos_table to have batch dimension: (B, D, resolution, resolution)
        pos_table = self.pos_table.expand(B, -1, -1, -1)
        # Create a batch index tensor.
        batch_idx = (
            torch.arange(B, device=offgrid_coords.device).view(B, 1).expand(B, N)
        )
        # Directly index: for each b,n, pos[b, n, :] = pos_table[b, :, y_idx[b,n], x_idx[b,n]]
        pos_interp = pos_table[batch_idx, :, y_idx, x_idx]  # shape: (B, N, embed_dim)
        return x + pos_interp


class PosEmbedDynamicDiffOptimizedDiscreteV2(nn.Module):
    """
    Optimized dynamic positional embedding implementation for the case where off-grid positions
    are given as integer indices in pixel space. This implementation precomputes a high-resolution
    sinusoidal table and uses torch.gather to directly retrieve embeddings.

    The table is of shape (1, embed_dim, resolution, resolution) and offgrid_coords should be of shape
    (B, N, 2) with integer values in the range [0, resolution - 1].
    """

    def __init__(self, embed_dim: int, resolution: int = 224):
        super().__init__()
        assert embed_dim % 2 == 0, "Embedding dimension must be even."
        self.embed_dim = embed_dim
        self.resolution = resolution
        # Precompute full table for all pixel positions.
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(resolution, dtype=torch.float32),
                torch.arange(resolution, dtype=torch.float32),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(
            -1, 2
        )  # shape: (resolution*resolution, 2)
        pos_embed = get_2d_sincos_pos_embed(
            coords, embed_dim
        )  # shape: (resolution*resolution, embed_dim)
        # Reshape to table of shape (1, embed_dim, resolution, resolution)
        pos_table = (
            pos_embed.view(resolution, resolution, embed_dim)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        self.register_buffer("pos_table", pos_table)

    def forward(self, x: Tensor, offgrid_coords: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape (B, N, embed_dim) to which positional embeddings will be added.
            offgrid_coords: Tensor of shape (B, N, 2) with integer pixel indices in [0, resolution-1].
        Returns:
            Tensor of shape (B, N, embed_dim)
        """
        B, N, _ = offgrid_coords.shape
        # Ensure offgrid_coords are long.
        offgrid_coords = offgrid_coords.long()  # expected shape: (B, N, 2)
        # Get table: shape (B, D, resolution, resolution)
        pos_table = self.pos_table.expand(B, -1, -1, -1)
        D, H, W = pos_table.shape[1], pos_table.shape[2], pos_table.shape[3]
        # Flatten spatial dimensions: shape (B, D, H*W)
        pos_flat = pos_table.reshape(B, D, H * W)
        # Compute linear indices: linear_idx = y * W + x, shape (B, N)
        linear_idx = offgrid_coords[..., 1] * W + offgrid_coords[..., 0]  # (B, N)
        # Expand indices to shape (B, D, N) for gather.
        linear_idx_exp = linear_idx.unsqueeze(1).expand(B, D, N)
        # Use torch.gather to fetch embeddings: result has shape (B, D, N)
        gathered = pos_flat.gather(2, linear_idx_exp)
        # Transpose to (B, N, D)
        pos_interp = gathered.transpose(1, 2)
        return x + pos_interp


class PosEmbedStaticDiff(nn.Module):  # renamed from PosEmbedOption2Diff
    """
    Computes 2D positional embeddings via grid interpolation from a precomputed canonical grid,
    for heterogeneous batch coordinates.
    """

    def __init__(
        self, embed_dim: int, patch_size: int = 16, grid_size: tuple = (14, 14)
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.grid_size = grid_size  # (H, W)
        # Precompute the canonical grid and register it as a buffer.
        pos_embed = get_canonical_pos_embed(embed_dim, grid_size, patch_size)
        self.register_buffer("pos_embed", pos_embed)

    def forward(
        self, x: Float[torch.Tensor, "B N D"], coords: Float[torch.Tensor, "B N 2"]
    ) -> Float[torch.Tensor, "B N D"]:
        B, N, _ = x.shape
        # Reshape self.pos_embed from (1, N, D) to (1, D, H, W)
        H, W = self.grid_size
        canonical = self.pos_embed.reshape(1, H, W, -1).permute(0, 3, 1, 2)
        coords_flat = coords.reshape(-1, 2)
        pos_embed_flat = interpolate_pos_embed(canonical, coords_flat, self.patch_size)
        pos_embed = pos_embed_flat.view(B, N, self.embed_dim)
        return x + pos_embed


class PosEmbedDynamicSame(nn.Module):  # renamed from PosEmbedOption1Same
    """
    Computes 2D sinusoidal positional embeddings on-the-fly for SAME (constant) coordinates.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        assert embed_dim % 2 == 0, "Embedding dimension must be even."
        self.embed_dim = embed_dim

    def forward(
        self, x: Float[torch.Tensor, "B N D"], coords: Float[torch.Tensor, "N 2"]
    ) -> Float[torch.Tensor, "B N D"]:
        # Always compute new pos embeddings
        pos_embed = get_2d_sincos_pos_embed(coords, self.embed_dim)
        pos_embed = pos_embed.unsqueeze(0).expand(x.size(0), -1, -1)
        return x + pos_embed


class PosEmbedStaticSame(nn.Module):  # renamed from PosEmbedOption2Same
    """
    Computes 2D positional embeddings via grid interpolation for SAME (constant) coordinates.
    """

    def __init__(
        self, embed_dim: int, patch_size: int = 16, grid_size: tuple = (14, 14)
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.grid_size = grid_size  # (H, W)
        # Precompute the canonical grid and register it as a buffer.
        pos_embed = get_canonical_pos_embed(embed_dim, grid_size, patch_size)
        self.register_buffer("pos_embed", pos_embed)

    def forward(
        self, x: Float[Tensor, "B N D"], coords: Float[Tensor, "N 2"]
    ) -> torch.Tensor:
        B, N, _ = x.shape
        H, W = self.grid_size
        # Reshape the precomputed pos_embed to shape (1, D, H, W)
        canonical = self.pos_embed.reshape(1, H, W, -1).permute(0, 3, 1, 2)
        # Interpolate using the same coordinates for all batches
        pos_embed_flat = interpolate_pos_embed(canonical, coords, self.patch_size)
        # Expand to all batches
        pos_embed = pos_embed_flat.unsqueeze(0).expand(B, -1, -1)
        return x + pos_embed


class PosEmbedBaseline(nn.Module):
    """
    Standard positional embeddings, precomputed and added to input.
    """

    def __init__(
        self, embed_dim: int, patch_size: int = 16, grid_size: tuple = (14, 14)
    ):
        super().__init__()
        self.embed_dim = embed_dim
        # init with get_2d_sincos_pos_embed
        self.register_buffer(
            "pos_embed", get_canonical_pos_embed(embed_dim, grid_size, patch_size)
        )

    def forward(self, x: Float[torch.Tensor, "B N D"]) -> Float[torch.Tensor, "B N D"]:
        return x + self.pos_embed.expand(x.size(0), -1, -1)
