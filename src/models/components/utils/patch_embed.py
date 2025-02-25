import torch
from torch import nn, Tensor
import math
from jaxtyping import Int, Float
from typing import Union, Tuple
from torch.nn.modules.utils import _pair


def stratified_jittered_sampling_same(
    B: int,
    H: int,
    W: int,
    patch_size: int,
    N_vis: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """
    Generate stratified jittered patch positions in a vectorized manner.

    Args:
        B: Batch size.
        H, W: Image height and width.
        patch_size: Size of each patch.
        N_vis: Number of visible patches to sample.
        device: Torch device.

    Returns:
        ys, xs: Tensors of shape [B, N_vis] with integer top-left coordinates.
    """
    valid_H = H - patch_size  # maximum valid y coordinate
    valid_W = W - patch_size  # maximum valid x coordinate

    # Determine grid dimensions for stratification.
    # We choose grid_rows such that grid_rows = floor(sqrt(N_vis)),
    # and grid_cols = ceil(N_vis / grid_rows) so that grid_rows * grid_cols >= N_vis.
    grid_rows = int(math.floor(math.sqrt(N_vis)))
    grid_cols = int(math.ceil(N_vis / grid_rows))
    cell_h = valid_H / grid_rows
    cell_w = valid_W / grid_cols

    # Create a meshgrid of cell indices.
    row_idx = (
        torch.arange(grid_rows, device=device).float().unsqueeze(1)
    )  # shape [grid_rows, 1]
    col_idx = (
        torch.arange(grid_cols, device=device).float().unsqueeze(0)
    )  # shape [1, grid_cols]

    # Generate random jitter for each cell.
    rand_y = torch.rand((grid_rows, grid_cols), device=device)
    rand_x = torch.rand((grid_rows, grid_cols), device=device)

    # Compute jittered coordinates within each cell.
    ys_grid = (row_idx + rand_y) * cell_h  # shape [grid_rows, grid_cols]
    xs_grid = (col_idx + rand_x) * cell_w  # shape [grid_rows, grid_cols]

    # Clamp coordinates to the valid range.
    ys_grid = torch.clamp(ys_grid, max=valid_H)
    xs_grid = torch.clamp(xs_grid, max=valid_W)

    # Flatten the grid.
    ys_flat = ys_grid.flatten()  # shape [grid_rows * grid_cols]
    xs_flat = xs_grid.flatten()  # shape [grid_rows * grid_cols]
    # equals grid_rows * grid_cols, which is >= N_vis
    total_candidates = ys_flat.shape[0]

    # Always select N_vis indices without needing a conditional.
    perm = torch.randperm(total_candidates, device=device)[:N_vis]
    ys_sampled = ys_flat[perm].long()
    xs_sampled = xs_flat[perm].long()

    # Expand to batch size.
    ys_sampled = ys_sampled.unsqueeze(0).expand(B, -1)
    xs_sampled = xs_sampled.unsqueeze(0).expand(B, -1)
    return ys_sampled, xs_sampled


def stratified_jittered_sampling(
    B: int,
    H: int,
    W: int,
    patch_size: int,
    N_vis: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """
    Generate stratified jittered patch positions independently for each batch item.

    For each batch item, we:
      1. Divide the valid region ([0, H - patch_size] x [0, W - patch_size]) into a grid.
      2. Add random jitter to the grid cell centers.
      3. Flatten the candidate grid to get a list of candidate coordinates.
      4. For each batch item, select N_vis indices from its candidate grid.

    This yields a tensor of y coordinates and a tensor of x coordinates of shape [B, N_vis]
    with different values for each batch item.
    """
    valid_H = H - patch_size  # maximum valid top-left y
    valid_W = W - patch_size  # maximum valid top-left x

    # Determine grid dimensions such that grid_rows * grid_cols >= N_vis.
    grid_rows = int(math.floor(math.sqrt(N_vis)))
    grid_cols = int(math.ceil(N_vis / grid_rows))
    cell_h = valid_H / grid_rows
    cell_w = valid_W / grid_cols

    # Create a candidate grid per batch.
    # row_idx: shape [B, grid_rows, grid_cols]
    row_idx = (
        torch.arange(grid_rows, device=device)
        .float()
        .unsqueeze(0)
        .unsqueeze(2)
        .expand(B, -1, grid_cols)
    )
    # col_idx: shape [B, grid_rows, grid_cols]
    col_idx = (
        torch.arange(grid_cols, device=device)
        .float()
        .unsqueeze(0)
        .unsqueeze(1)
        .expand(B, grid_rows, -1)
    )

    # Generate random jitter for each batch independently.
    rand_y = torch.rand(B, grid_rows, grid_cols, device=device)
    rand_x = torch.rand(B, grid_rows, grid_cols, device=device)

    # Compute jittered candidate coordinates.
    ys_grid = (row_idx + rand_y) * cell_h
    xs_grid = (col_idx + rand_x) * cell_w

    # Clamp to valid range.
    ys_grid = torch.clamp(ys_grid, max=valid_H)
    xs_grid = torch.clamp(xs_grid, max=valid_W)

    # Flatten candidate grid per batch: shape [B, total_candidates]
    ys_flat = ys_grid.reshape(B, -1)
    xs_flat = xs_grid.reshape(B, -1)
    total_candidates = ys_flat.size(1)  # equals grid_rows * grid_cols

    # For each batch element, randomly select N_vis candidate indices.
    # Generate random scores per batch and sort.
    random_scores = torch.rand(B, total_candidates, device=device)
    sorted_indices = random_scores.argsort(dim=1)
    # Select the first N_vis indices for each batch.
    sel_indices = sorted_indices[:, :N_vis]

    # Gather the coordinates for each batch element.
    ys_sampled = torch.gather(ys_flat, 1, sel_indices).long()
    xs_sampled = torch.gather(xs_flat, 1, sel_indices).long()
    return ys_sampled, xs_sampled


def ongrid_sampling(
    B: int,
    H: int,
    W: int,
    patch_size: int,
    N_vis: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate on-grid patch positions independently for each batch element.

    The function computes a full grid of top-left coordinates for patches,
    then for each batch element, it generates random scores for each candidate and
    selects the top N_vis indices. This yields tensors of shape [B, N_vis] with
    different coordinates for each batch element.

    Args:
        B: Batch size.
        H, W: Image dimensions.
        patch_size: Patch size (assumed square).
        N_vis: Number of visible patches to sample.
        device: Torch device.

    Returns:
        ys, xs: Tensors of shape [B, N_vis] with integer coordinates.
    """
    # Create full grid of valid top-left coordinates.
    grid_y = torch.arange(0, H - patch_size + 1, patch_size, device=device)
    grid_x = torch.arange(0, W - patch_size + 1, patch_size, device=device)
    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
    grid_y = grid_y.flatten()  # [total_patches]
    grid_x = grid_x.flatten()  # [total_patches]
    total_patches = grid_y.numel()

    # Ensure N_vis does not exceed the total number of patches.
    N_vis = min(N_vis, total_patches)

    # For each batch element, generate random scores for all candidates.
    random_scores = torch.rand(B, total_patches, device=device)
    # Sort to get indices, then select top N_vis candidates.
    _, perm = random_scores.sort(dim=1)
    perm = perm[:, :N_vis]  # shape: [B, N_vis]

    # Gather candidate coordinates for each batch element.
    ys_sampled = torch.gather(grid_y.unsqueeze(0).expand(B, -1), 1, perm)
    xs_sampled = torch.gather(grid_x.unsqueeze(0).expand(B, -1), 1, perm)

    return ys_sampled, xs_sampled


def random_sampling(
    B: int, H: int, W: int, patch_size: int, N_vis: int, device: torch.device
) -> tuple[Int[Tensor, "B N"], Int[Tensor, "B N"]]:
    xs = torch.randint(0, W - patch_size + 1, (B, N_vis), device=device)
    ys = torch.randint(0, H - patch_size + 1, (B, N_vis), device=device)
    return ys, xs


class OffGridPatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        mask_ratio: float = 0.0,
        sampler: callable = random_sampling,
    ) -> None:
        super().__init__()
        self.patch_size = _pair(patch_size)
        self.img_size = _pair(img_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Linear(patch_size * patch_size * in_chans, embed_dim)
        self.sampler = sampler
        self.mask_ratio = mask_ratio

    def _init_img_size(self, img_size: Union[int, Tuple[int, int]]):
        assert self.patch_size
        if img_size is None:
            return None, None, None
        img_size = _pair(img_size)
        grid_size = tuple([s // p for s, p in zip(img_size, self.patch_size)])
        num_patches = grid_size[0] * grid_size[1]
        return img_size, grid_size, num_patches

    def _create_sampled_grid_flattened(
        self,
        patch_size: int,
        H: int,
        W: int,
        ys: Int[Tensor, "B N"],
        xs: Int[Tensor, "B N"],
    ) -> Int[Tensor, "B N*P*P"]:
        B, n_patches = ys.size(0), ys.size(1)
        dy = torch.arange(patch_size, device=ys.device).unsqueeze(1)
        dx = torch.arange(patch_size, device=ys.device).unsqueeze(0)
        d = (dy * W + dx).view(-1)
        base = ys * W + xs
        indices = base.unsqueeze(-1) + d.unsqueeze(0)
        return indices.view(B, n_patches * patch_size * patch_size)

    def forward(
        self, x: Float[Tensor, "B C H W"]
    ) -> tuple[Float[Tensor, "B N D"], Int[Tensor, "B N 2"]]:
        """
        :param x: input tensor of shape [B, C, H, W]
        :return: patch_tokens of shape [B, N, D], patch_positions of shape [B, N, 2]
        """
        B, C, H, W = x.shape
        N = (H // self.patch_size[0]) * (W // self.patch_size[1])
        N_vis = int(N * (1 - self.mask_ratio))
        ys, xs = self.sampler(B, H, W, self.patch_size[0], N_vis, x.device)
        patch_positions = torch.stack([ys, xs], dim=-1)
        patch_area = self.patch_size[0] * self.patch_size[1]
        indices = self._create_sampled_grid_flattened(self.patch_size[0], H, W, ys, xs)
        indices = indices.unsqueeze(1).expand(B, C, -1)
        patches_flat = x.flatten(-2).gather(2, indices)
        patches_flat = patches_flat.view(B, C, N_vis, patch_area)
        patches_flat = patches_flat.permute(0, 2, 1, 3).reshape(
            B, N_vis, C * patch_area
        )
        patch_tokens = self.proj(patches_flat)
        return patch_tokens, patch_positions
