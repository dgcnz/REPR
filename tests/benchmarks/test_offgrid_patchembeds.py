import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from jaxtyping import Float, Int
import math


def to_2tuple(x: int) -> tuple[int, int]:
    return (x, x)

def random_sampling(
    B: int, H: int, W: int, patch_size: int, N_vis: int, device: torch.device
) -> tuple[Int[Tensor, "B N"], Int[Tensor, "B N"]]:
    xs = torch.randint(0, W - patch_size, (B, N_vis), device=device)
    ys = torch.randint(0, H - patch_size, (B, N_vis), device=device)
    return ys, xs

def stratified_jittered_sampling(
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
    row_idx = torch.arange(grid_rows, device=device).float().unsqueeze(1)  # shape [grid_rows, 1]
    col_idx = torch.arange(grid_cols, device=device).float().unsqueeze(0)   # shape [1, grid_cols]
    
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
    total_candidates = ys_flat.shape[0]  # equals grid_rows * grid_cols, which is >= N_vis

    # Always select N_vis indices without needing a conditional.
    perm = torch.randperm(total_candidates, device=device)[:N_vis]
    ys_sampled = ys_flat[perm].long()
    xs_sampled = xs_flat[perm].long()
    
    # Expand to batch size.
    ys_sampled = ys_sampled.unsqueeze(0).expand(B, -1)
    xs_sampled = xs_sampled.unsqueeze(0).expand(B, -1)
    return ys_sampled, xs_sampled


import torch
import math
from torch import Tensor

def stratified_jittered_sampling_per_batch(
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
    row_idx = torch.arange(grid_rows, device=device).float().unsqueeze(0).unsqueeze(2).expand(B, -1, grid_cols)
    # col_idx: shape [B, grid_rows, grid_cols]
    col_idx = torch.arange(grid_cols, device=device).float().unsqueeze(0).unsqueeze(1).expand(B, grid_rows, -1)
    
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



# -------------------------------
# Provided Patch Embedding Modules
# -------------------------------
class Conv2DPatchEmbed(nn.Module):
    """2D Image to Patch Embedding using Conv2d."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: nn.Module | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        img_size_tuple = to_2tuple(img_size)
        patch_size_tuple = to_2tuple(patch_size)
        self.img_size: tuple[int, int] = img_size_tuple
        self.patch_size: tuple[int, int] = patch_size_tuple
        self.grid_size: tuple[int, int] = (
            img_size_tuple[0] // patch_size_tuple[0],
            img_size_tuple[1] // patch_size_tuple[1],
        )
        self.num_patches: int = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size_tuple,
            stride=patch_size_tuple,
            bias=bias,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Float[Tensor, "B C H W"]) -> Float[Tensor, "B N D"]:
        x = self.proj(x)  # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2)  # -> [B, N, D]
        return self.norm(x)


class LinearPatchEmbed(nn.Module):
    """
    Linear patch embedding using nn.Unfold.
    Expects input patches of shape [B, num_patches, C*patch_size*patch_size].
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.patch_size: int = patch_size
        self.embed_dim: int = embed_dim
        self.num_patches: int = (img_size // patch_size) ** 2
        self.proj = nn.Linear(patch_size * patch_size * in_channels, embed_dim)

    def forward(self, x: Float[Tensor, "B N (C P P)"]) -> Float[Tensor, "B N D"]:
        return self.proj(x)


# -------------------------------
# Original Methods
# -------------------------------
class OffGridPatchEmbedStitching(nn.Module):
    """
    Off-grid sampling by stitching. Samples patches, restitches into an image,
    then applies Conv2DPatchEmbed.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        sampler: callable = random_sampling,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.patch_embed = Conv2DPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        # precompute d
        dy, dx = torch.arange(patch_size), torch.arange(patch_size)
        d = img_size * dy.unsqueeze(-1) + dx
        self.register_buffer("d", d)
        self.sampler = sampler


    def _create_sampled_grid_flattened(
        self,
        patch_size: int,
        H: int,
        W: int,
        ys: Int[Tensor, "B N"],
        xs: Int[Tensor, "B N"],
    ) -> Int[Tensor, "B (H W)"]:
        B, nH, nW = ys.size(0), H // patch_size, W // patch_size
        zs = ys * W + xs  # [B, N]
        # Create a 2D patch offset grid.
        d = self.d  # [P, P]
        indices = zs[:, :, None, None] + d  # [B, N, P, P]
        indices = indices.view(B, nH, nW, patch_size, patch_size)
        indices = indices.permute(0, 1, 3, 2, 4).reshape(B, H * W)
        return indices

    def sample_and_stitch(
        self, img: Float[Tensor, "B C H W"]
    ) -> tuple[Float[Tensor, "B C H W"], Int[Tensor, "B N 2"]]:
        B, C, H, W = img.shape
        num_patches = (H // self.patch_size) * (W // self.patch_size)
        ys, xs = self.sampler(B, H, W, self.patch_size, num_patches, img.device)
        patch_positions = torch.stack([ys, xs], dim=-1)
        indices = self._create_sampled_grid_flattened(self.patch_size, H, W, ys, xs)
        indices = indices.unsqueeze(1).expand(B, C, -1)
        stitched_img = img.flatten(-2).gather(2, indices).unflatten(-1, (H, W))
        return stitched_img, patch_positions

    def forward(
        self, x: Float[Tensor, "B C H W"]
    ) -> tuple[Float[Tensor, "B N D"], Int[Tensor, "B N 2"]]:
        stitched_img, patch_positions = self.sample_and_stitch(x)
        patch_tokens = self.patch_embed(stitched_img)
        return patch_tokens, patch_positions


class OffGridPatchEmbedGather(nn.Module):
    """
    Off-grid sampling via gather and linear projection.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        sampler: callable = random_sampling,
        mask_ratio: float = 0.0,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.patch_embed = LinearPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_chans,
            embed_dim=embed_dim,
        )
        self.sampler = sampler
        self.mask_ratio = mask_ratio


    def forward(
        self, x: Float[Tensor, "B C H W"]
    ) -> tuple[Float[Tensor, "B N D"], Int[Tensor, "B N 2"]]:
        B, C, H, W = x.shape
        N = (H // self.patch_size) * (W // self.patch_size)
        N_vis = int(N * (1 - self.mask_ratio))
        ys, xs = self.sampler(B, H, W, self.patch_size, N_vis, x.device)
        patch_positions = torch.stack([ys, xs], dim=-1)
        num_patches = ys.shape[1]
        device = x.device
        dy = torch.arange(self.patch_size, device=device).unsqueeze(1)
        dx = torch.arange(self.patch_size, device=device).unsqueeze(0)
        patch_delta = (dy * W + dx).view(-1)
        base_indices = ys * W + xs
        patch_indices = base_indices.unsqueeze(-1) + patch_delta.unsqueeze(0).unsqueeze(
            0
        )
        channel_offsets = torch.arange(C, device=device) * (H * W)
        patch_indices = patch_indices.unsqueeze(2) + channel_offsets.view(1, 1, C, 1)
        patch_indices = patch_indices.view(
            B, num_patches, C * self.patch_size * self.patch_size
        )
        img_flat = x.view(B, C * H * W)
        img_flat_expanded = img_flat.unsqueeze(1).expand(B, num_patches, -1)
        patches_flat = torch.gather(img_flat_expanded, 2, patch_indices)
        patch_tokens = self.patch_embed.proj(patches_flat)
        return patch_tokens, patch_positions


# -------------------------------
# Variant 1: Stitching with Linear Projection
# -------------------------------
class OffGridPatchEmbedStitchingLinear(nn.Module):
    """
    Variant: Gather flattened patches and use a linear projection.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        sampler: callable = random_sampling,
        mask_ratio: float = 0.0,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.patch_embed = LinearPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_chans,
            embed_dim=embed_dim,
        )
        self.sampler = sampler
        self.mask_ratio = mask_ratio


    def _create_sampled_grid_flattened(
        self,
        patch_size: int,
        H: int,
        W: int,
        ys: Int[Tensor, "B N"],
        xs: Int[Tensor, "B N"],
    ) -> Int[Tensor, "B (N*P*P)"]:
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
        B, C, H, W = x.shape
        N = (H // self.patch_size) * (W // self.patch_size)
        N_vis = int(N * (1 - self.mask_ratio))
        ys, xs = self.sampler(B, H, W, self.patch_size, N_vis, x.device)
        patch_positions = torch.stack([ys, xs], dim=-1)
        n_patches = ys.shape[1]
        patch_area = self.patch_size * self.patch_size
        indices = self._create_sampled_grid_flattened(self.patch_size, H, W, ys, xs)
        indices = indices.unsqueeze(1).expand(B, C, -1)
        patches_flat = x.flatten(-2).gather(2, indices)
        patches_flat = patches_flat.view(B, C, n_patches, patch_area)
        patches_flat = patches_flat.permute(0, 2, 1, 3).reshape(
            B, n_patches, C * patch_area
        )
        patch_tokens = self.patch_embed(patches_flat)
        return patch_tokens, patch_positions


# -------------------------------
# Variant 2: Gather with Conv2D Projection
# -------------------------------
class OffGridPatchEmbedGatherConv2D(nn.Module):
    """
    Variant: Gather patches and reshape into an image for Conv2D projection.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        sampler: callable = random_sampling,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.patch_embed = Conv2DPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.sampler = sampler


    def forward(
        self, x: Float[Tensor, "B C H W"]
    ) -> tuple[Float[Tensor, "B N D"], Int[Tensor, "B N 2"]]:
        B, C, H, W = x.shape
        N = (H // self.patch_size) * (W // self.patch_size)
        ys, xs = self.sampler(B, H, W, self.patch_size, N, x.device)
        patch_positions = torch.stack([ys, xs], dim=-1)
        num_patches = ys.shape[1]
        patch_area = self.patch_size * self.patch_size
        base_indices = ys * W + xs
        device = x.device
        dy = torch.arange(self.patch_size, device=device).unsqueeze(1)
        dx = torch.arange(self.patch_size, device=device).unsqueeze(0)
        patch_delta = (dy * W + dx).view(-1)
        patch_indices = base_indices.unsqueeze(-1) + patch_delta.unsqueeze(0).unsqueeze(
            0
        )
        channel_offsets = torch.arange(C, device=device) * (H * W)
        patch_indices = patch_indices.unsqueeze(2) + channel_offsets.view(1, 1, C, 1)
        patch_indices = patch_indices.view(B, num_patches, C * patch_area)
        img_flat = x.view(B, C * H * W)
        img_flat_expanded = img_flat.unsqueeze(1).expand(B, num_patches, -1)
        patches_flat = torch.gather(img_flat_expanded, 2, patch_indices)
        nH = H // self.patch_size
        nW = W // self.patch_size
        patches_img = patches_flat.view(B, nH, nW, C, patch_area)
        patches_img = patches_img.view(B, nH, nW, C, self.patch_size, self.patch_size)
        patches_img = patches_img.permute(0, 3, 1, 4, 2, 5).contiguous()
        restitched_img = patches_img.view(
            B, C, nH * self.patch_size, nW * self.patch_size
        )
        patch_tokens = self.patch_embed(restitched_img)
        return patch_tokens, patch_positions
    



# -------------------------------
# Pytest Tests
# -------------------------------
@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("img_size", [224])
@pytest.mark.parametrize("patch_size", [16])
@pytest.mark.parametrize("in_chans", [3])
@pytest.mark.parametrize("embed_dim", [768])
def test_offgrid_all_equivalence(
    B: int, img_size: int, patch_size: int, in_chans: int, embed_dim: int
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = torch.randn(B, in_chans, img_size, img_size, device=device)


    # Instantiate all four models.
    model_stitching = OffGridPatchEmbedStitching(
        img_size, patch_size, in_chans, embed_dim
    ).to(device)
    model_gather = OffGridPatchEmbedGather(
        img_size, patch_size, in_chans, embed_dim
    ).to(device)
    model_stitching_linear = OffGridPatchEmbedStitchingLinear(
        img_size, patch_size, in_chans, embed_dim
    ).to(device)
    model_gather_conv2d = OffGridPatchEmbedGatherConv2D(
        img_size, patch_size, in_chans, embed_dim
    ).to(device)

    # Force same weights.
    with torch.no_grad():
        weight_conv = model_stitching.patch_embed.proj.weight.data.clone()
        bias_conv = (
            model_stitching.patch_embed.proj.bias.data.clone()
            if model_stitching.patch_embed.proj.bias is not None
            else None
        )
        model_stitching.patch_embed.proj.weight.data.copy_(weight_conv)
        if bias_conv is not None:
            model_stitching.patch_embed.proj.bias.data.copy_(bias_conv)
        model_gather.patch_embed.proj.weight.data.copy_(weight_conv.view(embed_dim, -1))
        if bias_conv is not None:
            model_gather.patch_embed.proj.bias.data.copy_(bias_conv.view(-1))
        model_gather_conv2d.patch_embed.proj.weight.data.copy_(weight_conv)
        if bias_conv is not None:
            model_gather_conv2d.patch_embed.proj.bias.data.copy_(bias_conv)
        weight_lin = weight_conv.view(embed_dim, -1).clone()
        model_stitching_linear.patch_embed.proj.weight.data.copy_(weight_lin)
        if bias_conv is not None:
            model_stitching_linear.patch_embed.proj.bias.data.copy_(bias_conv.view(-1))

    torch.manual_seed(42)
    tokens1, pos1 = model_stitching(img)
    torch.manual_seed(42)
    tokens2, pos2 = model_gather(img)
    torch.manual_seed(42)
    tokens3, pos3 = model_stitching_linear(img)
    torch.manual_seed(42)
    tokens4, pos4 = model_gather_conv2d(img)

    for p in [pos1, pos2, pos3, pos4]:
        assert torch.allclose(
            pos1.float(), p.float(), atol=1e-6
        ), "Patch positions differ."

    tol = 1e-3
    assert torch.allclose(
        tokens1, tokens2, atol=tol
    ), f"Original methods differ; max diff: {(tokens1 - tokens2).abs().max().item()}"
    assert torch.allclose(
        tokens1, tokens4, atol=tol
    ), f"Stitching vs. gather_conv2d differ; max diff: {(tokens1 - tokens4).abs().max().item()}"
    assert torch.allclose(
        tokens1, tokens3, atol=tol
    ), f"Stitching vs. stitching_linear differ; max diff: {(tokens1 - tokens3).abs().max().item()}"


@pytest.mark.parametrize(
    "module_class, mask_ratio",
    [
        (OffGridPatchEmbedStitching, 0.0),
        (OffGridPatchEmbedGather, 0.0),
        (OffGridPatchEmbedGather, 0.75),
        (OffGridPatchEmbedGather, 0.5),
        (OffGridPatchEmbedStitchingLinear, 0.0),
        (OffGridPatchEmbedStitchingLinear, 0.75),
        (OffGridPatchEmbedStitchingLinear, 0.5),
        (OffGridPatchEmbedGatherConv2D, 0.0),
    ],
)
@pytest.mark.parametrize("sampler", [random_sampling, stratified_jittered_sampling, stratified_jittered_sampling_per_batch])
@pytest.mark.parametrize("compile", [False, True])
def test_offgrid_patch_embed_benchmark(
    module_class, mask_ratio, sampler, compile: bool, benchmark_v2
) -> None:
    B = 128
    img_size = 224
    patch_size = 16
    in_chans = 3
    embed_dim = 768
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(B, in_chans, img_size, img_size, device=device)

    kwargs = {"sampler": sampler}
    if mask_ratio:
        kwargs["mask_ratio"] = mask_ratio

    model = (
        module_class(img_size, patch_size, in_chans, embed_dim, **kwargs)
        .to(device)
        .eval()
    )
    if compile:
        model = torch.compile(model)

    def ffn(
        inputs: tuple[tuple[Tensor, ...], ...]
    ) -> tuple[Float[Tensor, "B N D"], Int[Tensor, "B N 2"]]:
        with torch.no_grad():
            return model(inputs[0])

    args_gen = lambda: ((x,),)
    benchmark_v2.benchmark(ffn, args_gen=args_gen, n_warmup=50, n_runs=1000)
    benchmark_v2.group_by(f"compile: {compile}, mask_ratio: {mask_ratio}")
    benchmark_v2.drop_columns(exclude=["time/min (ms)", "time/max (ms)"])
    benchmark_v2.sort_by("time/median (ms)")


# -------------------------------
# Pytest Test
# -------------------------------
@pytest.mark.parametrize("B", [16])
@pytest.mark.parametrize("img_size", [224])
@pytest.mark.parametrize("patch_size", [16])
@pytest.mark.parametrize("in_chans", [3])
@pytest.mark.parametrize("embed_dim", [768])
@pytest.mark.parametrize("sampler", [random_sampling])
def test_offgrid_all_equivalence(B, img_size, patch_size, in_chans, embed_dim, sampler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = torch.randn(B, in_chans, img_size, img_size, device=device)

    # Instantiate all four models.
    model_stitching = OffGridPatchEmbedStitching(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        sampler=sampler,
    ).to(device)
    model_gather = OffGridPatchEmbedGather(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        sampler=sampler,
    ).to(device)
    model_stitching_linear = OffGridPatchEmbedStitchingLinear(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        sampler=sampler,
    ).to(device)
    model_gather_conv2d = OffGridPatchEmbedGatherConv2D(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        sampler=sampler,
    ).to(device)

    # Force all projection layers to use the same weights.
    with torch.no_grad():
        # Use model_stitching (conv2d) as reference.
        weight_conv = model_stitching.patch_embed.proj.weight.data.clone()
        bias_conv = (
            model_stitching.patch_embed.proj.bias.data.clone()
            if model_stitching.patch_embed.proj.bias is not None
            else None
        )
        # For conv2d modules:
        model_stitching.patch_embed.proj.weight.data.copy_(weight_conv)
        if bias_conv is not None:
            model_stitching.patch_embed.proj.bias.data.copy_(bias_conv)
        # For the linear module, flatten the conv2d weight.
        model_gather.patch_embed.proj.weight.data.copy_(weight_conv.view(embed_dim, -1))
        if bias_conv is not None:
            model_gather.patch_embed.proj.bias.data.copy_(bias_conv.view(-1))
        model_gather_conv2d.patch_embed.proj.weight.data.copy_(weight_conv)
        if bias_conv is not None:
            model_gather_conv2d.patch_embed.proj.bias.data.copy_(bias_conv)
        # For linear modules: flatten conv2d weight.
        weight_lin = weight_conv.view(embed_dim, -1).clone()
        model_stitching_linear.patch_embed.proj.weight.data.copy_(weight_lin)
        if bias_conv is not None:
            model_stitching_linear.patch_embed.proj.bias.data.copy_(bias_conv.view(-1))

    # Run each model with the same seed so that patch sampling is identical.
    torch.manual_seed(42)
    tokens1, pos1 = model_stitching(img)
    torch.manual_seed(42)
    tokens2, pos2 = model_gather(img)
    torch.manual_seed(42)
    tokens3, pos3 = model_stitching_linear(img)
    torch.manual_seed(42)
    tokens4, pos4 = model_gather_conv2d(img)

    # Check that patch positions are identical across methods.
    for p in [pos1, pos2, pos3, pos4]:
        assert torch.allclose(
            pos1.float(), p.float(), atol=1e-6
        ), "Patch positions differ among methods."

    # Check that the token outputs are nearly identical.
    diff12 = (tokens1 - tokens2).abs().max().item()
    diff13 = (tokens1 - tokens3).abs().max().item()
    diff14 = (tokens1 - tokens4).abs().max().item()
    print("Max token difference between stitching and gather (original):", diff12)
    print(
        "Max token difference between stitching and stitching_linear (variant):", diff13
    )
    print("Max token difference between stitching and gather_conv2d (variant):", diff14)
    tol = 1e-3
    assert torch.allclose(
        tokens1, tokens2, atol=tol
    ), f"Original methods differ; max diff: {diff12}"
    assert torch.allclose(
        tokens1, tokens4, atol=tol
    ), f"Stitching vs. gather_conv2d differ; max diff: {diff14}"
    assert torch.allclose(
        tokens1, tokens3, atol=tol
    ), f"Stitching vs. stitching_linear differ; max diff: {diff13}"


if __name__ == "__main__":
    pytest.main([__file__])
