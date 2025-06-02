import torch
import torch.nn.functional as F
import pytest
from src.models.components.utils.offgrid_pos_embed import get_2d_sincos_pos_embed, get_canonical_pos_embed
from functools import partial

def interpolate_pos_embed(
    grid: torch.Tensor,    # [D, H, W]
    coords: torch.Tensor,  # [N, 2], (y, x) in pixel coords
) -> torch.Tensor:        # returns [N, D]
    """
    Fast bilinear interpolation by explicit gather + weighted sum.
    """
    D, H, W = grid.shape
    # 1) convert pixel coords → grid units
    u = coords.float() 
    y, x = u.unbind(-1)  # each [N]

    # 2) integer bounds
    y0 = y.floor().clamp(0, H - 1)
    x0 = x.floor().clamp(0, W - 1)
    y0i = y0.long()
    x0i = x0.long()
    y1i = (y0i + 1).clamp(max=H - 1)
    x1i = (x0i + 1).clamp(max=W - 1)

    # 3) fractional offsets
    dy = (y - y0).unsqueeze(1)  # [N,1]
    dx = (x - x0).unsqueeze(1)
    w00 = (1 - dy) * (1 - dx)   # [N,1]
    w01 = (1 - dy) * dx
    w10 = dy * (1 - dx)
    w11 = dy * dx

    # 4) gather the four neighbors from a flattened grid
    flat = grid.reshape(D, H * W)            # [D, H*W]
    idx00 = (y0i * W + x0i)                  # [N]
    idx01 = (y0i * W + x1i)
    idx10 = (y1i * W + x0i)
    idx11 = (y1i * W + x1i)
    v00 = flat[:, idx00]                     # [D, N]
    v01 = flat[:, idx01]
    v10 = flat[:, idx10]
    v11 = flat[:, idx11]

    # 5) weighted sum → [N, D]
    return (v00.T * w00 + v01.T * w01 + v10.T * w10 + v11.T * w11)

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

@pytest.mark.parametrize("seed", [0, 42, 123])
def test_interpolation_against_grid_sample(seed):
    torch.manual_seed(seed)
    # random grid dim
    D = 8
    H, W = 224, 224
    P = 16
    grid_size = (H // P, W // P)
    grid = torch.randn(D, *grid_size, dtype=torch.float32)

    # random float coords in [0, H-1] × [0, W-1]
    N = 300
    coords = torch.stack([
        torch.rand(N).mul(H - P),
        torch.rand(N).mul(W - P)
    ], dim=1)

    # patch_size = 1 for direct comparison
    emb_manual = interpolate_pos_embed(grid, coords.float() / P) 
    emb_ref    = interpolate_grid_sample(grid, coords.float() / P)

    # they should be numerically very close
    assert torch.allclose(emb_manual, emb_ref, atol=1e-6), \
        "Bilinear gather does not match grid_sample!"


@pytest.mark.parametrize("name", ["interpolate", "grid_sample", "sincos"])
def test_benchmark_posembed(name, benchmark_v2):
    """
    Benchmark the full PatchLoss: naive vs optimized.
    """
    # shapes matching your multi-crop: gV=1,gT=37; lV=5,lT=7 → T=37+5*7=72
    D = 768
    H, W = 224, 224
    P = 16
    B = 128
    N = 188 * B
    grid_size = (H // P, W // P)
    # grid = torch.randn(D, *grid_size, dtype=torch.float32)
    grid = get_canonical_pos_embed(D, grid_size=grid_size, patch_size=P, device="cuda")
    grid = grid.squeeze(0).permute(1, 0).unflatten(1, grid_size)  # [D, H//P, W//P]
    if name == "sincos":
        # use sin-cos positional embedding
        fn = partial(
            get_2d_sincos_pos_embed, embed_dim=D
        )
    elif name == "interpolate":
        # use bilinear interpolation
        fn = partial(
            interpolate_pos_embed, grid=grid
        )
    elif name == "grid_sample":
        # use torch.grid_sample
        fn = partial(
            interpolate_grid_sample, grid=grid
        )
    else:
        raise ValueError(f"Unknown name: {name}")
    
    def run():
        with torch.amp.autocast("cuda"):
            coords = torch.stack([
                torch.rand(N, device='cuda').mul(H - P),
                torch.rand(N, device='cuda').mul(W - P)
            ], dim=1)
            coords = coords.float() / P
            return fn(coords=coords)

    # warmup & benchmark
    benchmark_v2.benchmark(run, n_warmup=10, n_runs=200)

    # keep only timing columns
    benchmark_v2.drop_columns(exclude=["time/min (ms)", "time/max (ms)"])