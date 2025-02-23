import torch
import pytest
from models.components.utils.offgrid_pos_embed import (
    PosEmbedDynamicDiff,
    PosEmbedStaticDiff,
    PosEmbedDynamicSame,
    PosEmbedStaticSame,
    PosEmbedBaseline,
    PosEmbedPrecomputedDiff,
    PosEmbedDynamicDiffOptimizedDiscreteV2,
)

from src.models.components.utils.posembed_triton import PosEmbedDynamicDiffTriton


@pytest.mark.parametrize(
    "module_class, coord_type",
    [
        (PosEmbedDynamicSame, "same"),
        (PosEmbedStaticSame, "same"),
        (PosEmbedDynamicDiff, "diff"),
        (PosEmbedStaticDiff, "diff"),
        (PosEmbedPrecomputedDiff, "diff"),
        (PosEmbedDynamicDiffOptimizedDiscreteV2, "diff"),
        (PosEmbedBaseline, "baseline"),
        (PosEmbedDynamicDiffTriton, "diff"),
    ],
)
@pytest.mark.parametrize("compile", [True, False])
def test_offgrid_posembeds(module_class, coord_type, compile: bool, benchmark_v2):
    # Settings (e.g., ViT with 14x14 patches)
    B = 256  # Batch size
    grid_size = (14, 14)  # Grid of patches (H, W)
    embed_dim = 768
    patch_size = 16
    N = grid_size[0] * grid_size[1]  # Number of tokens
    # N = int(0.25 * N * 0.75) # to simulate masking
    device = "cuda"

    max_coord = patch_size * (grid_size[0] - 1)
    gen_kwargs = {"device": device, "dtype": torch.float32}
    # Prepare coordinates
    if module_class in (PosEmbedPrecomputedDiff, PosEmbedDynamicDiffOptimizedDiscreteV2 ):
        # For the optimized discrete module, use a fixed set of coordinates.
        args_gen = lambda: ((torch.randn(B, N, embed_dim, device=device), torch.randint(0, max_coord, (B, N, 2), device=device)),)
    elif coord_type == "same":
        # For SAME modules, use a fixed set of coordinates.
        args_gen = lambda: ((torch.randn(B, N, embed_dim, device=device), torch.randint(0, max_coord, (N, 2), **gen_kwargs)),)
    elif coord_type == "diff":
        # For DIFF modules, create heterogeneous coordinates per batch.
        args_gen = lambda: ((torch.randn(B, N, embed_dim, device=device), torch.randint(0, max_coord, (B, N, 2), **gen_kwargs)),)
    else:
        args_gen = lambda: ((torch.randn(B, N, embed_dim, device=device),),)

    # Instantiate the module based on its class.
    if module_class in (PosEmbedDynamicSame, PosEmbedDynamicDiff, PosEmbedPrecomputedDiff, PosEmbedDynamicDiffOptimizedDiscreteV2, PosEmbedDynamicDiffTriton):
        module = module_class(embed_dim)
    else:
        module = module_class(embed_dim, patch_size, grid_size)

    module = module.to(device)
    module.eval()
    if compile:
        module = torch.compile(module)

    # Define a lambda wrapper for benchmarking.
    def ffn(inputs):
        with torch.no_grad():
            return module(*inputs)

    benchmark_v2.benchmark(ffn, args_gen=args_gen, n_warmup=100, n_runs=1000)
    benchmark_v2.group_by(f"compile: {compile}")
    benchmark_v2.filter_columns(exclude=["time/min (ms)", "time/max (ms)"])
