from jaxtyping import Float, Int
from torch import Tensor
import torch
import pytest

def _drop_pos(
    x: Float[Tensor, "B N K"], ids_remove: Int[Tensor, "B M"]
) -> Float[Tensor, "B M K"]:
    idx = ids_remove.unsqueeze(-1).expand(-1, -1, x.shape[-1])
    return torch.gather(x, dim=1, index=idx)

def drop_pos1d(x: torch.Tensor, ids_remove: torch.Tensor) -> torch.Tensor:
    batch_idx = torch.arange(x.size(0), device=x.device)[:, None]  # → [B,1]
    return x[batch_idx, ids_remove]                               # → [B, M, K]


@pytest.mark.parametrize("fn", ["gather", "arange"])
def test_drop_pos(fn: str, benchmark_v2):
    # Settings
    B = 512 
    N = 188
    M = 144
    D = 768
    fun = _drop_pos if fn == "gather" else drop_pos1d

    x = torch.randn(B, N, D, device='cuda')  # Simulated input tensor
    ids = torch.randint(0, N, (B, M), device='cuda')  # Simulated indices to remove


    benchmark_v2.benchmark(fun, args=(x, ids), n_warmup=5, n_runs=100)
    # benchmark_v2.group_by(f"compile: {compile}")
    benchmark_v2.drop_columns(exclude=["time/min (ms)", "time/max (ms)"])


# check equivalence
def test_drop_pos_equivalence():
    B = 512 
    N = 188
    M = 144
    D = 768

    x = torch.randn(B, N, D, device='cuda')
    ids = torch.randint(0, N, (B, M), device='cuda')
    out_gather = _drop_pos(x, ids)
    out_arange = drop_pos1d(x, ids)
    assert torch.allclose(out_gather, out_arange, atol=1e-6), "Outputs are not equivalent"
