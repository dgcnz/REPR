import torch
import pytest

# --- three distance‐squared implementations ---


def dist2_baseline(z):
    # naive broadcast: [B,T,1,D] - [B,1,T,D] → [B,T,T,D]
    dz = z.unsqueeze(2) - z.unsqueeze(1)  # [B,T,T,D]
    return dz.pow(2).sum(dim=-1)  # [B,T,T]

def dist2_bmm(z):
    # ||zi-zj||^2 = ||zi||^2 + ||zj||^2 - 2 zi·zj
    B, T, D = z.shape
    z2 = (z * z).sum(dim=-1)  # [B,T]
    gram = torch.bmm(z, z.transpose(1, 2))  # [B,T,T]
    dist = z2.unsqueeze(2) + z2.unsqueeze(1) - 2 * gram
    dist = dist.clamp(min=0)  # numerical stability
    eye = torch.eye(T, device=dist.device, dtype=torch.bool).unsqueeze(0)  # [1,T,T]
    dist = dist.masked_fill(eye, 0.0)
    return dist  # [B,T,T]

def dist2_badbmm(z):
    # z: [B, T, D]
    # 1) precompute norms
    z2 = (z * z).sum(dim=-1)               # [B, T]
    # 2) build the 2*||z||^2 term once:
    #    dist[b,i,j] = ||z_i||^2 + ||z_j||^2
    dist = z2.unsqueeze(2) + z2.unsqueeze(1)  # [B, T, T]

    # 3) addbmm_: dist = 1.0*dist + (-2.0)*(z @ zᵀ)
    dist.baddbmm_(z, z.transpose(1, 2), beta=1.0, alpha=-2.0)


    # 4) clamp & zero‐diag for exactness
    dist.clamp_(min=0.0)
    dist.diagonal(dim1=1, dim2=2).zero_()
    return dist


def dist2_cdist(z):
    return torch.cdist(z, z, p=2).pow(2)  # [B,T,T]


FUNCTIONS = {
    "baseline": dist2_baseline,
    "bmm": dist2_bmm,
    "badbmm": dist2_badbmm,
    "cdist": dist2_cdist,
}

# --- the benchmark test ---


@pytest.mark.parametrize(
    "method",
    ["baseline", "bmm", "badbmm", "cdist"],
)
def test_patchloss_distances(method, benchmark_v2):
    """
    Compare three ways to compute pairwise-squared distances in PatchLoss:
      - baseline broadcast,
      - Gram-matrix via bmm,
      - torch.cdist
    """
    # create a random tensor of the same shape you use in PatchLoss
    gV, lV = 1, 5
    gT, lT = 37, 7
    T = gV * gT + lV * lT
    B, D = 256, 768
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z = torch.randn(B, T, D, device=device)

    fn = FUNCTIONS[method]

    # time it
    benchmark_v2.benchmark(fn, args=(z,), n_warmup=10, n_runs=100)

    # group and simplify the output table
    benchmark_v2.drop_columns(exclude=["time/min (ms)", "time/max (ms)"])


def test_implementation_equivalence():
    """Test that all three implementations produce equivalent results."""
    B, T, D = 32, 16, 128  # Smaller sizes for testing
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z = torch.randn(B, T, D, device=device)

    baseline_result = dist2_baseline(z)
    cdist_result = dist2_cdist(z)
    bmm_result = dist2_bmm(z)
    badbmm_result = dist2_badbmm(z)

    torch.testing.assert_close(baseline_result, cdist_result, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(baseline_result, bmm_result, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(baseline_result, badbmm_result, rtol=1e-5, atol=1e-5)
