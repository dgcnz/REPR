import torch
import torch.nn as nn
import pytest



class StressLossOptim(nn.Module):
    """
    “Fused” stress loss: ||z_i - z_j||² = |z_i|² + |z_j|² - 2 z_i·z_j,
    implemented with torch.baddbmm to fuse the two large steps.
    """
    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, z: torch.Tensor, dt_gt: torch.Tensor) -> torch.Tensor:
        # z:    [B, T, D]
        # dt_gt:[B, T, T, 4]
        # 1) GT distances
        d_gt = dt_gt.norm(dim=-1).clamp(min=self.eps)  # [B, T, T]

        # 2) squared norms of z
        z_norm2 = (z * z).sum(dim=-1)                  # [B, T]

        # 4) fused batched add-bmm: base + (-2)*(z @ z^T)
        dist2 = torch.baddbmm(
            # base = |z_i|^2 + |z_j|^2
            z_norm2.unsqueeze(2) + z_norm2.unsqueeze(1),  # [B, T, T]
            z,                      # [B, T, D]
            z.transpose(1, 2),      # [B, D, T]
            beta=1.0,
            alpha=-2.0
        ).clamp(min=self.eps)                        # [B, T, T]

        # 5) sqrt → distances
        dist = torch.sqrt(dist2)

        # 6) stress error
        err = dist - d_gt                             # [B, T, T]

        # 7) mean squared
        return (err * err).mean()



class StressLossSimple(nn.Module):
    """
    Naïve implementation of uniform stress loss, now using the full
    dt_gt: [B, T, T, 4] to compute ground-truth distances.
    """

    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, z: torch.Tensor, dt_gt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z     : [B, T, D]    patch embeddings
            dt_gt : [B, T, T, 4] ground-truth deltas (dy, dx, dlogh, dlogw)
        Returns:
            scalar stress loss
        """
        B, T, D = z.shape

        # 1) compute scalar ground‐truth distances: ||Δp||₂ over the last dim
        d_gt = dt_gt.norm(dim=-1).clamp(min=self.eps)  # [B, T, T]

        # 2) pairwise embedding differences [B, T, T, D]
        diff = z.unsqueeze(2) - z.unsqueeze(1)

        # 3) embedding‐space distances [B, T, T]
        dist = diff.norm(dim=-1).clamp(min=self.eps)

        # 4) stress error = (||z_i−z_j|| − ||Δp_{ij}||)^2
        err = dist - d_gt

        # 5) mean squared error over all pairs and batches
        return (err * err).mean()

    

@pytest.fixture(
    params=[
        "baseline",
        "optimized",
    ]
)
def stressloss_fn(request):
    # choose a fixed sigma vector
    if request.param == "baseline":
        return StressLossSimple().eval().cuda()
    elif request.param == "optimized":
        return StressLossOptim().eval().cuda()
    else:
        raise ValueError(f"Unknown stressloss_fn: {request.param}")



def test_stressloss_full(stressloss_fn, benchmark_v2):
    """
    Benchmark the full PatchLoss: naive vs optimized.
    """
    # shapes matching your multi-crop: gV=1,gT=37; lV=5,lT=7 → T=37+5*7=72
    gV, lV = 1, 5
    gT, lT = 37, 7
    T = gV * gT + lV * lT
    B, D = 256, 768
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # random inputs
    z = torch.randn(B, T, D, device=device)
    gt_dT = torch.randn(B, T, T, 4, device=device)
    gt_dT = gt_dT + gt_dT.transpose(1, 2)  # make it symmetric

    fn = stressloss_fn
    def run(*inputs):
        return fn(*inputs)
        # with torch.no_grad():

    # warmup & benchmark
    benchmark_v2.benchmark(run, args=(z, gt_dT), n_warmup=10, n_runs=100)

    # keep only timing columns
    benchmark_v2.drop_columns(exclude=["time/min (ms)", "time/max (ms)"])

    benchmark_v2.group_by(f"compile={compile}")


# ---------------------------------------------------
# Equivalence test
# ---------------------------------------------------


def test_stressloss_equivalence():
    """Ensure baseline and optimized produce the same loss (in float64)."""
    B, T, D = 128, 72, 768
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # use double precision for exactness
    z = torch.randn(B, T, D, device=device, dtype=torch.double)
    gt_dT = torch.randn(B, T, T, 4, device=device, dtype=torch.double)
    gt_dT = gt_dT + gt_dT.transpose(1, 2)  # make it symmetric

    # cast both modules to double
    m0 = StressLossSimple().to(device).double()
    m1 = StressLossOptim().to(device).double()

    loss0 = m0(z, gt_dT)
    loss1 = m1(z, gt_dT)

    # now they’ll agree to within 1e-12 or so
    torch.testing.assert_close(loss0, loss1, rtol=1e-9, atol=1e-9)
