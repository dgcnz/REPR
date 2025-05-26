import torch
import pytest

@pytest.mark.parametrize("B,D,m,eps", [
    (1,  8, 16, 0.5),
    (3, 16, 32, 0.7),
    (5, 32, 64, 0.3),
])
def test_batched_vs_loop_cholesky_ex(B, D, m, eps):
    """
    Compare original looped cholesky_ex expansion‐loss to a fully batched one.
    """
    torch.manual_seed(0)
    # generate random "feature" matrices W of shape (m, p)
    Ws = [torch.randn(m, D) for _ in range(B)]
    # build cov_list exactly as in SimDINO: W^T W
    cov_list = torch.stack([W.T @ W for W in Ws], dim=0)  # [V, p, p]

    # constants
    N = 1  # world‐size
    scalar = D / (m * N * eps)
    I = torch.eye(D)

    # --- original looped version ---
    loss_loop = 0.0
    for i in range(B):
        M_i = I + scalar * cov_list[i]
        L_i, info_i = torch.linalg.cholesky_ex(M_i)
        # make sure factorization succeeded
        assert int(info_i) == 0
        loss_loop = loss_loop + L_i.diagonal().log().sum()
    loss_loop = loss_loop / B * ((D + N * m) / (D * N * m))

    # --- fully batched version ---
    M   = I.unsqueeze(0) + scalar * cov_list            # [V, p, p]
    Lb, info_b = torch.linalg.cholesky_ex(M)            # batched cholesky_ex
    # all factorizations must succeed
    assert torch.all(info_b == 0)
    # sum log diag per view, then mean
    log_diag = Lb.diagonal(dim1=-2, dim2=-1).log().sum(dim=1)  # [V]
    loss_batch = log_diag.mean() * ((D + N * m) / (D * N * m))

    # they should be (nearly) identical
    assert torch.allclose(loss_loop, loss_batch, rtol=1e-6, atol=1e-6), \
        f"loop {loss_loop.item()} vs batched {loss_batch.item()}"

@pytest.mark.parametrize("fn", ["looped", "batched"])
@pytest.mark.parametrize("B,D,m,eps", [
    (256,  768, 109, 0.1),
])
def test_cholesky_benchmark(fn, B, D, m, eps, benchmark_v2):
    """
    Benchmark looped vs batched cholesky_ex for the expansion‐loss.
    """
    torch.manual_seed(0)
    # generate random W and cov_list on CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Ws = [torch.randn(m, D, device=device) for _ in range(B)]
    cov_list = torch.stack([W.T @ W for W in Ws], dim=0)  # [V, p, p]

    N = 1
    scalar = D / (m * N * eps)
    I = torch.eye(D, device=device)

    def looped(cov_list):
        loss = 0.0
        for i in range(B):
            M_i = I + scalar * cov_list[i]
            L_i, info_i = torch.linalg.cholesky_ex(M_i)
            # ignore info check in benchmark
            loss = loss + L_i.diagonal().log().sum()
        loss = loss / B * ((D + N * m) / (D * N * m))
        return loss

    def batched(cov_list):
        M_batch = I.unsqueeze(0) + scalar * cov_list  # precompute
        loss = torch.linalg.cholesky_ex(M_batch)[0].diagonal(dim1=-2, dim2=-1).log().sum(dim=1).mean()
        return loss * ((D + N * m) / (D * N * m))

    fun = looped if fn == "looped" else batched
    args =  (cov_list,) if fn == "looped" else (cov_list,)
    # run the benchmark
    # benchmark_v2.group_by("cholesky_ex", params={"views,p,m,eps": f"{B},{p},{m},{eps}"})

    benchmark_v2.group_by(f"{B} views, p={D}, m={m}, eps={eps}")
    benchmark_v2.benchmark(fun,  args=args, n_runs=50, n_warmup=5)
    # drop min/max columns to focus on average speed
    benchmark_v2.drop_columns(exclude=["time/min (ms)", "time/max (ms)"])
