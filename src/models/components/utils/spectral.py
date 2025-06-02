import torch
from torch import Tensor

def batched_logdet(
    base: Tensor, z: Tensor, alpha: float, num_chunks: int = 1
) -> Tensor:
    """
    Compute the average log-determinant of matrices: base_mat + α * (z_i^T z_i) for each sample in the batch,
    splitting the batch into `num_chunks` pieces for memory efficiency.

    Args:
        base_mat: Tensor of shape [D, D], typically an identity matrix or precomputed constant matrix.
        z: Tensor of shape [B, T, D], batch of embeddings.
        alpha: Scalar multiplier for the covariance term.
        num_chunks: Number of chunks to split the batch into. Must divide batch size exactly.

    Returns:
        A scalar Tensor containing (1/B) * sum_i log det(base_mat + α z_i^T z_i).
    """
    B, T, D = z.shape
    # num_chunks must divide B exactly
    assert B % num_chunks == 0, (
        f"Batch size ({B}) must be divisible by num_chunks ({num_chunks})"
    )
    chunk_size = B // num_chunks

    # compute per-chunk log-det sums via list comprehension
    chunk_logs = [
        torch.linalg.cholesky_ex(
            torch.baddbmm(base, z_chunk.transpose(1, 2), z_chunk, alpha=alpha).float()
        )[0]
        .diagonal(dim1=-2, dim2=-1)
        .log()
        .sum()
        for z_chunk in z.split(chunk_size, dim=0)
    ]

    # aggregate across all chunks
    sum_log = torch.stack(chunk_logs, dim=0).sum()
    # return the mean log-det across the batch
    mean_logdet = sum_log / B
    return mean_logdet