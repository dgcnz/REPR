import torch
from torch import nn, Tensor
from jaxtyping import Float
from src.models.components.utils.spectral import batched_logdet


class PatchCodingRateLoss(nn.Module):
    """
    Encourages diversity in patch embeddings.
    """

    def __init__(self, embed_dim: int, eps: float = 0.5, num_chunks: int = 2):
        super().__init__()
        self.eps = eps
        self.num_chunks = num_chunks
        self.register_buffer(
            "I", torch.eye(embed_dim, dtype=torch.float32), persistent=False
        )

    def forward(self, z: Float[Tensor, "B T D"]) -> Float[Tensor, "1"]:
        """
        :param z: l2 normalized patch embeddings

        For gV global views and lV local views, with gN global patches per view
        and lN local patches per view, T is gV * gN + lV * lN.

        This loss is meant to encourage the embeddings of patches
        across all views to be diverse.
        """
        _, T, D = z.shape

        # add identity to avoid ill-conditioned case: I + α·cov
        alpha = D / (T * self.eps)

        expa = batched_logdet(self.I, z, alpha, num_chunks=self.num_chunks)

        # 4) final γ‐scaling
        gamma = (D + T) / (D * T)
        return {"loss_pcr": -expa * gamma}


class PatchCodingRateLossV2(nn.Module):
    """
    Encourages diversity in patch embeddings.
    """

    def __init__(
        self, embed_dim: int, gN: int, gV: int, pool_size: int, eps: float = 0.5, num_chunks: int = 2
    ):
        super().__init__()
        self.eps = eps
        self.num_chunks = num_chunks
        self.gN = gN
        self.gV = gV
        self.pool_size = pool_size
        self.register_buffer(
            "I", torch.eye(embed_dim, dtype=torch.float32), persistent=False
        )

    def forward(self, z: Float[Tensor, "B T D"]) -> Float[Tensor, "1"]:
        """
        :param z: l2 normalized patch embeddings

        For gV global views and lV local views, with gN global patches per view
        and lN local patches per view, T is gV * gN + lV * lN.

        This loss is meant to encourage the embeddings of patches
        across all views to be diverse.
        """
        B, T, D = z.shape

        z = z[:, : self.gN * self.gV]
        z = (
            z.view(B // self.pool_size, self.pool_size, self.gV, self.gN, D)
            .permute(2, 0, 1, 3, 4)
            .reshape(self.gV * (B // self.pool_size), self.pool_size * self.gN, D)
        )
        # [gV * (B // pool_size), pool_size * gN, D]
        assert B % self.pool_size == 0, (
            "Batch size must be divisible by pool_size for this loss."
        )
        # add identity to avoid ill-conditioned case: I + α·cov
        Tv = z.shape[1]
        alpha = D / (Tv * self.eps)
        gamma = (D + Tv) / (D * Tv)

        expa = batched_logdet(self.I, z, alpha, num_chunks=self.num_chunks)

        # 4) final γ‐scaling
        return {"loss_pcr": -expa * gamma}
