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
        return -expa * gamma