from torch import Tensor
from jaxtyping import Float, Int
import torch


def index_pairs(
    x: Float[Tensor, "b n d"], indices: Int[Tensor, "b k 2"]
) -> Float[Tensor, "b k 2 d"]:
    """
    For each pair indices[b][k][0] and indices[b][k][1],
    gather the corresponding patches from x and stack them together.
    """
    b, n, dim = x.shape
    _, k, _ = indices.shape
    batch_offsets = torch.arange(b, device=x.device)[:, None, None] * n
    flat_indices = torch.flatten(indices + batch_offsets)
    return x.flatten(0, 1)[flat_indices].view(b, k, 2, dim)
