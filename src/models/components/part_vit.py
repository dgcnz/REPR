import torch
import timm
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int
import torch.nn.functional as F
import math
import timeit
import lightning as L
import logging
from typing import Any
from src.data.components.sampling_utils import sample_and_stitch


class PARTViT(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        n_pairs: int,
        embed_dim: int = 384,
        head_type: str = "pairwise_mlp",
        num_targets: int = 2,  # 2: [dx, dy] 4: [dx, dy, dw, dh]
    ):
        super().__init__()
        self.backbone = backbone
        self.n_pairs = n_pairs
        if head_type == "pairwise_mlp":
            self.head = nn.Linear(2 * embed_dim, num_targets)
        else:
            raise NotImplementedError(f"Head type {head_type} not implemented")

    def forward(
        self, x: Float[Tensor, "b c h w"]
    ) -> tuple[Float[Tensor, "b n_pairs 2"], Int[Tensor, "b n_pairs 2"]]:
        z: Float[Tensor, "b n embed_dim"] = self.backbone(x)
        b, n, _ = z.size()
        assert (
            math.isqrt(n) ** 2 == n
        ), f"If the number of patches is not a perfect square, the cls token might be sneaking in. n={n}"

        idx = torch.randint(n, (b, self.n_pairs), device=z.device)
        jdx = torch.randint(n, (b, self.n_pairs), device=z.device)
        indices = torch.stack([idx, jdx], dim=2)

        z: Float[Tensor, "b n_pairs 2 embed_dim"] = index_pairs(z, indices)
        z: Float[Tensor, "b n_pairs 2"] = self.head(z.flatten(2, 3))
        return z, indices


def compute_gt_transform(
    patch_pair_indices: Int[Tensor, "b n_pairs 2"],
    patch_positions: Int[Tensor, "b n_patches 2"],
) -> Float[Tensor, "b n_pairs 2"]:
    true_positions: Float[Tensor, "b n_pairs 2 2"] = index_pairs(
        patch_positions, patch_pair_indices
    )
    true_transform: Float[Tensor, "b n_pairs 2"] = (
        true_positions[:, :, 0] - true_positions[:, :, 1]
    )
    return true_transform.float()


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


if __name__ == "__main__":
    backbone = timm.create_model(
        "vit_base_patch16_384",
        pretrained=False,
        num_classes=0,
        global_pool="",
        class_token=False,
        img_size=64,
    )
    model = PARTViT(
        backbone=backbone,
        n_pairs=4,
        embed_dim=768,
        head_type="pairwise_mlp",
        num_targets=2,
    )
    x = torch.randn(2, 3, 64, 64)
    x, y = sample_and_stitch(x, 16, "offgrid")
    model = torch.compile(model)
    logits = model(x)
    criterion = PARTLoss()
    loss = criterion(*logits, y)
    print(loss)
    loss.backward()
