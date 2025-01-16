import torch
import timm
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int
import math
from typing import Any
from src.data.components.sampling_utils import sample_and_stitch
from src.models.components.cross_attention import CrossAttention
from src.utils.fancy_indexing import index_pairs


class PARTViT(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int,
        num_patches: int,
        logit_scale_tanh: bool = False,
        logit_scale_init: float = 1.0,
        logit_scale_learnable: bool = False,
        head_type: str = "pairwise_mlp",
        num_targets: int = 2,  # 2: [dx, dy] 4: [dx, dy, dw, dh]
        cross_attention_num_heads: int = None,
        cross_attention_query_type: str = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.head_type = head_type
        self.logit_scale_learnable = logit_scale_learnable
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale_init)))
        self.logit_scale_tanh = logit_scale_tanh
        if not self.logit_scale_learnable:
            self.logit_scale.requires_grad = False
        if head_type == "pairwise_mlp":
            self.head = nn.Linear(2 * embed_dim, num_targets)
        elif self.head_type == "cross_attention":
            assert cross_attention_num_heads is not None
            assert cross_attention_query_type is not None
            self.head = CrossAttention(
                embed_dim=embed_dim,
                num_channels=num_targets,
                num_patches=num_patches,
                num_heads=cross_attention_num_heads,
                query_type=cross_attention_query_type,
            )
        elif self.head_type == "residual_mlp":
            self.proj = nn.Linear(embed_dim, num_targets)
            # K: [2, D], P = [D, D], M = [2, D]
            # K * (P * z1) - K * (P * z2) = K * P * (z1 - z2) = M * (z1 - z2)
        else:
            raise NotImplementedError(f"Head type {head_type} not implemented")

    def forward(
        self,
        x: Float[Tensor, "B C H W"],
        patch_pair_indices: Int[Tensor, "B NP 2"],
    ) -> tuple[Float[Tensor, "B NP 2"], Int[Tensor, "B NP 2"]]:
        z: Float[Tensor, "B N D"] = self.backbone(x)
        b, n, _ = z.size()
        assert (
            math.isqrt(n) ** 2 == n
        ), f"If the number of patches is not a perfect square, the cls token might be sneaking in. n={n}"

        z_pairs: Float[Tensor, "B NP 2 D"] = index_pairs(
            z, patch_pair_indices
        )
        if self.head_type == "pairwise_mlp":
            z: Float[Tensor, "B NP 2"] = self.head(z_pairs.flatten(2, 3))
        elif self.head_type == "cross_attention":
            z: Float[Tensor, "B NP 2"] = self.head(z, patch_pair_indices)
        elif self.head_type == "residual_mlp":
            z: Float[Tensor, "B NP 2"] = self.proj( z_pairs[:, :, 1] - z_pairs[:, :, 0])
        else:
            raise NotImplementedError(f"Head type {self.head_type} not implemented")


        if self.logit_scale_tanh:
            z = z.tanh()
        z = z * self.logit_scale
        return z


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
        embed_dim=backbone.embed_dim,
        num_patches=backbone.patch_embed.num_patches,
        head_type="cross_attention",
        num_targets=2,
        cross_attention_num_heads=4,
        cross_attention_query_type="positional",
    )
    x = torch.randn(1, 3, 64, 64)
    x, patch_positions = sample_and_stitch(x, 16, "offgrid")
    y_pred, patch_pair_indices = model(x)
    # y_gt = compute_gt_transform(patch_pair_indices, patch_positions)

    # plot reconstructed image given the predicted patch pair indices

    # mse = nn.L1Loss()(y_pred, y_gt)
    # print(mse)
    # print(y_pred)
    # print(y_gt)
