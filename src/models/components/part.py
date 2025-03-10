from typing import Type

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from src.models.components.utils.part_utils import (
    compute_gt_transform,
    get_all_pairs,
)
from torch import Tensor
from timm.models.vision_transformer import Block
from torch.nn.functional import mse_loss, l1_loss


from src.models.components.utils.patch_embed import (
    OffGridPatchEmbed,
    random_sampling,
    stratified_jittered_sampling,
    ongrid_sampling,
    ongrid_sampling_canonical,
)


SAMPLERS = {
    "random": random_sampling,
    "stratified_jittered": stratified_jittered_sampling,
    "ongrid": ongrid_sampling,
    "ongrid_canonical": ongrid_sampling_canonical,
}
CRITERIONS = {
    "mse": mse_loss,
    "l1": l1_loss,
}


class PARTViT(nn.Module):
    """ Self-supervised training by predicting distances between patches in an image."""
    def __init__(
        self,
        # Encoder params
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        mask_ratio: float = 0.75,
        pos_mask_ratio: float = 0.75,
        num_targets: int = 2,
        sampler: str = "random",
        criterion: str = "mse",
    ):
        super().__init__()

        # --------------------------------------------------------------------------
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.pos_mask_ratio = pos_mask_ratio
        self.num_targets = num_targets
        self.patch_embed = OffGridPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            mask_ratio=0,  # samples 224/16 * 224/16 = 196 patches by default
            sampler=SAMPLERS[sampler],
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_pos_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim), requires_grad=True
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, num_targets, bias=False)
        self.criterion = CRITERIONS[criterion]
        self.tanh = nn.Tanh()  # Adding tanh activation

    def forward_encoder(self, x: Float[Tensor, "B C H W"]) -> Float[Tensor, "B N D"]:
        x, patch_positions_vis = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return {
            "z": x,
            "patch_positions_vis": patch_positions_vis,
        }

    def forward_decoder(
        self,
        x: Float[Tensor, "B (N+1) D"],
        img_size: int,
    ):
        x = self.decoder_pred(x)
        x: Float[Tensor, "B N 2"] = x[:, 1:, :]  # remove CLS
        # compute pairwise differences
        x: Float[Tensor, "B N N C"] = x.unsqueeze(2) - x.unsqueeze(1)
        x: Float[Tensor, "B N**2 C"] = x.flatten(1, 2)
        x = self.tanh(x) * img_size
        return {"pred_T": x}

    def forward_loss(
        self,
        pred_T: Float[Tensor, "B N**2 2"],
        patch_positions_vis: Int[Tensor, "B N 2"],
        img_size: int,
    ):
        B, N = patch_positions_vis.shape[:2]
        patch_pair_indices = get_all_pairs(B, N, device=pred_T.device)
        gt_T = compute_gt_transform(patch_pair_indices, patch_positions_vis)
        loss = self.criterion(pred_T / img_size, gt_T / img_size)
        return {
            "loss": loss,
            "patch_pair_indices": patch_pair_indices,
            "gt_T": gt_T,
        }

    def forward(self, x: Float[Tensor, "B C H W"]):
        out = dict()
        img_size = x.shape[-2]
        assert img_size == x.shape[-1], "Input image must be square"
        out |= self.forward_encoder(x)
        out |= self.forward_decoder(out["z"], img_size)
        out |= self.forward_loss(out["pred_T"], out["patch_positions_vis"], img_size)
        return out
