# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# DropPos: https://github.com/Haochen-Wang409/DropPos
# --------------------------------------------------------

from functools import partial
from typing import Type, TypedDict

import torch
from torch import nn, Tensor
from jaxtyping import Float, Int, Bool
from src.models.components.utils.part_utils import (
    compute_gt_transform,
    get_all_pairs_subset,
)
from torch.nn.functional import mse_loss, l1_loss
from timm.models.vision_transformer import PatchEmbed, Block
from src.models.components.utils.patch_embed import OffGridPatchEmbed
from src.models.components.utils.offgrid_pos_embed import (
    get_2d_sincos_pos_embed,
    get_canonical_pos_embed,
)
from torch.nn.modules.utils import _pair


class EncoderOutput(TypedDict):
    "The output of the encoder"
    z: Float[Tensor, "B N_vis D"]
    patch_positions_vis: Float[Tensor, "B N_vis 2"]
    patch_positions_pos: Float[Tensor, "B N_pos 2"]
    ids_keep: Int[Tensor, "B N_vis"]
    ids_nopos: Int[Tensor, "B N_nopos"]
    mask_pos: Bool[Tensor, "B N_vis"]
    ids_keep_pos: Int[Tensor, "B N_pos"]
    ids_restore_pos: Int[Tensor, "B ?"]


class DecoderOutput(TypedDict):
    pred_T: Float[Tensor, "B N_nopos**2 2"]


class LossOutput(TypedDict):
    loss: Float[Tensor, "B"]
    patch_pair_indices: Int[Tensor, "B N_nopos**2 2"]
    gt_T: Float[Tensor, "B N_nopos**2 2"]


class ForwardOutput(EncoderOutput, DecoderOutput, LossOutput):
    pass


class PARTMaskedAutoEncoderViT(nn.Module):
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
        # Decoder params
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        num_targets: int = 2,
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
            mask_ratio=mask_ratio,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )
        # mask token for position
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
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, num_targets, bias=False)
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        grid_size = _pair(self.img_size // self.patch_size)
        pos_embed = get_canonical_pos_embed(self.embed_dim, grid_size, self.patch_size)
        cls_pos_embed = torch.zeros(1, 1, self.embed_dim)
        self.pos_embed.data = torch.cat([cls_pos_embed, pos_embed], dim=1)

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_pos_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)
        # initialize decoder to avoid tanh saturation
        torch.nn.init.kaiming_uniform_(self.decoder_pred.weight, a=2)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x: Float[Tensor, "B N D"], mask_ratio: float):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        B, N, D = x.shape  # batch, length, dim
        len_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        # remove the second subset
        ids_remove = ids_shuffle[:, len_keep:]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore).bool()

        return ids_keep, mask, ids_restore, ids_remove

    def forward_encoder(self, x: Float[Tensor, "B C H W"]) -> EncoderOutput:
        # patch_embed already includes masking by offgrid subsampling
        x, patch_positions_vis = self.patch_embed(x)
        # patch_positions_vis[i] is the position of x[i]
        B, N_vis, D = x.shape

        # random masking for position embedding
        ids_keep_pos, mask_pos, ids_restore_pos, ids_remove_pos = self.random_masking(
            x, self.pos_mask_ratio
        )
        N_pos = ids_keep_pos.shape[1]
        N_nopos = N_vis - N_pos

        # patch_positions_pos[i] is NOT the position of x[i]
        # patch_positions_pos[i] = patch_positions_vis[index_keep_pos[i]]
        patch_positions_pos = torch.gather(
            patch_positions_vis, dim=1, index=ids_keep_pos.unsqueeze(-1).repeat(1, 1, 2)
        )
        # pos_embed[i] = pos_embed_vis[index_keep_pos[i]] kinda, not quite
        pos_embed = get_2d_sincos_pos_embed(
            patch_positions_pos.flatten(0, 1) / self.patch_size, self.embed_dim
        )
        pos_embed = pos_embed.unflatten(0, (B, N_pos))

        # append mask tokens to position embeddings
        mask_pos_tokens = self.mask_pos_token.repeat(B, N_nopos, 1)
        pos_embed = torch.cat([pos_embed, mask_pos_tokens], dim=1)

        # restore position embeddings before adding
        pos_embed = torch.gather(
            pos_embed, dim=1, index=ids_restore_pos.unsqueeze(-1).repeat(1, 1, D)
        )

        # add position embedding w/o [cls] token
        x = x + pos_embed

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return {
            "z": x,  # still in the original order of x_vis
            "patch_positions_vis": patch_positions_vis,
            "patch_positions_pos": patch_positions_pos,
            "ids_remove_pos": ids_remove_pos,
            "mask_pos": mask_pos,
            "ids_keep_pos": ids_keep_pos,
            "ids_restore_pos": ids_restore_pos,
        }

    def forward_decoder(
        self,
        x: Float[Tensor, "B N_vis D"],
        img_size: int,
        ids_remove_pos: Int[Tensor, "B N_nopos"],
    ) -> DecoderOutput:
        x = self.decoder_embed(x)
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x: Float[Tensor, "B N_vis 2"] = x[:, 1:, :]  # remove CLS
        B, _, C = x.shape
        # only predict pairwise transformations for visible patches with no position embeddings
        x: Float[Tensor, "B N_nopos C"] = torch.gather(
            x, dim=1, index=ids_remove_pos.unsqueeze(-1).expand(B, -1, C)
        )
        # compute pairwise differences
        x: Float[Tensor, "B N_nopos N_nopos C"] = x.unsqueeze(2) - x.unsqueeze(1)
        x: Float[Tensor, "B N_nopos**2 C"] = x.flatten(1, 2)
        x = x.tanh() * img_size
        return {"pred_T": x}

    def forward_loss(
        self,
        pred_T: Float[Tensor, "B N_nopos**2 2"],
        ids_remove_pos: Int[Tensor, "B N_nopos"],
        patch_positions_vis: Int[Tensor, "B N_vis 2"],
        img_size: int,
    ) -> LossOutput:
        """
        Compute the loss for the model.
        :param pred_T: predicted transformations
        :param ids_remove_pos: indices of the visible patches without position embeddings
        :param patch_positions_vis: positions of the patches
        :param img_size: size of the image
        """
        # only compute loss over the visible paches **without** position embeddings
        patch_pair_indices: Int[Tensor, "B N_nopos**2 2"] = get_all_pairs_subset(
            ids=ids_remove_pos
        )
        gt_T = compute_gt_transform(patch_pair_indices, patch_positions_vis)
        loss = mse_loss(pred_T / img_size, gt_T / img_size)
        return {
            "loss": loss,
            "patch_pair_indices": patch_pair_indices,
            "gt_T": gt_T,
        }

    def forward(self, x: Float[Tensor, "B C H W"]) -> ForwardOutput:
        """
        :param x: image batch
        :return: ForwardOutput
        """
        out = dict()
        img_size = x.shape[-2]
        assert img_size == x.shape[-1], "Input image must be square"
        out |= self.forward_encoder(x)
        out |= self.forward_decoder(out["z"], img_size, out["ids_remove_pos"])
        out |= self.forward_loss(
            out["pred_T"], out["ids_remove_pos"], out["patch_positions_vis"], img_size
        )
        return out


def PART_mae_vit_base_patch16_dec512d8b(**kwargs):
    model = PARTMaskedAutoEncoderViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


# set recommended archs
PART_mae_vit_base_patch16 = (
    PART_mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
)

if __name__ == "__main__":
    backbone = PART_mae_vit_base_patch16(pos_mask_ratio=0.4, mask_ratio=0.3)
    x = torch.randn(2, 3, 224, 224)
    # def forward(self, imgs, mask_ratio, pos_mask_ratio):
    patch_size = 16
    num_patches = (224 // patch_size) ** 2
    print(backbone.patch_embed.patch_size)
    out = backbone.forward(x)
    print(out["loss"])
    # compute expected loss
    best_T = torch.zeros_like(out["gt_T"]) / 224
    print(mse_loss(best_T, out["gt_T"] / 224))
