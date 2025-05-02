import torch
import math
from torch import nn, Tensor
from torch.nn.functional import mse_loss, l1_loss
from timm.models.vision_transformer import Block
from functools import partial
from src.models.components.utils.patch_embed import (
    OffGridPatchEmbed,
    random_sampling,
    stratified_jittered_sampling,
    ongrid_sampling,
    ongrid_sampling_canonical,
)
from src.models.components.utils.offgrid_pos_embed import (
    get_2d_sincos_pos_embed,
    get_canonical_pos_embed,
)
from torch.nn.modules.utils import _pair
from jaxtyping import Int, Float
from typing import Literal

SAMPLERS = {
    "random": random_sampling,
    "stratified_jittered": stratified_jittered_sampling,
    "ongrid": ongrid_sampling,
    "ongrid_canonical": ongrid_sampling_canonical,
}


class PARTMaskedAutoEncoderViT(nn.Module):
    r"""
    Self-supervised training by generalized patch-level transforms between augmented views of an image.
    """

    def __init__(
        self,
        img_size: int = 224,
        canonical_img_size: int = 512,
        max_scale_ratio: float = 6.0,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        norm_layer: type = nn.LayerNorm,
        mask_ratio: float = 0.75,
        pos_mask_ratio: float = 0.75,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        sampler: str = "random",
        criterion: str = "l1",
        alpha_t: float = 0.5,
        alpha_ts: float = 0.5,
        alpha_s: float = 0.75,
        verbose: bool = False,
        num_views: int = 6,  # default is 1 global 5 local
        # TODO: remove after 200ep training of old model
        segment_embed_mode: Literal["permute", "none", "fixed"] = "none",
        freeze_encoder: bool = False,
        apply_tanh: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.canonical_img_size = canonical_img_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.pos_mask_ratio = pos_mask_ratio
        self.max_scale_ratio = max_scale_ratio
        self.alpha_ts = alpha_ts
        self.alpha_t = alpha_t
        self.alpha_s = alpha_s

        self.patch_embed = OffGridPatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            mask_ratio=mask_ratio,
            sampler=SAMPLERS[sampler],
        )
        criterions = {"l1": l1_loss, "mse": mse_loss}
        self.criterion = criterions[criterion]

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )
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
        self.norm: nn.Module = norm_layer(embed_dim)
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
        self.num_targets = 4
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.num_targets, bias=False)
        self.apply_tanh = apply_tanh
        self.tanh = nn.Tanh() if self.apply_tanh else nn.Identity()
        self.verbose = verbose

        # Instead of using nn.Embedding, use a raw parameter for view-specific segment embeddings.
        self.num_views = num_views
        self.segment_embed_mode = segment_embed_mode
        if self.segment_embed_mode == "none":
            self.register_buffer("segment_embed", torch.zeros(num_views, embed_dim))
        else:
            self.segment_embed = nn.Parameter(torch.randn(num_views, embed_dim))

        self.initialize_weights()
        if freeze_encoder:
            self.freeze_encoder()

    def freeze_encoder(self):
        r"""
        Freeze the encoder blocks.
        """
        for param in self.patch_embed.parameters():
            param.requires_grad = False

        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False

        for param in self.norm.parameters():
            param.requires_grad = False

        self.cls_token.requires_grad = False

    def update_conf(
        self,
        sampler: str = None,
        mask_ratio: float = None,
        pos_mask_ratio: float = None,
    ):
        r"""
        Update configuration parameters.

        :param sampler: New sampler type, if provided.
        :param mask_ratio: New mask ratio, if provided.
        :param pos_mask_ratio: New positional mask ratio, if provided.
        """
        if mask_ratio is not None:
            self.mask_ratio = mask_ratio
            self.patch_embed.mask_ratio = mask_ratio
        if pos_mask_ratio is not None:
            self.pos_mask_ratio = pos_mask_ratio
        if sampler is not None:
            self.patch_embed.sampler = SAMPLERS[sampler]

    def initialize_weights(self):
        r"""
        Initialize positional embeddings, cls token, mask token, and other weights.
        """
        grid_size = _pair(self.img_size // self.patch_size)
        pos_embed = get_canonical_pos_embed(
            self.embed_dim, grid_size, self.patch_size, device=self.pos_embed.device
        )
        cls_pos_embed = torch.zeros(1, 1, self.embed_dim)
        self.pos_embed.data = torch.cat([cls_pos_embed, pos_embed], dim=1)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_pos_token, std=0.02)
        self.apply(self._init_weights)
        nn.init.kaiming_uniform_(self.decoder_pred.weight, a=4)

    def _init_weights(self, m):
        r"""
        Initialize weights for linear and norm layers.
        """
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def encode_views(self, x: Tensor) -> dict:
        r"""
        Encode a batch of views.

        :param x: Tensor of shape [B, V, C, H, W]
        :return: Dict with keys:
                 "z_enc": Tensor of shape [B, V, N+1, D] (N visible patches + 1 class token),
                 "patch_positions_vis": Tensor of shape [B, V, N, 2] (N visible patch positions),
                 "ids_remove_pos": Tensor of shape [B, V, M] (M indices of masked tokens)
        """
        B, V, C, H, W = x.shape
        x_flat = x.view(B * V, C, H, W)
        out = self.forward_encoder(x_flat)
        out["z_enc"] = out["z_enc"].view(B, V, -1, self.embed_dim)
        out["patch_positions_vis"] = out["patch_positions_vis"].view(B, V, -1, 2)
        out["ids_remove_pos"] = out["ids_remove_pos"].view(B, V, -1)
        return out

    def prepare_joint_inputs(
        self, g_enc: dict, l_enc: dict
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""
        Prepare joint inputs for the decoder given global and local encoder outputs.

        This function inlines the logic for splitting the per-view tokens (class vs. patch), then builds:
          - The joint latent sequence: all class tokens (from both global and local) come first, followed by all patch tokens (flattened).
          - The joint patch positions: flatten and concatenate per-view patch positions.
          - The joint drop indices: flatten and concatenate per-view ids_remove.

        :param g_enc: Global encoder outputs with keys "z_enc", "patch_positions_vis", "ids_remove_pos"
                      Shapes: z_enc [B, gV, N+1, D], patch_positions_vis [B, gV, N, 2], ids_remove_pos [B, gV, M]
        :param l_enc: Local encoder outputs (same keys) with shapes for lV views.
        :return: Tuple (joint_latents, joint_patch_pos, joint_ids_remove) where:
                 joint_latents: [B, (gV+lV) + (gV*N_g + lV*N_l), D],
                 joint_patch_pos: [B, (gV*N_g + lV*N_l), 2],
                 joint_ids_remove: [B, (gV*M_g + lV*M_l)]
        """
        # Inline splitting: first token per view is cls; remaining are patch tokens.
        g_cls = g_enc["z_enc"][:, :, 0, :]  # [B, gV, D]
        g_patch = g_enc["z_enc"][:, :, 1:, :]  # [B, gV, tokens_g-1, D]
        l_cls = l_enc["z_enc"][:, :, 0, :]  # [B, lV, D]
        l_patch = l_enc["z_enc"][:, :, 1:, :]  # [B, lV, tokens_l-1, D]
        # Build joint latent sequence.
        joint_cls = torch.cat([g_cls, l_cls], dim=1)  # [B, gV+lV, D]
        joint_patch = torch.cat(
            [g_patch.flatten(1, 2), l_patch.flatten(1, 2)], dim=1
        )  # [B, total_patch, D]
        joint_latents = torch.cat([joint_cls, joint_patch], dim=1)
        # Build joint patch positions.
        joint_patch_pos = torch.cat(
            [
                g_enc["patch_positions_vis"].flatten(1, 2),
                l_enc["patch_positions_vis"].flatten(1, 2),
            ],
            dim=1,
        )
        # For global views.
        B, gV, M_g = g_enc["ids_remove_pos"].shape
        N_g = g_enc["patch_positions_vis"].shape[2]
        # number of patch tokens per global view
        global_offsets = (
            torch.arange(gV, device=g_enc["ids_remove_pos"].device) * N_g
        ).view(1, gV, 1)
        adjusted_global_ids = g_enc["ids_remove_pos"] + global_offsets  # [B, gV, M_g]

        # For local views.
        B, lV, M_l = l_enc["ids_remove_pos"].shape
        N_l = l_enc["patch_positions_vis"].shape[2]
        # number of patch tokens per local view
        # All global patch tokens come first, so offset local ids by (gV * N_g)
        local_offsets = gV * N_g + (
            torch.arange(lV, device=l_enc["ids_remove_pos"].device) * N_l
        ).view(1, lV, 1)
        adjusted_local_ids = l_enc["ids_remove_pos"] + local_offsets  # [B, lV, M_l]

        # Concatenate the adjusted indices.
        joint_ids_remove = torch.cat(
            [adjusted_global_ids.view(B, -1), adjusted_local_ids.view(B, -1)], dim=1
        )
        return joint_latents, joint_patch_pos, joint_ids_remove

    def forward_encoder(self, x: Tensor) -> dict:
        r"""
        Encode an image crop.

        :param x: Tensor of shape [B*n_views, C, H, W]
        :return: Dict with:
                 "z_enc": Tensor of shape [B*n_views, 1+N_vis, D] (1 class token + N_vis visible tokens),
                 "patch_positions_vis": Tensor of shape [B*n_views, N_vis, 2],
                 "ids_remove_pos": Tensor of shape [B*n_views, M] (M indices of masked tokens)
        """
        x, patch_positions_vis = self.patch_embed(x)
        B_total, N_vis, D = x.shape
        ids_keep, _, ids_restore, ids_remove = self.random_masking(
            x, self.pos_mask_ratio
        )
        N_pos = ids_keep.shape[1]
        N_nopos = N_vis - N_pos
        patch_positions_pos = torch.gather(
            patch_positions_vis, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 2)
        )
        pos_embed = get_2d_sincos_pos_embed(
            patch_positions_pos.flatten(0, 1) / self.patch_size, self.embed_dim
        )
        pos_embed = pos_embed.unflatten(0, (B_total, N_pos))
        mask_pos_tokens = self.mask_pos_token.expand(B_total, N_nopos, -1)
        pos_embed = torch.cat([pos_embed, mask_pos_tokens], dim=1)
        pos_embed = torch.gather(
            pos_embed, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D)
        )
        x = x + pos_embed
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return {
            "z_enc": x,
            "patch_positions_vis": patch_positions_vis,
            "ids_remove_pos": ids_remove,
        }

    def forward_decoder(self, z: Tensor) -> dict:
        r"""
        Decode latent representations.

        :param z: Tensor of shape [B, L, D]
        :return: Dict with "pose_pred": Tensor of shape [B, L, num_targets]
        """
        x = self.decoder_embed(z)
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return {"pose_pred": x}

    def _drop_pos(
        self, patch_pred: Tensor, patch_pos: Tensor, joint_ids_remove: Tensor
    ) -> tuple[Tensor, Tensor]:
        r"""
        Drop tokens from the joint patch predictions and patch positions.

        :param patch_pred: Tensor of shape [B, P, num_targets]
        :param patch_pos: Tensor of shape [B, P, 2]
        :param joint_ids_remove: Tensor of shape [B, M] (indices into patch tokens)
        :return: Tuple (pred_nopos, pos_nopos) with shapes [B, M, num_targets] and [B, M, 2]
        """
        pred_nopos = torch.gather(
            patch_pred,
            dim=1,
            index=joint_ids_remove.unsqueeze(-1).expand(-1, -1, self.num_targets),
        )
        pos_nopos = torch.gather(
            patch_pos, dim=1, index=joint_ids_remove.unsqueeze(-1).expand(-1, -1, 2)
        )
        return pred_nopos, pos_nopos

    def random_masking(self, x: Tensor, mask_ratio: float):
        r"""
        Randomly mask tokens.

        :param x: Tensor of shape [B, N, D]
        :param mask_ratio: Fraction of tokens to mask.
        :return: Tuple (ids_keep, mask, ids_restore, ids_remove) where:
                 ids_keep: Tensor of shape [B, N*(1-mask_ratio)]
                 mask: Tensor of shape [B, N] (boolean)
                 ids_restore: Tensor of shape [B, N]
                 ids_remove: Tensor of shape [B, N*mask_ratio]
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        ids_remove = ids_shuffle[:, len_keep:]
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore).bool()

        return ids_keep, mask, ids_restore, ids_remove

    def _compute_gt(
        self,
        patch_pos_nopos: Int[Tensor, "B T 2"],
        params: Int[Tensor, "B T 4"],
        crop_sizes: Float[Tensor, "B T 1"],
    ) -> Tensor:
        r"""
        Compute ground-truth pairwise differences using per-token crop sizes.

        :param patch_pos_nopos: Tensor of shape [B, T, 2]
        :param params: Tensor of shape [B, T, 4]
        :param crop_sizes: Tensor of shape [B, T, 1]
        :return: Tensor of shape [B, T, T, 4] (pairwise differences)
        """
        canonical_size = self.canonical_img_size

        # Extract crop parameters: [x,y] offsets and [width,height]
        offset = params[..., :2]  # Top-left corner coordinates
        size = params[..., 2:4]  # Width and height of the crop

        # Scale factor to map from crop coordinates to canonical space
        scale = size / crop_sizes

        # Transform patch positions to canonical space
        gt_patch_pos = offset + patch_pos_nopos * scale

        # Calculate size of patches in canonical space
        patch_canonical_size = self.patch_size * scale

        # Convert patch sizes to log space for more stable relative comparison
        gt_log_hw = torch.log(patch_canonical_size)

        # Combine position and size information
        gt_pose = torch.cat([gt_patch_pos, gt_log_hw], dim=-1)

        # Compute pairwise differences (for all possible token pairs)
        gt_dT = gt_pose.unsqueeze(2) - gt_pose.unsqueeze(1)  # [B, T, T, 4]

        # Normalize the differences for stable learning
        gt_dT[..., :2] /= canonical_size  # Normalize positions by canonical image size
        gt_dT[..., 2:] /= math.log(self.max_scale_ratio)  # Normalize log sizes

        return gt_dT

    def _compute_pred(self, pose_pred_nopos: Tensor) -> Tensor:
        r"""
        Compute pairwise differences of predicted transforms.

        :param pose_pred_nopos: Tensor of shape [B, T, 4]
        :return: Tensor of shape [B, T, T, 4] (pairwise differences)
        """
        pose_u = pose_pred_nopos.unsqueeze(2)
        pose_v = pose_pred_nopos.unsqueeze(1)
        return self.tanh(pose_u - pose_v)

    def _forward_loss(
        self,
        pose_pred_nopos: Tensor,
        patch_pos_nopos: Tensor,
        params: Tensor,
        shapes: tuple,
    ) -> dict:
        r"""
        Compute intra-crop and inter-crop losses.

        :param pose_pred_nopos: Tensor of shape [B, T, 4]
        :param patch_pos_nopos: Tensor of shape [B, T, 2]
        :param params: Tensor of shape [B, T, 4]
        :param shapes: Tuple ((global_size, M_g, gV), (local_size, M_l, lV))
        :return: Dict with loss components and prediction information.
        """
        B, T, _ = pose_pred_nopos.shape
        device = pose_pred_nopos.device
        global_size, M_g, gV = shapes[0]
        local_size, M_l, lV = shapes[1]

        global_crop_sizes = torch.full((gV * M_g, 1), global_size, device=device)
        local_crop_sizes = torch.full((lV * M_l, 1), local_size, device=device)
        token_crop_sizes = (
            torch.cat([global_crop_sizes, local_crop_sizes], dim=0)
            .unsqueeze(0)
            .expand(B, -1, -1)
        )

        pred_dT = self._compute_pred(pose_pred_nopos)
        gt_dT = self._compute_gt(patch_pos_nopos, params, token_crop_sizes)
        loss_full = self.criterion(pred_dT, gt_dT, reduction="none")

        global_labels = torch.arange(gV, device=device).repeat_interleave(M_g)
        local_labels = torch.arange(gV, gV + lV, device=device).repeat_interleave(M_l)
        labels = torch.cat([global_labels, local_labels], dim=0)
        intra_mask = (
            (labels.unsqueeze(0) == labels.unsqueeze(1))
            .float()
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(B, -1, -1, 1)
        )
        inter_mask = 1 - intra_mask

        loss_t_full = loss_full[..., :2]
        loss_s_full = loss_full[..., 2:]
        diag_denom = intra_mask.sum()
        offdiag_denom = inter_mask.sum()
        loss_intra_t = (loss_t_full * intra_mask).sum() / diag_denom
        loss_inter_t = (loss_t_full * inter_mask).sum() / offdiag_denom
        loss_intra_s = (loss_s_full * intra_mask).sum() / diag_denom
        loss_inter_s = (loss_s_full * inter_mask).sum() / offdiag_denom
        loss_t = loss_inter_t * self.alpha_t + loss_intra_t * (1 - self.alpha_t)
        loss_s = loss_inter_s * self.alpha_s + loss_intra_s * (1 - self.alpha_s)
        loss = self.alpha_ts * loss_t + (1 - self.alpha_ts) * loss_s

        return {
            "loss_intra_t": loss_intra_t,
            "loss_inter_t": loss_inter_t,
            "loss_intra_s": loss_intra_s,
            "loss_inter_s": loss_inter_s,
            "loss_t": loss_t,
            "loss_s": loss_s,
            "loss": loss,
            "pred_dT": pred_dT,
            "gt_dT": gt_dT,
        }

    def forward(
        self,
        g_x: Float[Tensor, "B gV C gH gW"],
        g_params: Float[Tensor, "B gV 8"],
        l_x: Float[Tensor, "B lV C lH lW"],
        l_params: Float[Tensor, "B lV 8"],
    ) -> dict:
        r"""
        Forward pass:
          - Encode global and local views.
          - Add view-specific segment embeddings.
          - Split class and patch tokens per view.
          - Prepare joint inputs (latent sequence, patch positions, drop indices) in one step.
          - Decode jointly; discard the class tokens.
          - Drop tokens from the joint patch predictions and patch positions.
          - Expand augmentation parameters (using crop params from indices 4:8) and compute loss.

        :param g_x: Global views [B, gV, C, gH, gW]
        :param g_params: Global augmentation parameters [B, gV, 8] (crop parameters are columns 4:8)
        :param l_x: Local views [B, lV, C, lH, lW]
        :param l_params: Local augmentation parameters [B, lV, 8] (crop parameters are columns 4:8)
        :return: Dict with loss components and auxiliary outputs.
        """
        _, gV, _, _, _ = g_x.shape
        _, lV, _, _, _ = l_x.shape
        V = gV + lV

        # Encode views.
        g_enc = self.encode_views(g_x)
        l_enc = self.encode_views(l_x)

        if self.segment_embed_mode == "permute":
            perm = torch.randperm(V)

            # Randomly assign embeddings to global and local views based on their counts.
            g_perm = perm[:gV]
            l_perm = perm[gV : gV + lV]

            g_segment = self.segment_embed[g_perm]
            l_segment = self.segment_embed[l_perm]
        else:
            g_segment = self.segment_embed[:gV]
            l_segment = self.segment_embed[gV : gV + lV]

        # Add view-specific segment embeddings.
        g_enc["z_enc"] = g_enc["z_enc"] + g_segment.unsqueeze(0).unsqueeze(2)
        l_enc["z_enc"] = l_enc["z_enc"] + l_segment.unsqueeze(0).unsqueeze(2)

        # Prepare joint inputs.
        joint_latents, joint_patch_pos, joint_ids_remove = self.prepare_joint_inputs(
            g_enc, l_enc
        )

        # Decode jointly.
        dec_out = self.forward_decoder(joint_latents)  # [B, L, num_targets]
        total_cls = gV + lV
        patch_pred = dec_out["pose_pred"][:, total_cls:, :]

        # Drop tokens from the joint patch predictions.
        pose_pred_nopos, patch_pos_nopos = self._drop_pos(
            patch_pred, joint_patch_pos, joint_ids_remove
        )

        # Use crop parameters from columns 4:8.
        g_params_crop = g_params[:, :, 4:8]
        l_params_crop = l_params[:, :, 4:8]

        M_g = g_enc["ids_remove_pos"].shape[2]
        M_l = l_enc["ids_remove_pos"].shape[2]
        g_params_exp = g_params_crop.unsqueeze(2).expand(-1, -1, M_g, -1).flatten(1, 2)
        l_params_exp = l_params_crop.unsqueeze(2).expand(-1, -1, M_l, -1).flatten(1, 2)
        params_all = torch.cat([g_params_exp, l_params_exp], dim=1)

        shapes = ((g_x.shape[-2], M_g, gV), (l_x.shape[-2], M_l, lV))
        loss_dict = self._forward_loss(
            pose_pred_nopos, patch_pos_nopos, params_all, shapes
        )
        out = {"patch_positions_nopos": patch_pos_nopos, "shapes": shapes}
        out.update(loss_dict)
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
        **kwargs,
    )
    return model


# set recommended archs
PART_mae_vit_base_patch16 = (
    PART_mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
)

if __name__ == "__main__":
    from src.data.components.transforms.multi_crop_v2 import ParametrizedMultiCropV2
    from PIL import Image
    import torch.utils._pytree as pytree
    from lightning import seed_everything

    seed_everything(42)
    backbone = PART_mae_vit_base_patch16(pos_mask_ratio=0.75, mask_ratio=0.75).cuda()
    gV, lV = 2, 4
    V = gV + lV
    t = ParametrizedMultiCropV2(n_global_crops=gV, n_local_crops=lV, distort_color=True)
    print(t.compute_max_scale_ratio_aug())  # <5.97

    class MockedDataset(torch.utils.data.Dataset):
        def __init__(self, transform=None, n: int = 4):
            self.img = Image.open("artifacts/labrador.jpg")
            self.transform = transform
            self.n = n

        def __getitem__(self, idx):
            return self.transform(self.img)

        def __len__(self):
            return self.n

    dataset = MockedDataset(t)
    seed_everything(42)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    seed_everything(42)
    batch = next(iter(loader))
    batch = pytree.tree_map_only(
        Tensor, lambda x: x.cuda(), batch
    )  # Move to GPU if available

    seed_everything(42)
    out = backbone(*batch)
    print("Output keys:", out.keys())
    print("loss", out["loss"].detach().item())