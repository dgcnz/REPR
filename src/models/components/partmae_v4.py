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

SAMPLERS = {
    "random": random_sampling,
    "stratified_jittered": stratified_jittered_sampling,
    "ongrid": ongrid_sampling,
    "ongrid_canonical": ongrid_sampling_canonical,
}


class PARTMaskedAutoEncoderViT(nn.Module):
    """
    Self-supervised training by generalized patch-level transforms between augmented views of an image.

    CHANGELOG:
    - Augmented views:
        - Dataset generates n_views augmented views per image.
        - Each view is augmented with random crop and resize.
        - Model receives n_views consecutive views per image and the corresponding augmentation parameters (y, x, h, w).
        - Model encodes and decodes each view separately but computes the scale ratio and patch distances across and within views.
        - This framework allows for generalized patch-level transforms between augmented views (scale, rotation).
        - New: Handles multi-crop, 2 (224x224) + 6 (96x96) crops.
    """

    def __init__(
        self,
        # Encoder parameters
        img_size: int = 224,  # global crop size (e.g., 224)
        # canonical_img_size and max_scale_ratio are used for normalization.
        canonical_img_size: int = 512,  # canonical (original) image size (e.g., 512)
        max_scale_ratio: float = 6.0,  # maximum scale ratio for normalization
        ###  ViT parameters
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        norm_layer: type = nn.LayerNorm,
        mask_ratio: float = 0.75,
        pos_mask_ratio: float = 0.75,
        # Decoder parameters
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        # PART parameters
        sampler: str = "random",
        criterion: str = "l1",
        # Loss weighting parameters.
        alpha_t: float = 0.5,  # weight between inter/intra translation
        alpha_ts: float = 0.5,  # weight between translation and scale
        alpha_s: float = 0.75,  # weight between inter/intra scale
        # Debugging parameters
        verbose: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size = img_size  # crop size
        self.canonical_img_size = canonical_img_size  # canonical image size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.pos_mask_ratio = pos_mask_ratio
        self.max_scale_ratio = max_scale_ratio
        self.alpha_ts = alpha_ts
        self.alpha_t = alpha_t
        self.alpha_s = alpha_s  # weight between inter/intra scale

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
        self.num_targets = 4
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.num_targets, bias=False)
        self.tanh = nn.Tanh()
        self.verbose = verbose
        self.initialize_weights()

    def update_conf(
        self,
        sampler: str = None,
        mask_ratio: float = None,
        pos_mask_ratio: float = None,
    ):
        if mask_ratio is not None:
            self.mask_ratio = mask_ratio
            self.patch_embed.mask_ratio = mask_ratio
        if pos_mask_ratio is not None:
            self.pos_mask_ratio = pos_mask_ratio
        if sampler is not None:
            self.patch_embed.sampler = SAMPLERS[sampler]

    def initialize_weights(self):
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
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _drop_pos(
        self, pose_pred: Tensor, patch_positions: Tensor, ids_remove_pos: Tensor
    ):
        """
        Drop from pose_pred and patch_positions the tokens that have positional embeddings.
        Args:
          pose_pred: [B*n_views, N_vis, 4]
          patch_positions: [B*n_views, N_vis, 2]
          ids_remove_pos: [B*n_views, N_nopos]
        Returns:
          pose_pred_nopos: [B*n_views, N_nopos, 4]
          patch_positions_nopos: [B*n_views, N_nopos, 2]
        """
        pose_pred_nopos = torch.gather(
            pose_pred,
            dim=1,
            index=ids_remove_pos.unsqueeze(-1).expand(-1, -1, self.num_targets),
        )
        patch_positions_nopos = torch.gather(
            patch_positions, dim=1, index=ids_remove_pos.unsqueeze(-1).expand(-1, -1, 2)
        )
        return pose_pred_nopos, patch_positions_nopos

    def random_masking(self, x: Tensor, mask_ratio: float):
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

    def forward_encoder(self, x: Tensor):
        # x: [B*n_views, C, H, W] (crops of size img_size, e.g. 224)
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
        # x = x + pos_embed
        x.add_(pos_embed)
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

    def forward_decoder(self, z: Tensor):
        # Decode patch tokens to transformation predictions.
        x = self.decoder_embed(z)
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return {"pose_pred": x[:, 1:, :]}  # [B*n_views, N_vis, 4]

    # --- Ground Truth Computation using Canonical Mapping ---
    # Here, each crop is assumed to be of fixed size self.img_size (e.g., 224),
    # and the canonical image size is self.canonical_img_size (e.g., 512).
    # The augmentation parameters (y, x, h, w) specify the crop in canonical coordinates.
    # A patch at coordinate pos in the crop is mapped to canonical coordinates as:
    #    canonical_pos = (y, x) + (pos / self.img_size) * (h, w)
    # Ground truth differences are then computed in canonical coordinates and normalized by self.canonical_img_size.

    def _forward_branch(
        self, x: Float[Tensor, "B V C H W"], params: Float[Tensor, "B V 8"]
    ):
        out = dict()
        B, V, _, H, W = x.shape
        assert H == W, "Input image must be square."
        # Drop the first 4 canonical parameters; keep only augmentation params.
        params = params[:, :, 4:]
        _out_enc = self.forward_encoder(x.flatten(0, 1))
        _out_dec = self.forward_decoder(_out_enc["z_enc"])
        pose_pred_nopos, patch_positions_nopos = self._drop_pos(
            _out_dec["pose_pred"],
            _out_enc["patch_positions_vis"],
            _out_enc["ids_remove_pos"],
        )
        N_nopos = pose_pred_nopos.shape[1]
        # Reshape so that tokens from each crop are contiguous.
        out["pose_pred_nopos"] = pose_pred_nopos.reshape(B, V * N_nopos, -1)
        out["patch_positions_nopos"] = patch_positions_nopos.reshape(B, V * N_nopos, 2)
        # TODO: this seems very inefficient
        out["params"] = params.unsqueeze(2).expand(-1, -1, N_nopos, -1).flatten(1, 2)
        # Store branch-specific shape: (crop_size, tokens_per_crop, number_of_crops)
        out["shapes"] = (H, N_nopos, V)

        if self.verbose:
            out["patch_positions_vis"] = _out_enc["patch_positions_vis"].detach()
        return out

    def forward(
        self,
        g_x: Float[Tensor, "B gV C gH gW"],
        g_params: Float[Tensor, "B gV 8"],
        l_x: Float[Tensor, "B lV C lH lW"],
        l_params: Float[Tensor, "B lV 8"],
    ) -> dict:
        # Process each branch (global and local) separately.
        g_out = self._forward_branch(g_x, g_params)
        l_out = self._forward_branch(l_x, l_params)

        out = dict(
            g_patch_positions_nopos=g_out["patch_positions_nopos"],
            l_patch_positions_nopos=l_out["patch_positions_nopos"],
            g_shapes=g_out["shapes"],
            l_shapes=l_out["shapes"],
        )
        if self.verbose:
            out["g_patch_positions_vis"] = g_out["patch_positions_vis"]
            out["l_patch_positions_vis"] = l_out["patch_positions_vis"] 

        # Concatenate global and local branches.
        pose_pred_nopos = torch.cat(
            [g_out["pose_pred_nopos"], l_out["pose_pred_nopos"]], dim=1
        )
        patch_positions_nopos = torch.cat(
            [g_out["patch_positions_nopos"], l_out["patch_positions_nopos"]], dim=1
        )
        params = torch.cat([g_out["params"], l_out["params"]], dim=1)
        # Combine the shape information from both branches.
        shapes = (g_out["shapes"], l_out["shapes"])
        out.update(
            self._forward_loss(pose_pred_nopos, patch_positions_nopos, params, shapes)
        )
        return out

    @torch.no_grad()
    def _compute_gt(
        self,
        patch_positions_nopos: Int[Tensor, "B T 2"],
        params: Int[Tensor, "B T 4"],
        crop_sizes: Float[Tensor, "B T 1"],
    ):
        """
        Compute ground-truth pairwise differences using per-token crop sizes.
        Args:
          patch_positions_nopos: [B, T, 2] patch positions in crop coordinates.
          params: [B, T, 4] augmentation parameters (y, x, h, w) for each token.
          crop_sizes: [B, T, 1] crop size (e.g. 224 or 96) for each token.
        Returns:
          gt_dT: [B, T, T, 4] normalized differences.
        """
        canonical_size = self.canonical_img_size
        offset = params[..., :2]  # [B, T, 2]
        size = params[..., 2:4]  # [B, T, 2]
        scale = size / crop_sizes  # [B, T, 2]
        gt_patch_positions = offset + patch_positions_nopos * scale  # [B, T, 2]
        patch_canonical_size = self.patch_size * scale  # [B, T, 2]
        gt_log_hw = torch.log(patch_canonical_size)  # [B, T, 2]
        gt_pose = torch.cat([gt_patch_positions, gt_log_hw], dim=-1)  # [B, T, 4]
        gt_dT = gt_pose.unsqueeze(2) - gt_pose.unsqueeze(1)  # [B, T, T, 4]
        gt_dT[..., :2] /= canonical_size
        gt_dT[..., 2:] /= math.log(self.max_scale_ratio)
        return gt_dT

    def _compute_pred(self, pose_pred_nopos: Tensor) -> Tensor:
        """
        Computes the pairwise differences of the predicted transforms.
        Input:
          pose_pred_nopos: [B, T, 4] predicted transforms (flattened over crops).
        Returns:
          pred_dT: [B, T, T, 4]
        """
        pose_u = pose_pred_nopos.unsqueeze(2)  # [B, T, 1, 4]
        pose_v = pose_pred_nopos.unsqueeze(1)  # [B, 1, T, 4]
        pred_dT = self.tanh(pose_u - pose_v)  # [B, T, T, 4]
        return pred_dT

    def _forward_loss(
        self,
        pose_pred_nopos: Tensor,
        patch_positions_nopos: Tensor,
        params: Tensor,
        shapes: tuple,  # shapes is a tuple: (g_shapes, l_shapes) where each is (crop_size, tokens_per_crop, n_crops)
    ) -> dict:
        """
        Compute separate intra-crop and inter-crop losses.
        Args:
          pose_pred_nopos: [B, T, 4] predicted transforms.
          patch_positions_nopos: [B, T, 2] patch positions in crop coordinates.
          params: [B, T, 4] augmentation parameters (y, x, h, w) for each token.
          shapes: tuple of two tuples: (g_shapes, l_shapes) where
                  g_shapes = (global_crop_size, global_tokens_per_crop, num_global_crops)
                  l_shapes = (local_crop_size, local_tokens_per_crop, num_local_crops)
                  Total T = num_global_crops * global_tokens_per_crop + num_local_crops * local_tokens_per_crop.
        Returns:
          A dictionary with various loss components.
        """
        B, T, _ = pose_pred_nopos.shape
        device = pose_pred_nopos.device
        # Unpack shapes.
        global_size, global_N, gV = shapes[0]
        local_size, local_N, lV = shapes[1]

        # Build per-token crop size vector.
        global_crop_sizes = torch.full((gV * global_N, 1), global_size, device=device)
        local_crop_sizes = torch.full((lV * local_N, 1), local_size, device=device)
        token_crop_sizes = torch.cat(
            [global_crop_sizes, local_crop_sizes], dim=0
        )  # [T, 1]
        token_crop_sizes = token_crop_sizes.unsqueeze(0).expand(B, -1, -1)  # [B, T, 1]

        pred_dT = self._compute_pred(pose_pred_nopos)  # [B, T, T, 4]
        gt_dT = self._compute_gt(
            patch_positions_nopos, params, token_crop_sizes
        )  # [B, T, T, 4]
        loss_full = self.criterion(pred_dT, gt_dT, reduction="none")  # [B, T, T, 4]

        # Build a label vector for tokens: global crops get labels 0,...,gV-1, local crops get labels gV,...,gV+lV-1.
        global_labels = torch.arange(gV, device=device).repeat_interleave(global_N)
        local_labels = torch.arange(gV, gV + lV, device=device).repeat_interleave(
            local_N
        )
        labels = torch.cat([global_labels, local_labels], dim=0)  # [T]
        intra_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # [T, T]
        intra_mask = (
            intra_mask.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, 1)
        )  # [B, T, T, 1]
        inter_mask = 1 - intra_mask

        # Split loss into translation (first 2 channels) and scale (last 2 channels).
        loss_t_full = loss_full[..., :2]  # [B, T, T, 2]
        loss_s_full = loss_full[..., 2:]  # [B, T, T, 2]

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
    backbone = PART_mae_vit_base_patch16(pos_mask_ratio=0.75, mask_ratio=0.75)
    # x = torch.randn(1, 3, 224, 224).repeat(2, 1, 1, 1)
    # params = torch.tensor([[0, 0, 224, 224], [0, 0, 224, 224]])
    # patch_size = 16
    # num_patches = (224 // patch_size) ** 2
    # out = backbone.forward(x, params)
    # print("Output keys:", out.keys())
    # print(out["loss"])

    # Test forward pass
    B, gV, lV = 4, 2, 6
    g_x = torch.randn(B, gV, 3, 224, 224)
    g_params = torch.randn(B, gV, 8)
    l_x = torch.randn(B, lV, 3, 96, 96)
    l_params = torch.randn(B, lV, 8)
    out = backbone(g_x, g_params, l_x, l_params)
    print("Output keys:", out.keys())
