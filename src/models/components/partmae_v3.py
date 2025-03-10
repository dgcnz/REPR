import torch
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
    """

    def __init__(
        self,
        n_views: int = 2,  # number of views per image
        # Encoder parameters
        img_size: int = 224,  # crop size (e.g., 224)
        # canonical_img_size and max_scale_ratio are used for normalization.
        canonical_img_size: int = 512,  # canonical (original) image size (e.g., 512)
        max_scale_ratio: float = 4.0,  # maximum scale ratio for normalization
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer: type = nn.LayerNorm,
        mask_ratio: float = 0.75,
        pos_mask_ratio: float = 0.75,
        # Decoder parameters
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        # Prediction head outputs 4 numbers per patch:
        # first 2 for translation; last 2 for log-scale.
        num_targets: int = 4,
        # PART parameters
        sampler: str = "random",
        criterion: str = "l1",
        # Loss weighting parameters.
        alpha_t: float = 0.5,  # weight between inter/intra translation
        alpha_ts: float = 0.5,  # weight between translation and scale
        alpha_s: float = 0.75,  # weight between inter/intra scale
    ):
        super().__init__()
        self.n_views = n_views
        self.embed_dim = embed_dim
        self.img_size = img_size  # crop size
        self.canonical_img_size = canonical_img_size  # canonical image size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.pos_mask_ratio = pos_mask_ratio
        self.num_targets = num_targets
        self.max_scale_ratio = max_scale_ratio
        self.alpha_ts = alpha_ts
        self.alpha_t = alpha_t
        self.alpha_s = alpha_s  # weight between inter/intra scale

        self.patch_embed = OffGridPatchEmbed(
            img_size=img_size,  # note: patch embedding operates on the crop size
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
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, num_targets, bias=False)
        self.tanh = nn.Tanh()
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
        pos_embed = get_canonical_pos_embed(self.embed_dim, grid_size, self.patch_size)
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
        mask_pos_tokens = self.mask_pos_token.repeat(B_total, N_nopos, 1)
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
            "z_enc": x,  # [B*n_views, N_vis+1, D]
            "patch_positions_vis": patch_positions_vis,  # [B*n_views, N_vis, 2]
            "ids_remove_pos": ids_remove,  # [B*n_views, N_mask]
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

    def _compute_gt(
        self,
        patch_positions_nopos: Int[Tensor, "B V N_nopos 2"],
        params: Int[Tensor, "B V 4"],
    ):
        """
        patch_positions_nopos: [B, V, N_nopos, 2] patch positions in crop coordinates.
        params: [B, V, 4] augmentation parameters (y, x, h, w) in canonical coordinates.
        Returns:
          gt_dt: [B, V, V, N_nopos, N_nopos, 2] normalized translation differences.
          gt_ds: [B, V, V, N_nopos, N_nopos, 2] normalized log-scale ratios.
        """
        crop_size = self.img_size
        canonical_size = self.canonical_img_size
        params = params.unsqueeze(2)  # [B, V, 1, 4]

        # --- Compute patch positions in canonical coordinates ---
        offset = params[..., :2]  # [B, V, 1, 2]
        size = params[..., 2:4]  # [B, V, 1, 2]
        # gt_patch_positions: [B, V, N_nopos, 2]
        gt_patch_positions = offset + (patch_positions_nopos / crop_size) * size

        # --- Compute log scale parameters ---
        # gt_log_hw: [B, V, N_nopos, 2]
        gt_log_hw = torch.log(size).expand_as(gt_patch_positions)

        # --- Stack positions and scales into a single representation ---
        # gt_pose [B, V, N_nopos, 4]
        gt_pose = torch.cat([gt_patch_positions, gt_log_hw], dim=-1)

        # --- Create source and target view representations ---
        # [B, 1, V, N_nopos, 1, 4]
        pose_i = gt_pose.unsqueeze(1).unsqueeze(4)
        # [B, V, 1, 1, N_nopos, 4]
        pose_j = gt_pose.unsqueeze(2).unsqueeze(3)

        # --- Compute differences in one go ---
        gt_dT = pose_j - pose_i  # [B, V, V, N_nopos, N_nopos, 4]

        # --- Split and normalize ---
        gt_dt = gt_dT[..., :2] / canonical_size  # [B, V, V, N_nopos, N_nopos, 2]
        gt_ds = gt_dT[..., 2:] / self.max_scale_ratio  # [B, V, V, N_nopos, N_nopos, 2]

        return {"gt_dt": gt_dt, "gt_ds": gt_ds}

    def _compute_pred(self, pose_pred_nopos: Tensor) -> dict:
        """
        pose_pred_nopos: [B, V, N_nopos, 4]
        Returns:
          pred_dt: [B, V, V, N_nopos, N_nopos, 2]
          pred_ds: [B, V, V, N_nopos, N_nopos, 2]
        """
        # Create source (u) and target (v) view representations with one unsqueeze operation each
        pose_u = pose_pred_nopos.unsqueeze(1).unsqueeze(4)  # [B, 1, V, N_nopos, 1, 4]
        pose_v = pose_pred_nopos.unsqueeze(2).unsqueeze(3)  # [B, V, 1, 1, N_nopos, 4]

        # Compute differences for all transforms at once and apply tanh
        pred_dT = self.tanh(pose_v - pose_u)  # [B, V, V, N_nopos, N_nopos, 4]

        # Split the result into translation and scale components
        pred_dt = pred_dT[..., :2]  # [B, V, V, N_nopos, N_nopos, 2]
        pred_ds = pred_dT[..., 2:]  # [B, V, V, N_nopos, N_nopos, 2]

        return {"pred_dt": pred_dt, "pred_ds": pred_ds}

    def _forward_loss(
        self, pose_pred_nopos: Tensor, patch_positions_nopos: Tensor, params: Tensor
    ) -> dict:
        """
        Compute separated loss for inter‐view and intra‐view predictions.
        Args:
        pose_pred_nopos: [B, V, N_nopos, 4] predicted transforms for each view.
        patch_positions_nopos: [B, V, N_nopos, 2] patch positions in crop coordinates.
        params: [B, V, 4] augmentation parameters (y, x, h, w) in canonical coordinates.
        Returns:
        A dictionary with:
            - loss_intra_t: loss on the intra–view (diagonal) patch differences.
            - loss_inter_t: loss on the off–diagonal (true inter–view) patch differences.
            - loss_intra_s: loss on the intra-view scale differences.
            - loss_inter_s: loss on the inter–view scale differences.
            - And the separated prediction and ground truth tensors (for debugging).
        """
        # Compute the full inter-view predictions and ground truth.
        pred_dict = self._compute_pred(pose_pred_nopos)
        gt_dict = self._compute_gt(patch_positions_nopos, params)

        # Both tensors have shape [B, V, V, N_nopos, N_nopos, 2]
        # Extract the diagonal (intra-view): compare patches within the same view.
        pred_intra_dt = torch.diagonal(pred_dict["pred_dt"], dim1=1, dim2=2)
        gt_intra_dt = torch.diagonal(gt_dict["gt_dt"], dim1=1, dim2=2)
        loss_intra_t = self.criterion(pred_intra_dt, gt_intra_dt)

        # Also extract intra-view scale predictions (diagonal elements)
        pred_intra_ds = torch.diagonal(pred_dict["pred_ds"], dim1=1, dim2=2)
        gt_intra_ds = torch.diagonal(gt_dict["gt_ds"], dim1=1, dim2=2)
        loss_intra_s = self.criterion(pred_intra_ds, gt_intra_ds)

        # Extract off-diagonals (inter-view): unique comparisons where u != v.
        V = pose_pred_nopos.shape[1]
        triu_idx = torch.triu_indices(V, V, offset=1, device=pose_pred_nopos.device)
        pred_inter_dt = pred_dict["pred_dt"][:, triu_idx[0], triu_idx[1], :, :, :]
        gt_inter_dt = gt_dict["gt_dt"][:, triu_idx[0], triu_idx[1], :, :, :]
        loss_inter_t = self.criterion(pred_inter_dt, gt_inter_dt)

        # For inter-view scale
        pred_inter_ds = pred_dict["pred_ds"][:, triu_idx[0], triu_idx[1], :, :, :]
        gt_inter_ds = gt_dict["gt_ds"][:, triu_idx[0], triu_idx[1], :, :, :]
        loss_inter_s = self.criterion(pred_inter_ds, gt_inter_ds)

        # Combine losses with weighting.
        loss_t = loss_inter_t * self.alpha_t + loss_intra_t * (1 - self.alpha_t)

        # Apply alpha_s weighting between inter-view and intra-view scale losses
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
            "pred_intra_dt": pred_intra_dt,
            "gt_intra_dt": gt_intra_dt,
            "pred_inter_dt": pred_inter_dt,
            "gt_inter_dt": gt_inter_dt,
            "pred_inter_ds": pred_inter_ds,
            "gt_inter_ds": gt_inter_ds,
            "pred_intra_ds": pred_intra_ds,
            "gt_intra_ds": gt_intra_ds,
        }

    def forward(self, x: Tensor, params: Tensor):
        """
        Args:
        x: [B*n_views, C, H, W] with n_views per image arranged consecutively.
        params: [B*n_views, 4] augmentation parameters (y, x, h, w) for each view.
        """
        out = dict()
        B_total, _, H, W = x.shape
        assert H == W, "Input image must be square."
        V, nT = self.n_views, self.num_targets
        B = B_total // V

        # Run encoder and decoder.
        out.update(self.forward_encoder(x))
        out.update(self.forward_decoder(out["z_enc"]))

        # Drop the positional tokens to use only masked tokens.
        pose_pred_nopos, patch_positions_nopos = self._drop_pos(
            out["pose_pred"], out["patch_positions_vis"], out["ids_remove_pos"]
        )

        # Compute separated inter/intra loss.
        out.update(
            self._forward_loss(
                pose_pred_nopos.view(B, V, -1, nT),  # [B, V, N_mask, 4]
                patch_positions_nopos.view(B, V, -1, 2),  # [B, V, N_mask, 2]
                params.view(B, V, 4),
            )
        )

        return out


def PART_mae_vit_base_patch16_dec512d8b(alpha_s=0.5, **kwargs):
    model = PARTMaskedAutoEncoderViT(
        n_views=2,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        alpha_s=alpha_s,  # Add the new parameter
        **kwargs
    )
    return model


# set recommended archs
PART_mae_vit_base_patch16 = (
    PART_mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
)

if __name__ == "__main__":
    backbone = PART_mae_vit_base_patch16(pos_mask_ratio=0.4, mask_ratio=0.3)
    x = torch.randn(1, 3, 224, 224).repeat(2, 1, 1, 1)
    params = torch.tensor([[0, 0, 224, 224], [0, 0, 224, 224]])
    patch_size = 16
    num_patches = (224 // patch_size) ** 2
    out = backbone.forward(x, params)
    print("Output keys:", out.keys())
    print(out["loss"])
