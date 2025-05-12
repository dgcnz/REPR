from itertools import groupby, accumulate
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
from src.models.components.heads.dino_head import DINOHead
from src.models.components.loss.simdino import MCRLoss


class PoseLoss(nn.Module):
    def __init__(
        self,
        criterion: str = "mse",
        alpha_t: float = 0.5,
        alpha_s: float = 0.75,
        alpha_ts: float = 0.5,
    ):
        super().__init__()
        self.criterion = mse_loss if criterion == "mse" else l1_loss
        self.alpha_t = alpha_t
        self.alpha_s = alpha_s
        self.alpha_ts = alpha_ts

    def forward(self, pred_dT: Tensor, gt_dT: Tensor, mask: Tensor) -> Tensor:
        loss_full = self.criterion(pred_dT, gt_dT, reduction="none")

        diag, offdiag = mask.sum(), (1 - mask).sum()

        loss_intra_t = (loss_full[..., :2] * mask).sum() / diag
        loss_inter_t = (loss_full[..., :2] * (1 - mask)).sum() / offdiag
        loss_intra_s = (loss_full[..., 2:] * mask).sum() / diag
        loss_inter_s = (loss_full[..., 2:] * (1 - mask)).sum() / offdiag
        loss_t = self.alpha_t * loss_inter_t + (1 - self.alpha_t) * loss_intra_t
        loss_s = self.alpha_s * loss_inter_s + (1 - self.alpha_s) * loss_intra_s
        loss = self.alpha_ts * loss_t + (1 - self.alpha_ts) * loss_s
        return {
            "loss_pose_intra_t": loss_intra_t,
            "loss_pose_inter_t": loss_inter_t,
            "loss_pose_intra_s": loss_intra_s,
            "loss_pose_inter_s": loss_inter_s,
            "loss_pose_t": loss_t,
            "loss_pose_s": loss_s,
            "loss_pose": loss,
        }


class PatchLoss(nn.Module):
    def __init__(self, sigma: tuple[float, float, float, float] = (0.1, 0.1, 0.1, 0.1)):
        super().__init__()
        self.register_buffer("sigma", torch.tensor(sigma, dtype=torch.float32))

    def forward(self, z: torch.Tensor, gt_dT: torch.Tensor) -> torch.Tensor:
        _, T, _ = z.shape

        # 1) compute weights w_ij
        kernel = (gt_dT.pow(2) / (2 * self.sigma.pow(2))).sum(dim=-1)  # [B,T,T]
        w = torch.exp(-kernel)

        # 2) precompute ‖z_i‖²
        z2 = (z * z).sum(dim=-1)  # [B,T]

        # 3) term1 = Σ_i ‖z_i‖² (Σ_j w_ij)
        w_col_sum = w.sum(dim=2)  # [B,T] (sum over j for fixed i)
        term1 = (z2 * w_col_sum).sum(dim=1)  # [B]

        # 4) term2 = Σ_i z_iᵀ (Σ_j w_ij z_j)
        wz = torch.bmm(w, z)  # [B,T,D]
        term2 = (wz * z).sum(dim=(1, 2))  # [B]

        # 5) term3 = Σ_j ‖z_j‖² (Σ_i w_ij)
        w_row_sum = w.sum(dim=1)  # [B,T] (sum over i for fixed j)
        term3 = (z2 * w_row_sum).sum(dim=1)  # [B]

        # 6) combine & normalize
        # Loss per batch item = term1 - 2*term2 + term3
        # mean over patch pairs (i,j) and batch
        loss = (term1 - 2 * term2 + term3).mean() / (T * T)
        return loss


class CosineAlignmentLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred_dT: Tensor, gt_dT: Tensor) -> Tensor:
        # pred_dT, gt_dT: [B, T, T, C]
        B, _, _, C = pred_dT.shape
        # flatten the pair dims → [B, P, C], where P=T*T
        pred = pred_dT.view(B, -1, C)
        gt = gt_dT.view(B, -1, C)
        # normalize each vector to unit length
        pred_norm = pred / (pred.norm(dim=-1, keepdim=True) + self.eps)
        gt_norm = gt / (gt.norm(dim=-1, keepdim=True) + self.eps)
        # cosine similarity per pair
        cos_sim = (pred_norm * gt_norm).sum(dim=-1)  # [B, P]
        # loss = 1 − cos_sim, then mean over all pairs and batch
        # this is the same as -cos_sim but easier to interpret
        return (1 - cos_sim).mean()


class PoseHead(nn.Module):
    def __init__(self, embed_dim: int, num_targets: int, apply_tanh: bool = False):
        super().__init__()
        self.linear = nn.Linear(embed_dim, num_targets, bias=False)
        self.tanh = nn.Tanh() if apply_tanh else nn.Identity()

    def forward(self, z, ids_remove):
        patch_pred = self.linear(z)
        pose_pred_nopos = _drop_pos(patch_pred, ids_remove)
        pose_u = pose_pred_nopos.unsqueeze(2)
        pose_v = pose_pred_nopos.unsqueeze(1)
        pred_dT = self.tanh(pose_u - pose_v)
        return pred_dT


def _drop_pos(
    x: Float[Tensor, "B N K"], ids_remove: Int[Tensor, "B M"]
) -> Float[Tensor, "B M K"]:
    idx = ids_remove.unsqueeze(-1).expand(-1, -1, x.shape[-1])
    return torch.gather(x, dim=1, index=idx)


SAMPLERS = {
    "random": random_sampling,
    "stratified_jittered": stratified_jittered_sampling,
    "ongrid": ongrid_sampling,
    "ongrid_canonical": ongrid_sampling_canonical,
}


class PARTMaskedAutoEncoderViT(nn.Module):
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
        verbose: bool = False,
        num_views: int = 6,  # default is 1 global 5 local
        segment_embed_mode: Literal["permute", "none", "fixed"] = "none",
        freeze_encoder: bool = False,
        apply_tanh: bool = True,
        # losses
        lambda_cos: float = 0.1,
        lambda_pose: float = 0.6,
        lambda_patch: float = 0.0,
        lambda_dino: float = 0.0,
        # pose loss
        criterion: str = "l1",
        alpha_t: float = 0.5,
        alpha_ts: float = 0.5,
        alpha_s: float = 0.75,
        # patch loss
        sigma: tuple[float, float, float, float] = (0.09, 0.09, 0.3, 0.3),
        # cosine alignment loss
        cos_eps: float = 1e-8,
        # ...
        debug: bool = False,
    ):
        """
        :param segment_embed_mode: 'permute' (random order), 'fixed' (same order), 'none'
        """
        super().__init__()
        self.debug = debug
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.canonical_img_size = canonical_img_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.pos_mask_ratio = pos_mask_ratio
        self.max_scale_ratio = max_scale_ratio

        self.patch_embed = OffGridPatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            mask_ratio=mask_ratio,
            sampler=SAMPLERS[sampler],
        )

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
        self.pose_head = PoseHead(
            embed_dim=decoder_embed_dim,
            num_targets=self.num_targets,
            apply_tanh=apply_tanh,
        )
        self.dino_head = DINOHead(
            in_dim=embed_dim,
            use_bn=False,
            nlayers=3,
            hidden_dim=2048,
            bottleneck_dim=256,
        )
        # default
        self.dino_loss = MCRLoss(
            ncrops=num_views,
            reduce_cov=0,
            expa_type=0,
            # eps: 0.05 is the one used for vit_b originally, but with batch size <= 128 it goes to NAN
            eps=0.5,
            coeff=1.0,
        )
        self.verbose = verbose

        # losses
        self._pose_loss = PoseLoss(
            criterion=criterion,
            alpha_t=alpha_t,
            alpha_s=alpha_s,
            alpha_ts=alpha_ts,
        )
        self._patch_loss = PatchLoss(sigma=sigma)
        self._cosine_loss = CosineAlignmentLoss(eps=cos_eps)
        self.lambda_pose = lambda_pose
        self.lambda_patch = lambda_patch
        self.lambda_cos = lambda_cos
        self.lambda_dino = lambda_dino

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
        nn.init.kaiming_uniform_(self.pose_head.linear.weight, a=4)

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

    def prepare_tokens(self, x: Tensor):
        # 1. (Off-grid) Patch (sub-)sampling and embedding
        x, patch_positions_vis = self.patch_embed(x)
        # 2. Mask Position Embeddings
        B_total, N_vis, D = x.shape
        ids_keep, _, ids_restore, ids_remove = self.random_masking(
            x, self.pos_mask_ratio
        )
        N_pos = ids_keep.shape[1]
        N_nopos = N_vis - N_pos
        patch_positions_pos = torch.gather(
            patch_positions_vis, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 2)
        )
        # 3. Apply Position Embeddings
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
        # 4. Class Token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        return x, patch_positions_vis, ids_remove

    def forward_encoder(self, x: Tensor) -> dict:
        r"""
        Encode an image crop.

        :param x: Tensor of shape [B*n_views, C, H, W]
        :return: Dict with:
                 "z_enc": Tensor of shape [B*n_views, 1+N_vis, D] (1 class token + N_vis visible tokens),
                 "patch_positions_vis": Tensor of shape [B*n_views, N_vis, 2],
                 "ids_remove_pos": Tensor of shape [B*n_views, M] (M indices of masked tokens)
        """
        x, patch_positions_vis, ids_remove = self.prepare_tokens(x)
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
        return x

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

    @torch.no_grad
    def _compute_gt(
        self,
        patch_pos_nopos: Int[Tensor, "B T 2"],
        params: Int[Tensor, "B T 4"],
        shapes: list,
    ) -> Tensor:
        r"""
        Compute ground-truth pairwise differences using per-token crop sizes.

        :param patch_pos_nopos: Tensor of shape [B, T, 2]
        :param params: Tensor of shape [B, T, 4]
        :param shapes: list of tuples (crop_size, M, V)
        :return: Tensor of shape [B, T, T, 4] (pairwise differences)
        """
        device = patch_pos_nopos.device
        B = patch_pos_nopos.shape[0]

        crop_sizes = torch.cat(
            [torch.full((M * V, 1), cs, device=device) for cs, M, V in shapes]
        )[None].expand(B, -1, -1)

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

    def _create_mask(self, shapes, device):
        # !! torch.compile-compliant:
        #   - Build labels list in Python to avoid data-dependent tensor creation.
        # resgroup_idx  is the resolution group index
        # in_resgroup_idx is the index of the view within that group
        resgroup_idx, in_resgroup_idx = torch.tensor(
            [
                (i, j)
                for i, (_, M, V) in enumerate(shapes)
                for j in range(V)  # each view in that group
                for _ in range(M)  # one entry per patch
            ],
            device=device,
        ).unbind(1)

        # Build intra- and inter-view masks
        # mask tells use which patches belong to the same view
        mask = resgroup_idx[:, None] == resgroup_idx[None, :]
        mask = mask & (in_resgroup_idx[:, None] == in_resgroup_idx[None, :])
        mask = mask.float()[None, ..., None]
        return mask

    def _get_N(self, crop_size: int) -> tuple[int, int]:
        N = (crop_size // self.patch_size) ** 2
        N_vis = int(N * (1 - self.mask_ratio))
        N_pos = int(N_vis * (1 - self.pos_mask_ratio))
        return N_vis, N_vis - N_pos

    def _encode_resgroup(self, x, params, indices, seg_embed, base_offset, N_vis):
        device = x[0].device
        group_x = torch.stack([x[i] for i in indices], dim=1)
        group_params = torch.stack([params[i] for i in indices], dim=1)
        enc = self.encode_views(group_x)
        enc["z_enc"] = enc["z_enc"] + seg_embed[indices].unsqueeze(0).unsqueeze(2)

        B = group_x.shape[0]
        cls_tokens = enc["z_enc"][:, :, 0, :]
        flat_patches = enc["z_enc"][:, :, 1:, :].flatten(1, 2)
        flat_positions = enc["patch_positions_vis"].flatten(1, 2)

        M = enc["ids_remove_pos"].shape[2]
        view_offsets = (torch.arange(len(indices), device=device) * N_vis).view(
            1, len(indices), 1
        )
        flat_ids_remove = (enc["ids_remove_pos"] + view_offsets + base_offset).view(
            B, -1
        )

        expanded_params = (
            group_params[:, :, 4:8].unsqueeze(2).expand(-1, -1, M, -1).flatten(1, 2)
        )

        shape = (group_x.shape[-2], M, len(indices))
        return (
            cls_tokens,
            flat_patches,
            flat_positions,
            flat_ids_remove,
            expanded_params,
            shape,
        )

    def forward(self, x: list[Tensor], params: list[Tensor]) -> dict:
        device = x[0].device
        B, V = x[0].shape[0], len(x)
        assert V == self.num_views

        # Group views by their resolution
        resolutions = [inp.shape[-1] for inp in x]
        groups = [
            list(group) for _, group in groupby(range(V), key=lambda i: resolutions[i])
        ]

        # Precompute per-view visible‐token counts and base offsets
        N_vis_list = [self._get_N(res)[0] for res in resolutions]
        view_base_offset = [0] + list(accumulate(N_vis_list[:-1]))

        # Apply segment embeddings
        seg_embed = self.segment_embed[:V]
        if self.segment_embed_mode == "permute":
            seg_embed = seg_embed[torch.randperm(V, device=device)]

        # Encode each group via our helper, then unpack
        results = [
            self._encode_resgroup(
                x, params, g, seg_embed, view_base_offset[g[0]], N_vis_list[g[0]]
            )
            for g in groups
        ]
        # Unpack grouped results and concatenate tensors
        *tensor_groups, shapes_list = zip(*results)
        joint_cls, joint_patches, joint_positions, joint_ids_remove, joint_params = [
            torch.cat(group, dim=1) for group in tensor_groups
        ]
        z = torch.cat([joint_cls, joint_patches], dim=1)

        # Decode, drop masked tokens
        z = self.forward_decoder(z)

        # pose head (without the cls tokens)
        pred_dT = self.pose_head(z[:, V:, :], joint_ids_remove)

        # compute loss
        patch_pos_nopos = _drop_pos(joint_positions, joint_ids_remove)
        gt_dT = self._compute_gt(patch_pos_nopos, joint_params, shapes_list)
        mask = self._create_mask(shapes_list, device).expand(B, -1, -1, 1)
        losses = self._pose_loss(pred_dT, gt_dT, mask)
        # ----------------
        dino_loss_keys = ["loss_dino", "loss_dino_comp", "loss_dino_expa"]
        if self.lambda_dino:
            # joint_cls: [B, V, D]
            # DINO Loss assumes [V, B, D] instead of [B, V, D]
            student_z = self.dino_head(joint_cls.permute(1, 0, 2))
            teacher_z = student_z[: len(groups[0])].detach()  # Just the global views
            dino_losses = self.dino_loss(student_z, teacher_z)
        else:
            dino_losses = [0.0] * len(dino_loss_keys)
        losses.update(dict(zip(dino_loss_keys, dino_losses)))

        # Applying L_patch only on posmasked tokens
        # TODO: replace these lines when on production.
        # For now, we log L_patch even if lambda_patch=0
        # patches_nopos = _drop_pos(joint_patches, joint_ids_remove)
        # losses["loss_patch"] = self._patch_loss(patches_nopos, gt_dT)
        losses["loss_patch"] = 0.0
        # L_cos = self._cosine_loss(pred_dT, gt_dT) if self.lambda_cos > 0 else 0.0
        losses["loss_cos"] = self._cosine_loss(pred_dT, gt_dT)

        loss = self.lambda_pose * losses["loss_pose"]
        loss = loss + self.lambda_patch * losses["loss_patch"]
        loss = loss + self.lambda_cos * losses["loss_cos"]
        loss = loss + self.lambda_dino * losses["loss_dino"]

        return {
            "patch_positions_nopos": patch_pos_nopos,
            "joint_ids_remove": joint_ids_remove,
            "shapes": shapes_list,
            "pred_dT": pred_dT,
            "gt_dT": gt_dT,
            "loss": loss,
            **losses,
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
    from src.data.components.transforms.multi_crop_v3 import ParametrizedMultiCropV3
    from PIL import Image
    import torch.utils._pytree as pytree
    from lightning import seed_everything

    seed_everything(42)
    gV, lV = 2, 3
    V = gV + lV

    backbone = (
        PART_mae_vit_base_patch16(
            pos_mask_ratio=0.9, mask_ratio=0.9, lambda_dino=0.1, num_views=V
        )
        .cuda()
        .eval()
    )
    t = ParametrizedMultiCropV3(n_global_crops=gV, n_local_crops=lV, distort_color=True)
    print(t.compute_max_scale_ratio_aug())  # <5.97

    class MockedDataset(torch.utils.data.Dataset):
        def __init__(self, transform=None, n=4):
            self.img = Image.open("artifacts/labrador.jpg")
            self.transform = transform
            self.n = n

        def __getitem__(self, idx):
            return self.transform(self.img)

        def __len__(self):
            return self.n

    with torch.no_grad():
        dataset = MockedDataset(t)
        seed_everything(42)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        seed_everything(42)
        batch = next(iter(loader))
        seed_everything(42)
        batch = pytree.tree_map_only(
            Tensor, lambda x: x.cuda(), batch
        )  # Move to GPU if available

        seed_everything(42)
        out = backbone(*batch)
        print("Output keys:", out.keys())
