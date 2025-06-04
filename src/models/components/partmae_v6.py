import torch
import math
from torch import nn, Tensor
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
    interpolate_grid_sample,
    get_sincos_pos_embed,
)
from torch.nn.modules.utils import _pair
from jaxtyping import Int, Float
from src.models.components.heads.dino_head import DINOHead
import logging
from src.models.components.loss.match import PatchMatchingLoss
from src.models.components.loss.pose import PoseLoss
from src.models.components.loss.patch_coding_rate import PatchCodingRateLoss, PatchCodingRateLossV2
from src.models.components.loss.simdino_ref import CLSCodingRateLoss, CLSInvarianceLoss
from src.models.components.loss.cos_align import CosineAlignmentLoss
from src.models.components.loss.stress import PatchStressLoss


logging.basicConfig(level=logging.INFO)


class PoseHead(nn.Module):
    def __init__(self, embed_dim: int, num_targets: int, apply_tanh: bool = False):
        super().__init__()
        self.linear = nn.Linear(embed_dim, num_targets, bias=False)
        self.tanh = nn.Tanh() if apply_tanh else nn.Identity()

    def forward(self, z):
        pose_pred = self.linear(z)
        # or equivalently:
        # w * (z_i -  z_j) = pose_ij
        pred_dT = self.tanh(pose_pred.unsqueeze(2) - pose_pred.unsqueeze(1))
        return pred_dT


def _drop_pos(
    x: Float[Tensor, "B N K"], ids: Int[Tensor, "B M"]
) -> Float[Tensor, "B M K"]:
    batch_idx = torch.arange(x.size(0), device=x.device)[:, None]  # → [B,1]
    return x[batch_idx, ids]  # → [B, M, K]


def drop_pos_2d(
    x: Float[Tensor, "B N N K"],
    ids: Int[Tensor, "B M"],
) -> Float[Tensor, "B M M K"]:
    batch_idx = torch.arange(x.size(0), device=x.device)[:, None, None]
    row_idx = ids[:, :, None]
    col_idx = ids[:, None, :]
    return x[batch_idx, row_idx, col_idx]  # → [B, M, M, K]


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
        local_img_size: int = 96,
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
        num_views: int = 12,  # default is 2 global 10 local
        freeze_encoder: bool = False,
        apply_tanh: bool = False,
        # losses
        lambda_pose: float = 0.6,
        # patch losses
        lambda_psmooth: float = 0.0,
        lambda_pstress: float = 0.0,  # stress loss
        lambda_pmatch: float = 0.0,  # patch matching loss
        lambda_pcr: float = 0.0,
        # class losses
        lambda_ccr: float = 0.0,  # match them (ccr = cinv)
        lambda_cinv: float = 0.0,
        # cosine alignment loss
        lambda_cos: float = 0.0,
        # pose loss
        criterion: str = "l1",
        alpha_t: float = 0.5,
        alpha_ts: float = 0.5,
        alpha_s: float = 0.75,
        # patch loss
        sigma_yx: float = 0.2,
        sigma_hw: float = 1.0,
        beta_f: float = 0.1,  # for patch matching loss
        beta_w: float = 3.0,  # for patch matching loss
        cr_eps: float = 0.5,
        # cosine alignment loss
        cos_eps: float = 1e-8,
        # dino loss
        proj_bottleneck_dim: int = 256,
        num_register_tokens: int = 0,
        ls_init_values: float = 0.0,  # 1e-5 for dinov2
        pos_embed_mode: str = "sincos",  # "sincos" or "learn"
        decoder_from_proj: bool = False,
        # ..
    ):
        """ """
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.local_img_size = local_img_size
        self.canonical_img_size = canonical_img_size
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.pos_mask_ratio = pos_mask_ratio
        self.max_scale_ratio = max_scale_ratio
        self.num_register_tokens = num_register_tokens
        # cls token + register tokens
        self.num_prefix_tokens = 1 + num_register_tokens
        self.pos_embed_mode = pos_embed_mode
        self.grid_size = _pair(self.img_size // self.patch_size)
        assert pos_embed_mode in ("sincos", "learn")

        self.patch_embed = OffGridPatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            mask_ratio=mask_ratio,
            sampler=SAMPLERS[sampler],
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            if num_register_tokens
            else None
        )

        # 2d grid only for interpolation, override in state_dict
        self.pos_embed = nn.Parameter(
            torch.zeros(embed_dim, *self.grid_size),
            requires_grad=pos_embed_mode == "learn",
        )
        self.cls_pos_embed = nn.Parameter(
            torch.zeros(1, 1, embed_dim), requires_grad=pos_embed_mode == "learn"
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
                    init_values=ls_init_values,
                )
                for _ in range(depth)
            ]
        )
        self.norm: nn.Module = norm_layer(embed_dim)
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_from_proj = decoder_from_proj
        dec_in_dim = proj_bottleneck_dim if decoder_from_proj else embed_dim
        self.decoder_embed = nn.Linear(dec_in_dim, decoder_embed_dim, bias=True)
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
        self.num_views = num_views
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
            bottleneck_dim=proj_bottleneck_dim,
        )
        self._cache_shapes()

        # SimDINO losses
        self._ccr_loss = CLSCodingRateLoss(
            embed_dim=proj_bottleneck_dim,
            eps=cr_eps,
            num_chunks=1,  # for stability
            gV=2,  # global views
        )
        self._cinv_loss = CLSInvarianceLoss()

        # losses
        self._pose_loss = PoseLoss(
            criterion=criterion,
            alpha_t=alpha_t,
            alpha_s=alpha_s,
            alpha_ts=alpha_ts,
        )
        # self._psmooth_loss = PatchSmoothnessLoss(sigma_yx=sigma_yx, sigma_hw=sigma_hw)
        self._pmatch_loss = PatchMatchingLoss(
            beta_f=beta_f,
            beta_w=beta_w,
            sigma_yx=sigma_yx,
            sigma_hw=sigma_hw,
            gN=self._gN,
        )
        self._pstress_loss = PatchStressLoss()
        NUM_CHUNKS = 2
        self._pcr_loss = PatchCodingRateLossV2(
            embed_dim=proj_bottleneck_dim,
            eps=cr_eps,
            pool_size=4,  # 4 * (2*49) = 392 patches > 256 (bottleneck_dim)
            num_chunks=NUM_CHUNKS,  # for stability
            gN=self._gN,
            gV=self._gV,
        )
        self._cosine_loss = CosineAlignmentLoss(eps=cos_eps)

        self.lambdas = {
            # "loss_psmooth": lambda_psmooth, # comp
            # "loss_pstress": lambda_pstress,
            "loss_pmatch": lambda_pmatch,  # comp
            "loss_pcr": lambda_pcr,  # exp
            "loss_cinv": lambda_cinv,  # comp
            "loss_ccr": lambda_ccr,  # exp
            "loss_cos": lambda_cos,
        }
        if lambda_pose > 0:
            self.lambdas["loss_pose"] = lambda_pose
        # It's important to ensure that expansion and compression losses are complementary
        # otherwise representational collapse can happen
        # assert lambda_psmooth == lambda_pcr
        assert not lambda_psmooth

        assert lambda_ccr == lambda_cinv

        self.initialize_weights()
        if freeze_encoder:
            self.freeze_encoder()

    def initialize_weights(self):
        r"""
        Initialize positional embeddings, cls token, mask token, and other weights.
        """
        if self.pos_embed_mode == "sincos":
            pe = get_sincos_pos_embed(
                self.embed_dim,
                self.grid_size,
                self.patch_size,
                device=self.pos_embed.device,
            )  # [1, N+1, D]
            self.cls_pos_embed.data = pe[:, :1, :]
            self.pos_embed.data = (
                pe[:, 1:, :].squeeze(0).permute(1, 0).unflatten(-1, self.grid_size)
            )
        else:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            nn.init.trunc_normal_(self.cls_pos_embed, std=0.02)
        nn.init.normal_(self.mask_pos_token, std=1e-6)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        self.apply(self._init_weights)
        # TODO: should i really initialize pose_head.linear?
        # nn.init.kaiming_uniform_(self.pose_head.linear.weight, a=4)

    def _init_weights(self, m):
        """
        ViT weight initialization, original timm impl (for reproducibility)
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def state_dict(self, *args, **kwargs):
        r"""
        1. Override state_dict to ensure pos_embed is compatible with timm's ViT
        2. reshape linear patch embeds to conv2d patchembeds
        """
        state_dict = super().state_dict(*args, **kwargs)
        state_dict["pos_embed"] = torch.cat(
            [
                state_dict.pop("cls_pos_embed"),
                state_dict.pop("pos_embed").flatten(-2, -1).permute(1, 0).unsqueeze(0),
            ],
            dim=1,
        )
        state_dict["patch_embed.proj.weight"] = state_dict[
            "patch_embed.proj.weight"
        ].reshape(self.embed_dim, self.in_chans, self.patch_size, self.patch_size)
        return state_dict

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Override load_state_dict to ensure timm's pos_embed is loaded to 2d grid"""
        if "pos_embed" in state_dict:
            pos_embed = state_dict.pop("pos_embed")
            # separate cls token and pos embed
            assert pos_embed.shape[1] == self.grid_size[0] * self.grid_size[1] + 1

            if self.pos_embed_mode == "sincos":
                logging.warning(
                    "Using fixed-sincos positional embeddings, ignoring checkpoints pos_embed."
                )
                pe = get_sincos_pos_embed(
                    self.embed_dim,
                    self.grid_size,
                    self.patch_size,
                    device=self.pos_embed.device,
                )  # [1, N+1, D]
                state_dict["cls_pos_embed"] = pe[:, :1, :]
                state_dict["pos_embed"] = (
                    pe[:, 1:, :].squeeze(0).permute(1, 0).unflatten(-1, self.grid_size)
                )
            else:  # learn
                state_dict["cls_pos_embed"] = pos_embed[:, :1, :]
                state_dict["pos_embed"] = (
                    pos_embed[:, 1:, :]
                    .squeeze(0)
                    .permute(1, 0)
                    .unflatten(1, self.grid_size)
                )
        if "patch_embed.proj.weight" in state_dict:
            state_dict["patch_embed.proj.weight"] = state_dict[
                "patch_embed.proj.weight"
            ].flatten(1, 3)
        return super().load_state_dict(state_dict, strict, assign)

    def _cache_shapes(self):
        (gN, gM), gV = self._get_N(self.img_size), 2
        (lN, lM), lV = self._get_N(self.local_img_size), self.num_views - 2
        Ns = [gN] * gV + [lN] * lV
        Ms = [gM] * gV + [lM] * lV
        N = gN * gV + lN * lV
        M = gM * gV + lM * lV
        Is = [self.img_size] * gV + [self.local_img_size] * lV
        keys = ["gN", "gM", "gV", "lN", "lM", "lV", "Ns", "Ms", "Is", "N", "M"]
        values = [gN, gM, gV, lN, lM, lV, Ns, Ms, Is, N, M]
        for k, v in zip(keys, values):
            # have cpu/int and cuda tensors
            self._register_or_overwrite_buffer(k, v)
            setattr(self, f"_{k}", v)

    def _register_or_overwrite_buffer(self, name: str, value: int | list[int]):
        if hasattr(self, name):
            old = getattr(self, name)
            new = torch.tensor(value, dtype=torch.int32, device=old.device)
            setattr(self, name, new)
        else:
            self.register_buffer(
                name, torch.tensor(value, dtype=torch.int32), persistent=False
            )

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

        if mask_ratio is not None or pos_mask_ratio is not None:
            self._cache_shapes()

    def _validate_shapes(self, x: list[Float[Tensor, "B gV|lV C H W"]]):
        if len(x) not in (1, 2):
            raise ValueError(f"Expected 1 or 2 views, got {len(x)}")

        gV_actual = x[0].shape[1]
        lV_actual = x[1].shape[1] if len(x) == 2 else 0

        if self.training:
            if gV_actual != self._gV or lV_actual != self._lV:
                raise ValueError(
                    f"Expected {self._gV} global views and {self._lV} local views, "
                    f"but got {gV_actual} global views and {lV_actual} local views."
                )
        else:
            if gV_actual != self._gV or lV_actual != self._lV:
                self._cache_shapes()

    def prepare_tokens(self, x: Tensor):
        # 1. (Off-grid) Patch (sub-)sampling and embedding
        x, patch_positions_vis = self.patch_embed(x)
        # 2. Mask Position Embeddings
        B_total, N_vis, D = x.shape
        ids_keep, ids_restore, ids_remove = self.random_masking(x, self.pos_mask_ratio)
        N_pos = ids_keep.shape[1]
        N_nopos = N_vis - N_pos
        patch_positions_pos = torch.gather(
            patch_positions_vis, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 2)
        )
        # 3. Apply Position Embeddings
        if self.pos_embed_mode == "sincos":
            pos_embed = get_2d_sincos_pos_embed(
                patch_positions_pos.flatten(0, 1) / self.patch_size, self.embed_dim
            )
        else:
            pos_embed = interpolate_grid_sample(
                self.pos_embed, patch_positions_pos.flatten(0, 1) / self.patch_size
            )

        pos_embed = pos_embed.unflatten(0, (B_total, N_pos))
        mask_pos_tokens = self.mask_pos_token.expand(B_total, N_nopos, -1)
        pos_embed = torch.cat([pos_embed, mask_pos_tokens], dim=1)
        pos_embed = torch.gather(
            pos_embed, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D)
        )
        x = x + pos_embed
        # 4. Class Token
        to_cat = [
            (self.cls_token + self.cls_pos_embed).expand(x.shape[0], -1, -1),
        ]

        # 5. Register Tokens
        if self.register_tokens is not None:
            to_cat.append(self.register_tokens.expand(x.shape[0], -1, -1))

        x = torch.cat(to_cat + [x], dim=1)
        return x, patch_positions_vis, ids_remove

    def forward_encoder(self, x: Tensor) -> dict:
        r"""
        Encode an image crop.

        :param x: Tensor of shape [B*n_views, C, H, W]
        :return: tuple with
                 "z_enc": Tensor of shape [B*n_views, 1+N_vis, D] (1 class token + N_vis visible tokens),
                 "patch_positions_vis": Tensor of shape [B*n_views, N_vis, 2],
                 "ids_remove_pos": Tensor of shape [B*n_views, M] (M indices of masked tokens)
        """
        x, patch_positions_vis, ids_remove_pos = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, patch_positions_vis, ids_remove_pos

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
        return ids_keep, ids_restore, ids_remove

    def _compute_gt(
        self,
        patch_pos_nopos: Int[Tensor, "B T 2"],
        params: Int[Tensor, "B V 4"],
        token_sizes: Int[Tensor, "V"],
    ) -> Tensor:
        r"""
        Compute ground-truth pairwise differences using per-token crop sizes.

        :param patch_pos_nopos: Tensor of shape [B, T, 2]
        :param params: Tensor of shape [B, T, 4]
        :param shapes: list of tuples (crop_size, M, V)
        :return: Tensor of shape [B, T, T, 4] (pairwise differences)
        """
        N = patch_pos_nopos.shape[1]
        params = params.repeat_interleave(token_sizes, dim=1, output_size=N)
        crop_sizes = self.Is.float().repeat_interleave(token_sizes, output_size=N)[
            None, :, None
        ]

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
        gt_dT[..., :2] /= self.canonical_img_size
        gt_dT[..., 2:] /= math.log(self.max_scale_ratio)

        return gt_dT

    def _get_N(self, crop_size: int) -> tuple[int, int]:
        N = (crop_size // self.patch_size) ** 2
        N_vis = int(N * (1 - self.mask_ratio))
        N_pos = int(N_vis * (1 - self.pos_mask_ratio))
        return N_vis, N_vis - N_pos

    def _encode_resgroup(
        self,
        x: Float[Tensor, "B gV|lV C H W"],
    ):
        B, V, _, _, _ = x.shape
        z, patch_positions_vis, ids_remove_pos = self.forward_encoder(x.flatten(0, 1))
        N_vis, N_nopos = patch_positions_vis.shape[-2], ids_remove_pos.shape[-1]

        return (
            # cls token # [B, V, D]
            z[:, 0].reshape(B, V, -1),
            # drop the register tokens if any
            # patches # [B, V * N_vis, D]
            z[:, self.num_prefix_tokens :].reshape(B, V * N_vis, -1),
            patch_positions_vis.reshape(B, V * N_vis, -1),  # positions #
            ids_remove_pos.reshape(B, V * N_nopos),  # ids_remove
        )

    def forward_pose_loss(self, z, gt_dT, joint_ids_remove):
        if not self.lambdas.get("loss_pose", 0):
            return {}
        V = self.num_views
        z = self.forward_decoder(z)
        z_nopos = _drop_pos(z[:, V:, :], joint_ids_remove)
        pred_dT_nopos = self.pose_head(z_nopos)
        gt_dT_nopos = drop_pos_2d(gt_dT, joint_ids_remove)  # check
        return {
            **self._pose_loss(pred_dT_nopos, gt_dT_nopos, self._Ms),
            "pred_dT": pred_dT_nopos,
            "gt_dT": gt_dT_nopos,
        }

    def forward(
        self,
        x: list[Float[Tensor, "B gV|lV C gS|lS gW|lW"]],
        params: list[Float[Tensor, "B gV|lV 19"]],
    ) -> dict:
        assert 1 <= len(x) <= 2, "Global or global + local views are supported"
        self._validate_shapes(x)
        assert len(x) == len(params), "x and params must have the same length"
        params = torch.cat([p[:, :, 4:8] for p in params], dim=1)
        V = self.num_views

        ## ENCODING
        joint_cls, joint_patches, joint_pos, joint_ids_remove = [
            torch.cat(group, dim=1)
            for group in zip(*[self._encode_resgroup(_x) for _x in x])
        ]
        z = torch.cat([joint_cls, joint_patches], dim=1)
        del joint_cls, joint_patches, x
        # joint_ids_remove need to be offset:
        # (index 0 of second view is 0 + number of patches in first view)
        M = joint_ids_remove.shape[1]
        offset = (self.Ns.cumsum(0) - self.Ns).repeat_interleave(self.Ms, output_size=M)
        joint_ids_remove = joint_ids_remove + offset  # correct :)

        ## DECODE

        # joint_cls: [B, global_views + local_views, D]
        # joint_patches: [B, global_views * vis_global_tokens + local_views * vis_local_tokens, D]
        # PROJECTOR
        proj_z = self.dino_head(z)
        proj_cls = proj_z[:, :V]
        proj_patches = proj_z[:, V:]
        gt_dT = self._compute_gt(joint_pos, params, self.Ns)
        patch_pos_nopos = _drop_pos(joint_pos, joint_ids_remove)
        # TRANSFORMER DECODER
        dec_input = proj_z if self.decoder_from_proj else z
        losses = self.forward_pose_loss(dec_input, gt_dT, joint_ids_remove)
        # joint_cls: [B, V, D]
        # teacher is just the cls of the global views
        losses["loss_cinv"] = self._cinv_loss(
            proj_cls, proj_cls[:, : self._gV].detach()
        )
        losses["loss_ccr"] = self._ccr_loss(proj_cls)
        # losses["loss_psmooth"] = self._psmooth_loss(proj_patches, gt_dT)
        # losses["loss_pstress"] = self._pstress_loss(proj_patches, gt_dT)
        losses.update(self._pmatch_loss(proj_patches, gt_dT))
        losses["loss_pcr"] = self._pcr_loss(proj_patches)
        # losses["loss_cos"] = self._cosine_loss(pred_dT_nopos, gt_dT_nopos)

        loss = torch.stack(
            [self.lambdas[k] * losses[k] for k in self.lambdas if k in losses]
        ).sum()
        return {
            "patch_positions_nopos": patch_pos_nopos,
            "joint_ids_remove": joint_ids_remove,
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
    from src.data.components.transforms.multi_crop_v4 import ParametrizedMultiCropV4
    from PIL import Image
    import torch.utils._pytree as pytree
    from lightning import seed_everything

    seed_everything(42)
    gV, lV = 2, 10
    V = gV + lV

    backbone = (
        PART_mae_vit_base_patch16(
            pos_mask_ratio=0.75, mask_ratio=0.75, num_views=V, pos_embed_mode="sincos"
        )
        .cuda()
        .eval()
    )
    t = ParametrizedMultiCropV4(n_global_crops=gV, n_local_crops=lV, distort_color=True)
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

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        dataset = MockedDataset(t)
        seed_everything(42)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
        seed_everything(42)
        batch = next(iter(loader))
        seed_everything(42)
        batch = pytree.tree_map_only(
            Tensor, lambda x: x.cuda(), batch
        )  # Move to GPU if available

        seed_everything(42)
        out = backbone(*batch)
        print("Output keys:", out.keys())

    # print(out["loss_pose"])
    sd = backbone.state_dict()
    backbone.load_state_dict(sd, strict=True)
    print(backbone.blocks[0].attn.fused_attn)
