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
from src.models.components.heads.dino_head import DINOHead
import torch.nn.functional as F


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

    def forward(self, pred_dT: Tensor, gt_dT: Tensor, Ms: Tensor) -> Tensor:
        device, B, V = pred_dT.device, pred_dT.shape[0], Ms.shape[0]

        loss_full = self.criterion(pred_dT, gt_dT, reduction="none")

        view_ids = torch.arange(V, device=device).repeat_interleave(Ms)
        mask = (view_ids[None, :] == view_ids[:, None]).float()
        mask = mask[None, ..., None].expand(B, -1, -1, 1)
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


class PatchSmoothnessLoss(nn.Module):
    """
    Graph-Laplacian smoothness

    params:
        sigma_yx: bandwidth for spatial offsets (dy, dx)
        sigma_hw: bandwidth for log‐scale offsets (dlogh, dlogw)
    """

    def __init__(
        self,
        sigma_yx: float = 0.09,
        sigma_hw: float = 0.30,
    ):
        super().__init__()
        # sigma² per‐dimension: [dy, dx, dlogh, dlogw]
        self.register_buffer(
            "sigma2",
            torch.tensor(
                [sigma_yx**2, sigma_yx**2, sigma_hw**2, sigma_hw**2],
                dtype=torch.float32,
            ),
            persistent=False,
        )

    @staticmethod
    def _laplacian_energy(z: Tensor, w: Tensor) -> Tensor:
        """
        Compute batch-wise tr(Zᵀ L Z) where L = D - W.
        z: (B, T, D)
        w: (B, T, T) positive affinity matrix
        returns: (B,)
        """
        # degree d_i = ∑_j w_ij
        d = w.sum(-1)  # (B, T)
        # z² norms
        z2 = (z * z).sum(-1)  # (B, T)
        # ∑_j w_ij z_j
        wz = torch.bmm(w, z)  # (B, T, D)

        # tr(Zᵀ D Z) − 2 tr(Zᵀ W Z) + tr(Zᵀ W Z)
        # = ∑_i d_i ‖z_i‖² − 2 ∑_i z_i·(∑_j w_ij z_j) + ∑_j d_j ‖z_j‖²
        term1 = (z2 * d).sum(-1)
        term2 = 2 * (wz * z).sum((-1, -2))
        term3 = (z2.unsqueeze(1) * w).sum((-1, -2))
        return term1 - term2 + term3

    def forward(
        self, z: Float[Tensor, "B T D"], gt_dT: Float[Tensor, "B T T 4"]
    ) -> Float[Tensor, "1"]:
        """
        :param z:    L2-normalized patch embeddings
        :param gt_dT: normalized [dy, dx, dlogh, dlogw]
        """

        # --- 1) Gaussian kernel w_ij ---
        w = (gt_dT.pow(2) / (2 * self.sigma2)).sum(-1)  # (B, T, T)
        w = torch.exp(-w)  # (B, T, T)

        # --- 2) Laplacian smoothness term ---
        num = self._laplacian_energy(z, w)  # (B,)
        denom = w.sum((-1, -2)).clamp(min=1e-6)  # (B,)
        lap = num / denom  # (B,)
        return lap.mean()


class PatchCodingRateLoss(nn.Module):
    def __init__(self, embed_dim: int, eps: float = 0.5):
        super().__init__()
        self.eps = eps
        self.register_buffer(
            "I", torch.eye(embed_dim, dtype=torch.float32), persistent=False
        )

    def forward(self, z: Float[Tensor, "B T D"]) -> Float[Tensor, "1"]:
        """
        :param z: l2 normalized patch embeddings
        """
        _, T, D = z.shape

        # 1) batched covariance: [B, D, T] @ [B, D, T] → [B, D, D]
        cov = torch.bmm(z.transpose(1, 2), z)

        # 2) form I + α·cov  (broadcasting I over the batch dim)
        alpha = D / (T * self.eps)
        cov = alpha * cov + self.I  # shape [B, D, D]

        # 3) batch log determinant of I + α·cov
        expa = (
            torch.linalg.cholesky_ex(cov)[0]
            .diagonal(dim1=-2, dim2=-1)
            .log()
            .sum(dim=1)
            .mean()
        )

        # 4) final γ‐scaling
        gamma = (D + T) / (D * T)
        return expa * gamma


class CLSCodingRateLoss(nn.Module):
    # expansion loss for MCR
    def __init__(self, embed_dim: int, eps: float, gV: int):
        super().__init__()
        self.eps = eps
        self.gV = gV  # number of global views
        self.register_buffer(
            "I", torch.eye(embed_dim, dtype=torch.float32), persistent=False
        )

    def forward(self, z: Float[Tensor, "V B D"]) -> Tensor:
        """
        :param z: l2 normalized patch embeddings (student)
        """
        V, B, D = z.shape
        z = z[: self.gV]  # only use global views for CLS coding rate loss
        cov = torch.bmm(z.transpose(1, 2), z)  # [V, D, D]
        scalar = D / (B * self.eps)
        loss = (
            torch.linalg.cholesky_ex(self.I + scalar * cov)[0]
            .diagonal(dim1=-2, dim2=-1)
            .log()
            .sum(dim=1)
            .mean()
        )
        loss *= (D + B) / (D * B)
        # the balancing factor gamma, you can also use the next line. This is ultimately a heuristic, so feel free to experiment.
        # loss *= ((self.eps *  B) ** 0.5 / D)
        return loss


class CLSInvarianceLoss(nn.Module):
    # compression loss for MCR
    def __init__(self):
        super().__init__()

    def forward(
        self, z_stu: Float[Tensor, "V B D"], z_tea: Float[Tensor, "V B D"]
    ) -> Tensor:
        sim = F.cosine_similarity(z_tea.unsqueeze(1), z_stu.unsqueeze(0), dim=-1)
        # Trick to fill diagonal
        sim.view(-1, sim.shape[-1])[:: (len(z_stu) + 1), :].fill_(0)
        n_loss_terms = len(z_tea) * len(z_stu) - min(len(z_tea), len(z_stu))
        # Sum the cosine similarities
        comp_loss = sim.mean(2).sum() / n_loss_terms
        return -comp_loss  # negative because we want to maximize similarity


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

    def forward(self, z):
        pose_pred = self.linear(z)
        pose_u = pose_pred.unsqueeze(2)
        pose_v = pose_pred.unsqueeze(1)
        pred_dT = self.tanh(pose_u - pose_v)
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
        apply_tanh: bool = True,
        # losses
        lambda_pose: float = 0.6,
        # patch losses
        lambda_psmooth: float = 0.0,
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
        sigma_yx: float = 0.09,
        sigma_hw: float = 0.30,
        cr_eps: float = 0.5,
        # cosine alignment loss
        cos_eps: float = 1e-8,
        # dino loss
        proj_bottleneck_dim: int = 256,
        # ..
    ):
        """ """
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.local_img_size = local_img_size
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
            bottleneck_dim=proj_bottleneck_dim,
        )

        # SimDINO losses
        self._ccr_loss = CLSCodingRateLoss(
            embed_dim=proj_bottleneck_dim,
            eps=cr_eps,
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
        self._psmooth_loss = PatchSmoothnessLoss(sigma_yx=sigma_yx, sigma_hw=sigma_hw)
        self._pcr_loss = PatchCodingRateLoss(embed_dim=proj_bottleneck_dim, eps=cr_eps)
        self._cosine_loss = CosineAlignmentLoss(eps=cos_eps)

        self.lambdas = {
            "loss_pose": lambda_pose,
            "loss_psmooth": lambda_psmooth,
            "loss_pcr": lambda_pcr,
            "loss_ccr": lambda_ccr,
            "loss_cinv": lambda_cinv,
            "loss_cos": lambda_cos,
        }

        self.num_views = num_views
        self.initialize_weights()
        if freeze_encoder:
            self.freeze_encoder()

        self._cache_shapes()

    def _cache_shapes(self):
        (gN, gM), gV = self._get_N(self.img_size), 2
        (lN, lM), lV = self._get_N(self.local_img_size), self.num_views - 2
        Ns = [gN] * gV + [lN] * lV
        Ms = [gM] * gV + [lM] * lV
        Is = [self.img_size] * gV + [self.local_img_size] * lV
        keys = ["gN", "gM", "gV", "lN", "lM", "lV", "Ns", "Ms", "Is"]
        values = [gN, gM, gV, lN, lM, lV, Ns, Ms, Is]
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
        params = params.repeat_interleave(token_sizes, dim=1)
        crop_sizes = self.Is.float().repeat_interleave(token_sizes)[None, :, None]

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
            z[:, 0].reshape(B, V, -1),  # cls token # [B, V, D]
            z[:, 1:].reshape(B, V * N_vis, -1),  # patches # [B, V * N_vis, D]
            patch_positions_vis.reshape(B, V * N_vis, -1),  # positions #
            ids_remove_pos.reshape(B, V * N_nopos),  # ids_remove
        )

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
        results = [self._encode_resgroup(_x) for _x in x]
        joint_cls, joint_patches, joint_pos, joint_ids_remove = [
            torch.cat(group, dim=1) for group in zip(*results)
        ]
        # joint_ids_remove need to be offset:
        # (index 0 of second view is 0 + number of patches in first view)
        offset = (self.Ns.cumsum(0) - self.Ns).repeat_interleave(self.Ms)
        joint_ids_remove = joint_ids_remove + offset
        z = torch.cat([joint_cls, joint_patches], dim=1)

        ## DECODE

        # joint_cls: [B, global_views + local_views, D]
        # joint_patches: [B, global_views * vis_global_tokens + local_views * vis_local_tokens, D]
        # PROJECTOR
        proj_z = self.dino_head(z)
        # TRANSFORMER DECODER
        z = self.forward_decoder(z)
        # POSE HEAD
        z_nopos = _drop_pos(z[:, V:, :], joint_ids_remove)
        pred_dT_nopos = self.pose_head(z_nopos)
        ## LOSSES

        # joint_cls: [B, V, D]
        # SimDINO Loss assumes [V, B, D] instead of [B, V, D]
        # teacher is just the cls of the global views
        proj_cls = proj_z[:, :V].permute(1, 0, 2)
        proj_patches = proj_z[:, V:]
        gt_dT = self._compute_gt(joint_pos, params, self.Ns)
        gt_dT_nopos = drop_pos_2d(gt_dT, joint_ids_remove)
        patch_pos_nopos = _drop_pos(joint_pos, joint_ids_remove)

        losses = self._pose_loss(pred_dT_nopos, gt_dT_nopos, self.Ms)
        losses["loss_cinv"] = self._cinv_loss(proj_cls, proj_cls[: self._gV].detach())
        losses["loss_ccr"] = self._ccr_loss(proj_cls)
        losses["loss_psmooth"] = self._psmooth_loss(proj_patches, gt_dT)
        losses["loss_pcr"] = self._pcr_loss(proj_patches)
        losses["loss_cos"] = self._cosine_loss(pred_dT_nopos, gt_dT_nopos)

        loss = sum(self.lambdas[k] * losses[k] for k in self.lambdas)
        return {
            "patch_positions_nopos": patch_pos_nopos,
            "joint_ids_remove": joint_ids_remove,
            "pred_dT": pred_dT_nopos,
            "gt_dT": gt_dT_nopos,
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
    gV, lV = 2, 3
    V = gV + lV

    backbone = (
        PART_mae_vit_base_patch16(pos_mask_ratio=0.9, mask_ratio=0.9, num_views=V)
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

    with torch.no_grad():
        dataset = MockedDataset(t)
        seed_everything(42)
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
        seed_everything(42)
        batch = next(iter(loader))
        seed_everything(42)
        batch = pytree.tree_map_only(
            Tensor, lambda x: x.cuda(), batch
        )  # Move to GPU if available

        seed_everything(42)
        out = backbone(*batch)
        print("Output keys:", out.keys())
