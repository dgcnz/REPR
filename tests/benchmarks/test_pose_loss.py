import torch
import itertools
from torch import nn, Tensor
from jaxtyping import Float, Int
from torch.nn.functional import mse_loss, l1_loss
import torch.nn.functional as F
import pytest
from torch.masked import masked_tensor, as_masked_tensor


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

    def forward(
        self,
        pred_dT: Float[Tensor, "B M M 4"],
        gt_dT: Float[Tensor, "B M M 4"],
        view_ids: Int[Tensor, ' M'],  # [M] view_ids[i] = which view i belongs to
    ) -> Tensor:
        (B, M) = pred_dT.shape[:2]

        loss_full = self.criterion(pred_dT, gt_dT, reduction="none")

        mask = (view_ids[None, :] == view_ids[:, None])
        diag = mask.sum() * B
        offdiag = B * M * M - diag
        mask = mask[None, ..., None].expand(B, -1, -1, 1)

        loss_intra  = (loss_full[..., :4] * mask).sum((0, 1, 2)) / diag
        loss_inter = loss_full.sum((0, 1, 2)) - loss_intra
        # loss_inter = (loss_full[..., :4] * ~mask).sum((0, 1, 2)) / offdiag
        loss_intra_t, loss_intra_s = loss_intra[:2].sum(), loss_intra[2:].sum()
        loss_inter_t, loss_inter_s = loss_inter[:2].sum(), loss_inter[2:].sum()
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


class OptimizedPoseLossV3(nn.Module):
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

    def forward(
        self,
        pred_dT: Float[Tensor, "B M M 4"],
        gt_dT: Float[Tensor, "B M M 4"],
        Ms: "Int[Tensor, ' V']",  # sum(Ms) == M
    ) -> Tensor:
        device, (B, M), V = pred_dT.device, pred_dT.shape[:2], Ms.shape[0]

        view_ids = torch.arange(V, device=device).repeat_interleave(Ms, output_size=M)
        sum_Ms_sq = (Ms * Ms).sum()
        diag_count_val = sum_Ms_sq * B
        offdiag_count_val = (M * M - sum_Ms_sq) * B

        loss_full = self.criterion(pred_dT, gt_dT, reduction="none")  # [B, M, M, 4]
        ids = 2 * (view_ids[None, :] != view_ids[:, None]).int().unsqueeze(-1).expand(
            -1, -1, 4
        )
        ids = ids + torch.tensor([0, 0, 1, 1], device=device, dtype=torch.int64)
        ids = ids[None].expand(B, -1, -1, -1)  # [B, M, M, 4]
        ids = F.one_hot(ids, num_classes=4).float()  # [B, M, M, 4, 4]
        ids = ids.flatten(0, -2).transpose(0, 1)

        loss_intra_t, loss_intra_s, loss_inter_t, loss_inter_s = (
            ids @ loss_full.flatten()
        )  # [4]
        del loss_full
        loss_intra_t = loss_intra_t / diag_count_val
        loss_intra_s = loss_intra_s / diag_count_val
        loss_inter_t = loss_inter_t / offdiag_count_val
        loss_inter_s = loss_inter_s / offdiag_count_val

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


class OptimizedPoseLossV1(nn.Module):
    def __init__(
        self,
        criterion: str = "mse",
        alpha_t: float = 0.5,
        alpha_s: float = 0.75,
        alpha_ts: float = 0.5,
    ):
        super().__init__()
        self.criterion_fn = F.mse_loss if criterion == "mse" else F.l1_loss
        self.alpha_t = alpha_t
        self.alpha_s = alpha_s
        self.alpha_ts = alpha_ts

    def forward(
        self,
        pred_dT: Float[Tensor, "B M M 4"],
        gt_dT: Float[Tensor, "B M M 4"],
        Ms: Int[Tensor, " V"],  # sum(Ms) == M
    ) -> dict[str, Tensor]:
        device = pred_dT.device
        B, M = pred_dT.shape[0], pred_dT.shape[1]
        V = Ms.shape[0]

        # 1) Compute per‐pair, 4‐channel loss (no reduction)
        loss_full = self.criterion_fn(pred_dT, gt_dT, reduction="none")  # [B, M, M, 4]

        # 2) Compute diag_count and offdiag_count from Ms
        sum_Ms_sq = (Ms * Ms).sum()
        diag_count_val = sum_Ms_sq * B
        offdiag_count_val = (M * M - sum_Ms_sq) * B

        # 3) Sum over all (b,i,j) to get total_all_loss_per_channel: [4]
        total_all_loss_per_channel = loss_full.sum(
            dim=(0, 1, 2)
        )  # Sum over B, M_rows, M_cols
        total_all_t = total_all_loss_per_channel[0:2].sum()
        total_all_s = total_all_loss_per_channel[2:4].sum()

        # 4) Build view_ids = length M, indicating which view each index i∈[0..M-1] belongs to
        view_ids = torch.arange(V, device=device, dtype=torch.long).repeat_interleave(
            Ms, output_size=M
        )

        # 5) Use index_add to get group_sum_loss_full[b, v, j, c] = ∑_{i: view_ids[i]==v} loss_full[b,i,j,c]
        #    Source is loss_full [B,M,M,4]. Index is view_ids [M].
        #    group_sum_loss_full is [B,V,M,4]. index_add_ is along dim 1 (V-dim).
        group_sum_loss_full = loss_full.new_zeros((B, V, M, 4))
        group_sum_loss_full.index_add_(1, view_ids, loss_full)

        # 6) Gather only those terms where source row i and column j are in the same view:
        view_ids_expand_B_M = view_ids.unsqueeze(0).expand(B, M)  # [B,M]
        idx_gather = (
            view_ids_expand_B_M.unsqueeze(1).unsqueeze(-1).expand(B, 1, M, 4)
        )  # [B,1,M,4]
        diag_vals_loss_full = group_sum_loss_full.gather(1, idx_gather).squeeze(
            1
        )  # [B,M,4]

        # 7) Sum diag_vals_loss_full to get total_intra_loss_per_channel: [4]
        total_intra_loss_per_channel = diag_vals_loss_full.sum(
            dim=(0, 1)
        )  # Sum over B, M_cols
        total_intra_t = total_intra_loss_per_channel[0:2].sum()
        total_intra_s = total_intra_loss_per_channel[2:4].sum()

        # 8) Compute inter‐view = all − intra
        total_inter_t = total_all_t - total_intra_t
        total_inter_s = total_all_s - total_intra_s

        # 9) Normalize by diag_count_val and offdiag_count_val
        loss_intra_t = total_intra_t / diag_count_val
        loss_inter_t = total_inter_t / offdiag_count_val
        loss_intra_s = total_intra_s / diag_count_val
        loss_inter_s = total_inter_s / offdiag_count_val

        # 10) Blend with α‐weights
        loss_t = self.alpha_t * loss_inter_t + (1.0 - self.alpha_t) * loss_intra_t
        loss_s = self.alpha_s * loss_inter_s + (1.0 - self.alpha_s) * loss_intra_s
        loss = self.alpha_ts * loss_t + (1.0 - self.alpha_ts) * loss_s

        return {
            "loss_pose_intra_t": loss_intra_t,
            "loss_pose_inter_t": loss_inter_t,
            "loss_pose_intra_s": loss_intra_s,
            "loss_pose_inter_s": loss_inter_s,
            "loss_pose_t": loss_t,
            "loss_pose_s": loss_s,
            "loss_pose": loss,
        }


class OptimizedPoseLossV7(nn.Module):
    def __init__(
        self,
        criterion: str = "mse",
        alpha_t: float = 0.5,
        alpha_s: float = 0.75,
        alpha_ts: float = 0.5,
    ):
        super().__init__()
        self.criterion = F.mse_loss if criterion == "mse" else F.l1_loss
        self.alpha_t = alpha_t
        self.alpha_s = alpha_s
        self.alpha_ts = alpha_ts

    def forward(
        self,
        pred_dT: Float[Tensor, "B M M 4"],
        gt_dT: Float[Tensor, "B M M 4"],
        Ms: Int[Tensor, " V"],  # sum(Ms) == M, and each “view” is a contiguous block
    ) -> dict[str, Tensor]:
        device = pred_dT.device
        B, M = pred_dT.shape[0], pred_dT.shape[1]
        V = Ms.shape[0]

        # ─────────────── 1) Early collapse 4→2 channels ───────────────
        # This is identical to V6’s original “.view(B,M,M,2,2).sum(-1)”
        # result: loss_full[b,i,j,0] = translation‐error, loss_full[b,i,j,1] = scale‐error
        loss_full = (
            self.criterion(pred_dT, gt_dT, reduction="none").view(B, M, M, 2, 2).sum(-1)
        )  # → shape [B, M, M, 2]

        # ─────────────── 2) Build a single [M×M] boolean mask “i and j in same view” ───────────────
        # First create `view_ids[i]` = which view the i‐th row belongs to:
        view_ids = torch.arange(V, device=device, dtype=torch.long).repeat_interleave(
            Ms, output_size=M
        )
        # Now “mask[i,j] = True iff view_ids[i] == view_ids[j]”:
        mask = view_ids.unsqueeze(1) == view_ids.unsqueeze(0)  # → [M, M] bool
        mask_f = mask.to(loss_full.dtype)  # [M, M] float(0/1)

        # ─────────────── 3) Compute “total_intra” via a single einsum ───────────────
        # This does: ∑_{b=0..B-1} ∑_{i=0..M-1} ∑_{j=0..M-1} mask_f[i,j] * loss_full[b,i,j,c],
        # leaving only channel c.  → shape [2]
        total_intra_per_channel = torch.einsum("mn,bmnc->c", mask_f, loss_full)
        total_intra_t = total_intra_per_channel[0]
        total_intra_s = total_intra_per_channel[1]

        # ─────────────── 4) Compute “total_all” by summing every (b,i,j) ───────────────
        # This is equivalent to ∑_{b,i,j} loss_full[b,i,j,c] → shape [2]
        total_all_per_channel = loss_full.sum(dim=(0, 1, 2))  # → [2]
        total_all_t = total_all_per_channel[0]
        total_all_s = total_all_per_channel[1]

        # We no longer need loss_full after getting total_intra & total_all:
        del loss_full

        # ─────────────── 5) Build normalization constants ───────────────
        # Same as before in V6:
        sum_Ms_sq = (Ms * Ms).sum()  # ∑ m_v²
        diag_count_val = sum_Ms_sq * B
        offdiag_count_val = (M * M - sum_Ms_sq) * B

        # “inter” = all − intra
        total_inter_t = total_all_t - total_intra_t
        total_inter_s = total_all_s - total_intra_s

        # normalize
        loss_intra_t = total_intra_t / diag_count_val
        loss_inter_t = total_inter_t / offdiag_count_val
        loss_intra_s = total_intra_s / diag_count_val
        loss_inter_s = total_inter_s / offdiag_count_val

        # α‐blend exactly as before
        loss_t = self.alpha_t * loss_inter_t + (1.0 - self.alpha_t) * loss_intra_t
        loss_s = self.alpha_s * loss_inter_s + (1.0 - self.alpha_s) * loss_intra_s
        loss = self.alpha_ts * loss_t + (1.0 - self.alpha_ts) * loss_s

        return {
            "loss_pose_intra_t": loss_intra_t,
            "loss_pose_inter_t": loss_inter_t,
            "loss_pose_intra_s": loss_intra_s,
            "loss_pose_inter_s": loss_inter_s,
            "loss_pose_t": loss_t,
            "loss_pose_s": loss_s,
            "loss_pose": loss,
        }


class OptimizedPoseLossV6(nn.Module):
    def __init__(
        self,
        criterion: str = "mse",
        alpha_t: float = 0.5,
        alpha_s: float = 0.75,
        alpha_ts: float = 0.5,
    ):
        super().__init__()
        self.criterion = F.mse_loss if criterion == "mse" else F.l1_loss
        self.alpha_t = alpha_t
        self.alpha_s = alpha_s
        self.alpha_ts = alpha_ts

    def forward(
        self,
        pred_dT: Float[Tensor, "B M M 4"],
        gt_dT: Float[Tensor, "B M M 4"],
        Ms: Int[Tensor, " V"],  # sum(Ms) == M
    ) -> Tensor:
        device, (B, M), V = pred_dT.device, pred_dT.shape[:2], Ms.shape[0]

        loss_full = (
            self.criterion(pred_dT, gt_dT, reduction="none")
            # .view(B, M, M, 2, 2).sum(-1)
        )

        blocks = loss_full.split(tuple(Ms.tolist()), dim=1)
        #    → blocks is a length-V tuple, where blocks[v].shape == [B, m_v, M, 2]
        del loss_full

        # 2) Sum each block along its “row” dimension (dim=1):
        group_slices = [blk.sum(dim=1, keepdim=True) for blk in blocks]
        #    → each blk.sum is [B, 1, M, 2]

        # 3) Concatenate along the “view” axis to form [B, V, M, 2]:
        group_sum_loss_full = torch.cat(group_slices, dim=1)

        total_all = group_sum_loss_full.sum(dim=(0, 1, 2))
        total_all_t, total_all_s = total_all[0:2].sum(), total_all[2:4].sum()

        view_ids = torch.arange(V, device=device, dtype=torch.long).repeat_interleave(
            Ms, output_size=M
        )
        idx_M = torch.arange(M, device=device, dtype=torch.long)  # [0, 1, …, M-1]
        total_intra = group_sum_loss_full[:, view_ids, idx_M, :].sum(dim=(0, 1))
        total_intra_t, total_intra_s = total_intra[0:2].sum(), total_intra[2:4].sum()
        del group_sum_loss_full

        sum_Ms_sq = (Ms * Ms).sum()
        diag_count_val = sum_Ms_sq * B
        offdiag_count_val = (M * M - sum_Ms_sq) * B

        # 8) Compute inter‐view = all − intra
        total_inter_t = total_all_t - total_intra_t
        total_inter_s = total_all_s - total_intra_s

        # 9) Normalize by diag_count_val and offdiag_count_val
        loss_intra_t = total_intra_t / diag_count_val
        loss_inter_t = total_inter_t / offdiag_count_val
        loss_intra_s = total_intra_s / diag_count_val
        loss_inter_s = total_inter_s / offdiag_count_val

        # 10) Blend with α‐weights
        loss_t = self.alpha_t * loss_inter_t + (1.0 - self.alpha_t) * loss_intra_t
        loss_s = self.alpha_s * loss_inter_s + (1.0 - self.alpha_s) * loss_intra_s
        loss = self.alpha_ts * loss_t + (1.0 - self.alpha_ts) * loss_s
        return {
            "loss_pose_intra_t": loss_intra_t,
            "loss_pose_inter_t": loss_inter_t,
            "loss_pose_intra_s": loss_intra_s,
            "loss_pose_inter_s": loss_inter_s,
            "loss_pose_t": loss_t,
            "loss_pose_s": loss_s,
            "loss_pose": loss,
        }


class OptimizedPoseLossV8(nn.Module):
    def __init__(
        self,
        criterion: str = "mse",
        alpha_t: float = 0.5,
        alpha_s: float = 0.75,
        alpha_ts: float = 0.5,
    ):
        super().__init__()
        self.criterion = F.mse_loss if criterion == "mse" else F.l1_loss
        self.alpha_t = alpha_t
        self.alpha_s = alpha_s
        self.alpha_ts = alpha_ts

    def forward(
        self,
        pred_dT: Float[Tensor, "B M M 4"],
        gt_dT: Float[Tensor, "B M M 4"],
        Ms: Int[Tensor, " V"],  # sum(Ms) == M
    ) -> dict[str, Tensor]:
        device = pred_dT.device
        B, M = pred_dT.shape[0], pred_dT.shape[1]
        V = Ms.shape[0]

        # ──────────────────────────────────────────────────────────
        # 1) One-shot: compute 4-channel per-pair loss → [B, M, M, 4]
        raw4 = self.criterion(pred_dT, gt_dT, reduction="none")  # → [B, M, M, 4]

        # 2) Collapse 4→2 channels in one contiguous step:
        #    view as [B, M, M, 2, 2] then sum over the last dim → [B, M, M, 2]
        loss_full = raw4.view(B, M, M, 2, 2).sum(-1)  # → [B, M, M, 2]
        del raw4

        # 3) Split into two views on the last dimension (no copy—just a view):
        loss_t = loss_full[..., 0]  # [B, M, M], translation‐channel
        loss_s = loss_full[..., 1]  # [B, M, M], scale‐channel

        # 4) Compute “total_all” via two big reductions over [B, M, M]:
        total_all_t = loss_t.sum()  # scalar
        total_all_s = loss_s.sum()  # scalar

        # 5) Build start/end indices for each view block in 0..M-1
        ends = Ms.cumsum(0)  # e.g. [49, 98, 107, …], shape [V]
        starts = ends - Ms  # e.g. [0, 49, 98, …],   shape [V]

        # 6) Build two 1-D index arrays of length K = ∑(Ms[v]^2)
        #    Each (i,j) pair means row i and col j belong to the same v.
        intra_i_list = []
        intra_j_list = []
        for v in range(V):
            s = starts[v].item()
            e = ends[v].item()
            # all pairs i∈[s..e), j∈[s..e)
            # We can do a meshgrid and flatten:
            rows = torch.arange(s, e, device=device)
            cols = torch.arange(s, e, device=device)
            # meshgrid in “ij” indexing yields two [m_v, m_v] matrices
            I, J = torch.meshgrid(rows, cols, indexing="ij")
            intra_i_list.append(I.reshape(-1))
            intra_j_list.append(J.reshape(-1))

        intra_i = torch.cat(intra_i_list, dim=0)  # shape [K]
        intra_j = torch.cat(intra_j_list, dim=0)  # shape [K]
        # Now “(intra_i[k], intra_j[k])” runs over every diagonal block-entry
        # for each view.  K = sum(Ms[v]^2) ≪ M^2.

        # 7) Advanced‐indexing to gather *all* intra‐view errors in one shot:
        #
        #    – loss_t[:, intra_i, intra_j]  → [B, K]
        #    – loss_s[:, intra_i, intra_j]  → [B, K]
        #
        #    Summing over (0,1) gets us total_intra_t, total_intra_s.
        sel_t = loss_t[:, intra_i, intra_j]  # → [B, K]
        sel_s = loss_s[:, intra_i, intra_j]  # → [B, K]

        total_intra_t = sel_t.sum()  # scalar
        total_intra_s = sel_s.sum()  # scalar

        # 8) We no longer need any big intermediate tensors—free them immediately:
        del loss_full, loss_t, loss_s
        del sel_t, sel_s
        del intra_i, intra_j
        del intra_i_list, intra_j_list, starts, ends

        # 9) Compute counts and “inter = all − intra”
        sum_Ms_sq = (Ms * Ms).sum()  # scalar
        diag_count_val = sum_Ms_sq * B
        offdiag_count_val = (M * M - sum_Ms_sq) * B

        total_inter_t = total_all_t - total_intra_t
        total_inter_s = total_all_s - total_intra_s

        # 10) Normalize by diag/off-diag counts
        loss_intra_t = total_intra_t / diag_count_val
        loss_inter_t = total_inter_t / offdiag_count_val
        loss_intra_s = total_intra_s / diag_count_val
        loss_inter_s = total_inter_s / offdiag_count_val

        # 11) α-blend into final scalars
        loss_t_final = self.alpha_t * loss_inter_t + (1.0 - self.alpha_t) * loss_intra_t
        loss_s_final = self.alpha_s * loss_inter_s + (1.0 - self.alpha_s) * loss_intra_s
        loss_final = self.alpha_ts * loss_t_final + (1.0 - self.alpha_ts) * loss_s_final

        return {
            "loss_pose_intra_t": loss_intra_t,
            "loss_pose_inter_t": loss_inter_t,
            "loss_pose_intra_s": loss_intra_s,
            "loss_pose_inter_s": loss_inter_s,
            "loss_pose_t": loss_t_final,
            "loss_pose_s": loss_s_final,
            "loss_pose": loss_final,
        }


class OptimizedPoseLoss(nn.Module):
    def __init__(
        self,
        criterion: str = "mse",
        alpha_t: float = 0.5,
        alpha_s: float = 0.75,
        alpha_ts: float = 0.5,
    ):
        super().__init__()
        self.criterion_fn = F.mse_loss if criterion == "mse" else F.l1_loss
        self.alpha_t = alpha_t
        self.alpha_s = alpha_s
        self.alpha_ts = alpha_ts

    def forward(
        self,
        pred_dT: Float[Tensor, "B M M 4"],
        gt_dT: Float[Tensor, "B M M 4"],
        Ms: Int[Tensor, " V"],  # sum(Ms) == M
    ) -> dict[str, Tensor]:
        device = pred_dT.device
        B, M = pred_dT.shape[0], pred_dT.shape[1]
        V = Ms.shape[0]

        # 1) Compute per‐pair, 4‐channel loss (no reduction)
        loss_full = self.criterion_fn(pred_dT, gt_dT, reduction="none")  # [B, M, M, 4]

        # 2) Compute diag_count and offdiag_count from Ms
        sum_Ms_sq = (Ms * Ms).sum()
        diag_count_val = sum_Ms_sq * B
        offdiag_count_val = (M * M - sum_Ms_sq) * B

        # 3) Sum over all (b,i,j) to get total_all_loss_per_channel: [4]
        total_all_loss_per_channel = loss_full.sum(
            dim=(0, 1, 2)
        )  # Sum over B, M_rows, M_cols
        total_all_t = total_all_loss_per_channel[0:2].sum()
        total_all_s = total_all_loss_per_channel[2:4].sum()

        # 4) Build view_ids = length M, indicating which view each index i∈[0..M-1] belongs to
        view_ids = torch.arange(V, device=device, dtype=torch.long).repeat_interleave(
            Ms, output_size=M
        )

        # 5) Use index_add to get group_sum_loss_full[b, v, j, c] = ∑_{i: view_ids[i]==v} loss_full[b,i,j,c]
        #    Source is loss_full [B,M,M,4]. Index is view_ids [M].
        #    group_sum_loss_full is [B,V,M,4]. index_add_ is along dim 1 (V-dim).
        group_sum_loss_full = loss_full.new_zeros((B, V, M, 4))
        group_sum_loss_full.index_add_(1, view_ids, loss_full)

        # 6) Gather only those terms where source row i and column j are in the same view:
        view_ids_expand_B_M = view_ids.unsqueeze(0).expand(B, M)  # [B,M]
        idx_gather = (
            view_ids_expand_B_M.unsqueeze(1).unsqueeze(-1).expand(B, 1, M, 4)
        )  # [B,1,M,4]
        diag_vals_loss_full = group_sum_loss_full.gather(1, idx_gather).squeeze(
            1
        )  # [B,M,4]

        # 7) Sum diag_vals_loss_full to get total_intra_loss_per_channel: [4]
        total_intra_loss_per_channel = diag_vals_loss_full.sum(
            dim=(0, 1)
        )  # Sum over B, M_cols
        total_intra_t = total_intra_loss_per_channel[0:2].sum()
        total_intra_s = total_intra_loss_per_channel[2:4].sum()

        # 8) Compute inter‐view = all − intra
        total_inter_t = total_all_t - total_intra_t
        total_inter_s = total_all_s - total_intra_s

        # 9) Normalize by diag_count_val and offdiag_count_val
        loss_intra_t = total_intra_t / diag_count_val
        loss_inter_t = total_inter_t / offdiag_count_val
        loss_intra_s = total_intra_s / diag_count_val
        loss_inter_s = total_inter_s / offdiag_count_val

        # 10) Blend with α‐weights
        loss_t = self.alpha_t * loss_inter_t + (1.0 - self.alpha_t) * loss_intra_t
        loss_s = self.alpha_s * loss_inter_s + (1.0 - self.alpha_s) * loss_intra_s
        loss = self.alpha_ts * loss_t + (1.0 - self.alpha_ts) * loss_s

        return {
            "loss_pose_intra_t": loss_intra_t,
            "loss_pose_inter_t": loss_inter_t,
            "loss_pose_intra_s": loss_intra_s,
            "loss_pose_inter_s": loss_inter_s,
            "loss_pose_t": loss_t,
            "loss_pose_s": loss_s,
            "loss_pose": loss,
        }


class OptimizedPoseLossV10(nn.Module):
    def __init__(
        self,
        criterion: str = "mse",
        alpha_t: float = 0.5,
        alpha_s: float = 0.75,
        alpha_ts: float = 0.5,
    ):
        """
        Fully‐corrected “single‐loss” version. Main changes:
         - We compute the full [B, M, M, 4] once, collapse it to [B, M, M, 2].
         - We compute `total_all` by summing that [B, M, M, 2].
         - We compute `total_intra` by summing each view’s own square block individually,
           instead of summing a combined ‘run’ block.
         - No fancy indexing or scatter; just two nested loops over V small blocks (V=12 here).
         - Minimal GPU↔CPU syncs, no `.item()` calls in the hot path.
        """
        super().__init__()
        self.criterion = mse_loss if criterion == "mse" else l1_loss
        self.alpha_t = alpha_t
        self.alpha_s = alpha_s
        self.alpha_ts = alpha_ts

    def forward(
        self,
        pred_dT: Float[Tensor, "B M M 4"],
        gt_dT: Float[Tensor, "B M M 4"],
        # Ms: Int[Tensor, "V"],  # sum(Ms) == M
        Ms: list[int],  # sum(Ms) == M, and each “view” is a contiguous block
    ) -> dict[str, Tensor]:
        # —————————— 1) Prep and compute full 4‐channel loss ——————————
        B, M = pred_dT.shape[0], pred_dT.shape[1]

        # Compute raw 4‐channel per‐pair residual: shape [B, M, M, 4]
        loss_full = self.criterion(pred_dT, gt_dT, reduction="none")

        # —————————— 2) total_all over (b,i,j) ——————————
        # Sum “translation‐channel” and “scale‐channel” across all b,i,j:
        total_all = loss_full.sum(dim=(0, 1, 2))
        total_all_t, total_all_s = total_all[0:2].sum(), total_all[2:4].sum()

        # —————————— 3) Build per‐view start/end offsets on CPU (no GPU sync in loop) ——————————
        # Ms might be [49,49,9,9,…], so we want each view’s own block.
        view_ends = list(itertools.accumulate(Ms))
        view_begins = [0] + view_ends[:-1]
        # Then s = 0; e = 49 for view 0; s=49,e=98 for view1; etc.

        # —————————— 4) total_intra: sum each view’s square block individually ——————————
        total_intra = sum(
            [
                loss_full[:, s:e, s:e, :].sum(dim=(0, 1, 2))
                for s, e in zip(view_begins, view_ends)
            ]
        )
        total_intra_t, total_intra_s = total_intra[0:2].sum(), total_intra[2:4].sum()

        # Free the big intermediate now that we have both “all” and “intra”
        del loss_full

        # —————————— 5) Compute normalization counts ——————————
        # sum_Ms_sq = Σ_v (Ms[v]^2). Then:
        sum_Ms_sq = sum(m**2 for m in Ms)
        diag_count = sum_Ms_sq * B
        offdiag_count = (M * M - sum_Ms_sq) * B

        # —————————— 6) Compute “inter = all − intra”, normalize, blend with α ——————————
        total_inter_t = total_all_t - total_intra_t
        total_inter_s = total_all_s - total_intra_s

        loss_intra_t = total_intra_t / diag_count
        loss_inter_t = total_inter_t / offdiag_count
        loss_intra_s = total_intra_s / diag_count
        loss_inter_s = total_inter_s / offdiag_count

        loss_t = self.alpha_t * loss_inter_t + (1.0 - self.alpha_t) * loss_intra_t
        loss_s = self.alpha_s * loss_inter_s + (1.0 - self.alpha_s) * loss_intra_s
        loss = self.alpha_ts * loss_t + (1.0 - self.alpha_ts) * loss_s

        return {
            "loss_pose_intra_t": loss_intra_t,
            "loss_pose_inter_t": loss_inter_t,
            "loss_pose_intra_s": loss_intra_s,
            "loss_pose_inter_s": loss_inter_s,
            "loss_pose_t": loss_t,
            "loss_pose_s": loss_s,
            "loss_pose": loss,
        }
    



class OptimizedPoseLossV11(nn.Module):
    def __init__(
        self,
        criterion: str = "mse",
        alpha_t: float = 0.5,
        alpha_s: float = 0.75,
        alpha_ts: float = 0.5,
    ):
        """
        Fully‐corrected “single‐loss” version. Main changes:
         - We compute the full [B, M, M, 4] once, collapse it to [B, M, M, 2].
         - We compute `total_all` by summing that [B, M, M, 2].
         - We compute `total_intra` by summing each view’s own square block individually,
           instead of summing a combined ‘run’ block.
         - No fancy indexing or scatter; just two nested loops over V small blocks (V=12 here).
         - Minimal GPU↔CPU syncs, no `.item()` calls in the hot path.
        """
        super().__init__()
        self.criterion = mse_loss if criterion == "mse" else l1_loss
        self.alpha_t = alpha_t
        self.alpha_s = alpha_s
        self.alpha_ts = alpha_ts


    def forward(
        self,
        pred_dT: Tensor,       # shape [B, M, M, 4]
        gt_dT:   Tensor,       # shape [B, M, M, 4]
        Ms:      list[int],    # sum(Ms) == M, and each “view” block is contiguous
    ) -> dict[str, Tensor]:
        B, M = pred_dT.shape[0], pred_dT.shape[1]

        # —————————— 1) Build per‐block boundaries on CPU ——————————
        view_ends   = list(itertools.accumulate(Ms))
        view_begins = [0] + view_ends[:-1]
        # Now block v spans rows/cols [view_begins[v] : view_ends[v]]

        # —————————— 2) Identify contiguous runs of equal block‐sizes ——————————
        # Each run = (size, start_block_idx, run_count)
        runs: list[tuple[int, int, int]] = []
        idx = 0
        for size, group in itertools.groupby(Ms):
            run_count = sum(1 for _ in group)
            runs.append((size, idx, run_count))
            idx += run_count
        # e.g. if Ms = [49,49,9,9,9,9], then runs = [(49, 0, 2), (9, 2, 4)]

        # —————————— 3) One loop to compute both total_all and total_intra ——————————
        total_all = pred_dT.new_zeros((4,))
        total_intra = pred_dT.new_zeros((4,))

        for size_v, start_v, count_v in runs:
            # row‐span for this run (#rows = count_v * size_v)
            s_v = view_begins[start_v]
            e_v = view_ends[start_v + count_v - 1]

            # ────────── Slice rows [s_v:e_v] over all columns ──────────
            # pred_row: [B, run_count*size_v, M, 4]
            pred_row = pred_dT[:, s_v:e_v, :, :]
            gt_row   = gt_dT[:,   s_v:e_v, :, :]

            # Compute loss over that stripe of rows
            loss_row = self.criterion(pred_row, gt_row, reduction="none")
            # loss_row.shape == [B, run_count*size_v, M, 4]

            # ────────── 3a) Accumulate into total_all by summing all (b, i, j) in this stripe ──────────
            sum_row = loss_row.sum(dim=(0, 1, 2))  # shape [4]
            total_all = total_all + sum_row

            # ────────── 3b) From loss_row, extract the columns [s_v:e_v] to get the square for intra ──────────
            # loss_diag: [B, run_count*size_v, run_count*size_v, 4]
            loss_diag = loss_row[:, :, s_v:e_v, :]
            # ────────── Extract only the diagonal sub‐blocks of size size_v×size_v within loss_diag ──────────
            run = count_v
            sv  = size_v
            # 1) reshape → [B, run, sv, run, sv, 4]
            sum_diag = loss_diag.view(B, run, sv, run, sv, 4).diagonal(dim1=1, dim2=3).sum(dim=(0, 1, 2, 4))
            total_intra = total_intra + sum_diag

        # —————————— 4) Split channels into t vs s ——————————
        total_all_t   = total_all[0:2].sum()    # scalar
        total_all_s   = total_all[2:4].sum()    # scalar
        total_intra_t = total_intra[0:2].sum()  # scalar
        total_intra_s = total_intra[2:4].sum()  # scalar

        # —————————— 5) Compute normalization counts ——————————
        sum_Ms_sq     = sum(m * m for m in Ms)       # Σ_v (Ms[v]^2)
        diag_count    = sum_Ms_sq * B
        offdiag_count = (M * M - sum_Ms_sq) * B

        # —————————— 6) Compute inter/intra losses + blend with α ——————————
        total_inter_t = total_all_t   - total_intra_t
        total_inter_s = total_all_s   - total_intra_s

        loss_intra_t = total_intra_t / diag_count
        loss_inter_t = total_inter_t / offdiag_count
        loss_intra_s = total_intra_s / diag_count
        loss_inter_s = total_inter_s / offdiag_count

        loss_t = self.alpha_t * loss_inter_t + (1.0 - self.alpha_t) * loss_intra_t
        loss_s = self.alpha_s * loss_inter_s + (1.0 - self.alpha_s) * loss_intra_s
        loss   = self.alpha_ts * loss_t       + (1.0 - self.alpha_ts) * loss_s

        return {
            "loss_pose_intra_t": loss_intra_t,
            "loss_pose_inter_t": loss_inter_t,
            "loss_pose_intra_s": loss_intra_s,
            "loss_pose_inter_s": loss_inter_s,
            "loss_pose_t":       loss_t,
            "loss_pose_s":       loss_s,
            "loss_pose":         loss,
        }


class OptimizedPoseLossV4(nn.Module):
    def __init__(
        self,
        criterion: str = "mse",
        alpha_t: float = 0.5,
        alpha_s: float = 0.75,
        alpha_ts: float = 0.5,
        m_chunk_size: int = 64,  # Chunk size for processing M_rows dimension
    ):
        super().__init__()
        self.criterion_fn = F.mse_loss if criterion == "mse" else F.l1_loss
        self.alpha_t = alpha_t
        self.alpha_s = alpha_s
        self.alpha_ts = alpha_ts
        self.m_chunk_size = m_chunk_size

    def forward(
        self,
        pred_dT: Float[Tensor, "B M M 4"],
        gt_dT: Float[Tensor, "B M M 4"],
        Ms: Int[Tensor, " V"],  # sum(Ms) == M
    ) -> dict[str, Tensor]:
        device = pred_dT.device
        dtype = pred_dT.dtype
        B, M, _, C = pred_dT.shape  # Assuming M_rows == M_cols == M, C should be 4
        V = Ms.shape[0]

        # 2) Compute diag_count and offdiag_count from Ms
        sum_Ms_sq = torch.sum(Ms * Ms)
        diag_pairs_per_batch = sum_Ms_sq
        total_pairs_per_batch = M * M

        diag_count_val = diag_pairs_per_batch * B
        offdiag_count_val = (total_pairs_per_batch - diag_pairs_per_batch) * B
        # Ensure counts are not zero to prevent division by zero if M=0 or B=0 (though unlikely in practice)
        # Using float for counts to avoid type issues in division if losses are float
        diag_count_val = diag_count_val.float()
        offdiag_count_val = offdiag_count_val.float()

        # Calculate total_all_loss_per_channel by chunking M_rows
        total_all_loss_per_channel = torch.zeros(C, device=device, dtype=dtype)
        if M > 0:
            for r_start in range(0, M, self.m_chunk_size):
                r_end = min(r_start + self.m_chunk_size, M)
                # loss_chunk_rows: [B, r_end-r_start, M, C]
                loss_chunk_rows = self.criterion_fn(
                    pred_dT[:, r_start:r_end, :, :],
                    gt_dT[:, r_start:r_end, :, :],
                    reduction="none",
                )
                total_all_loss_per_channel += loss_chunk_rows.sum(
                    dim=(0, 1, 2)
                )  # Sum over B, chunk_rows, M_cols
                del loss_chunk_rows

        total_all_t = total_all_loss_per_channel[0:2].sum()
        total_all_s = total_all_loss_per_channel[2:4].sum()

        # Calculate total_intra_loss_per_channel by iterating through views
        total_intra_loss_per_channel = torch.zeros(C, device=device, dtype=dtype)
        if V > 0 and M > 0:
            view_limits = Ms.cumsum(0)
            starts = torch.cat(
                (torch.tensor([0], device=device, dtype=Ms.dtype), view_limits[:-1])
            )
            ends = view_limits

            for v_idx in range(V):
                s, e = starts[v_idx].item(), ends[v_idx].item()
                if s == e:
                    continue

                # loss_vv: [B, Ms[v_idx], Ms[v_idx], C]
                loss_vv = self.criterion_fn(
                    pred_dT[:, s:e, s:e, :], gt_dT[:, s:e, s:e, :], reduction="none"
                )
                total_intra_loss_per_channel += loss_vv.sum(
                    dim=(0, 1, 2)
                )  # Sum over B, Ms[v]_rows, Ms[v]_cols
                del loss_vv

        total_intra_t = total_intra_loss_per_channel[0:2].sum()
        total_intra_s = total_intra_loss_per_channel[2:4].sum()

        # 8) Compute inter‐view = all − intra
        total_inter_t = total_all_t - total_intra_t
        total_inter_s = total_all_s - total_intra_s

        # 9) Normalize
        loss_intra_t = (
            total_intra_t / diag_count_val
            if diag_count_val > 1e-8
            else torch.zeros_like(total_intra_t)
        )
        loss_inter_t = (
            total_inter_t / offdiag_count_val
            if offdiag_count_val > 1e-8
            else torch.zeros_like(total_inter_t)
        )
        loss_intra_s = (
            total_intra_s / diag_count_val
            if diag_count_val > 1e-8
            else torch.zeros_like(total_intra_s)
        )
        loss_inter_s = (
            total_inter_s / offdiag_count_val
            if offdiag_count_val > 1e-8
            else torch.zeros_like(total_inter_s)
        )

        # 10) Blend with α‐weights
        loss_t = self.alpha_t * loss_inter_t + (1.0 - self.alpha_t) * loss_intra_t
        loss_s = self.alpha_s * loss_inter_s + (1.0 - self.alpha_s) * loss_intra_s
        loss = self.alpha_ts * loss_t + (1.0 - self.alpha_ts) * loss_s

        return {
            "loss_pose_intra_t": loss_intra_t,
            "loss_pose_inter_t": loss_inter_t,
            "loss_pose_intra_s": loss_intra_s,
            "loss_pose_inter_s": loss_inter_s,
            "loss_pose_t": loss_t,
            "loss_pose_s": loss_s,
            "loss_pose": loss,
        }


class OptimizedPoseLossV5(nn.Module):
    def __init__(
        self,
        criterion: str = "mse",
        alpha_t: float = 0.5,
        alpha_s: float = 0.75,
        alpha_ts: float = 0.5,
    ):
        super().__init__()
        self.criterion_fn = F.mse_loss if criterion == "mse" else F.l1_loss
        self.alpha_t = alpha_t
        self.alpha_s = alpha_s
        self.alpha_ts = alpha_ts

    def forward(
        self,
        pred_dT: Float[Tensor, "B M M 4"],
        gt_dT: Float[Tensor, "B M M 4"],
        Ms: Int[Tensor, " V"],  # sum(Ms) == M
    ) -> dict[str, Tensor]:
        device = pred_dT.device
        dtype = pred_dT.dtype  # Added from V4
        B, M, _, C = pred_dT.shape  # C should be 4, M_rows = M_cols = M
        V = Ms.shape[0]

        # 1) Compute per‐pair, C‐channel loss (no reduction)
        loss_full = self.criterion_fn(pred_dT, gt_dT, reduction="none")  # [B, M, M, C]

        # 2) Compute diag_count and offdiag_count from Ms (refined from V4)
        sum_Ms_sq = torch.sum(Ms * Ms)
        diag_pairs_per_batch = sum_Ms_sq
        total_pairs_per_batch = M * M

        diag_count_val = (diag_pairs_per_batch * B).float()  # Cast to float
        offdiag_count_val = (
            (total_pairs_per_batch - diag_pairs_per_batch) * B
        ).float()  # Cast to float

        # 3) Sum over all (b,i,j) to get total_all_loss_per_channel: [C]
        total_all_loss_per_channel = loss_full.sum(
            dim=(0, 1, 2)
        )  # Sum over B, M_rows, M_cols
        total_all_t = total_all_loss_per_channel[0:2].sum()
        total_all_s = total_all_loss_per_channel[2:4].sum()

        # 4) Build view_ids = length M, indicating which view each index i∈[0..M-1] belongs to
        view_ids = torch.arange(V, device=device, dtype=torch.long).repeat_interleave(
            Ms, output_size=M
        )

        # 5) Use index_add to get group_sum_loss_full[b, v, j, c] = ∑_{i: view_ids[i]==v} loss_full[b,i,j,c]
        #    Source is loss_full [B,M,M,C]. Index is view_ids [M].
        #    group_sum_loss_full is [B,V,M,C]. index_add_ is along dim 1 (V-dim).
        #    Initialize with zeros of the correct dtype (from V4)
        group_sum_loss_full = torch.zeros((B, V, M, C), device=device, dtype=dtype)
        group_sum_loss_full.index_add_(1, view_ids, loss_full)

        # 6) Gather only those terms where source row i and column j are in the same view:
        view_ids_expand_B_M = view_ids.unsqueeze(0).expand(B, M)  # [B,M]
        # Ensure idx_gather has the correct shape for C channels
        idx_gather = (
            view_ids_expand_B_M.unsqueeze(1).unsqueeze(-1).expand(B, 1, M, C)
        )  # [B,1,M,C]
        diag_vals_loss_full = group_sum_loss_full.gather(1, idx_gather).squeeze(
            1
        )  # [B,M,C]

        # 7) Sum diag_vals_loss_full to get total_intra_loss_per_channel: [C]
        total_intra_loss_per_channel = diag_vals_loss_full.sum(
            dim=(0, 1)
        )  # Sum over B, M_cols
        total_intra_t = total_intra_loss_per_channel[0:2].sum()
        total_intra_s = total_intra_loss_per_channel[2:4].sum()

        # 8) Compute inter‐view = all − intra
        total_inter_t = total_all_t - total_intra_t
        total_inter_s = total_all_s - total_intra_s

        # 9) Normalize by diag_count_val and offdiag_count_val (with safe division from V4)
        loss_intra_t = (
            total_intra_t / diag_count_val
            if diag_count_val > 1e-8
            else torch.zeros_like(total_intra_t)
        )
        loss_inter_t = (
            total_inter_t / offdiag_count_val
            if offdiag_count_val > 1e-8
            else torch.zeros_like(total_inter_t)
        )
        loss_intra_s = (
            total_intra_s / diag_count_val
            if diag_count_val > 1e-8
            else torch.zeros_like(total_intra_s)
        )
        loss_inter_s = (
            total_inter_s / offdiag_count_val
            if offdiag_count_val > 1e-8
            else torch.zeros_like(total_inter_s)
        )

        # 10) Blend with α‐weights
        loss_t = self.alpha_t * loss_inter_t + (1.0 - self.alpha_t) * loss_intra_t
        loss_s = self.alpha_s * loss_inter_s + (1.0 - self.alpha_s) * loss_intra_s
        loss = self.alpha_ts * loss_t + (1.0 - self.alpha_ts) * loss_s

        return {
            "loss_pose_intra_t": loss_intra_t,
            "loss_pose_inter_t": loss_inter_t,
            "loss_pose_intra_s": loss_intra_s,
            "loss_pose_inter_s": loss_inter_s,
            "loss_pose_t": loss_t,
            "loss_pose_s": loss_s,
            "loss_pose": loss,
        }


@pytest.fixture(
    params=[
        "base",
        "optimized_v6",
        "optimized",
        # "optimized_v8",
        # "optimized_v7",
        # "optimized_v4",
        # "optimized_v5",
        # "optimized_v10",
        # "optimized_v11",
    ]
)
def poseloss_fn(request):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if request.param == "base":
        return PoseLoss().to(device)
    elif request.param == "optimized":
        return OptimizedPoseLoss().to(device)
    elif request.param == "optimized_v6":
        return OptimizedPoseLossV6().to(device)
    elif request.param == "optimized_v4":
        return OptimizedPoseLossV4().to(device)
    elif request.param == "optimized_v5":
        return OptimizedPoseLossV5().to(device)
    elif request.param == "optimized_v7":
        return OptimizedPoseLossV7().to(device)
    elif request.param == "optimized_v8":
        return OptimizedPoseLossV8().to(device)
    elif request.param == "optimized_v10":
        return OptimizedPoseLossV10().to(device)
    elif request.param == "optimized_v11":
        return OptimizedPoseLossV11().to(device)
    else:
        raise ValueError(f"Unknown poseloss_fn: {request.param}")


def test_poseloss_full(poseloss_fn, benchmark_v2):
    """
    Benchmark the full PoseLoss: base vs index vs scatter.
    """
    B = 256
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 196*0.25* 2 + 36*0.25*10
    gV = 2
    gM = 49
    lV = 10
    lM = 9
    M = gV * gM + lV * lM  # 188

    Ms = torch.tensor(
        [gM] * gV + [lM] * lV, dtype=torch.int64, device=device
    )  # [49, 49, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]

    # random inputs
    pred_dT = torch.randn(B, M, M, 4, device=device)
    gt_dT = torch.randn(B, M, M, 4, device=device)

    fn = poseloss_fn

    def run(*inputs):
        return fn(*inputs)
    # If the poseloss_fn is OptimizedPoseLossV10 or V11, we need to pass Ms as a lis
    if  "PoseLoss" == poseloss_fn.__class__.__name__:
        # For PoseLoss, we pass Ms as a tensor
        V = gV + lV
        view_ids = torch.arange(V, device=device).repeat_interleave(Ms, output_size=M)
        Ms = view_ids
    if "v10" in poseloss_fn.__class__.__name__.lower():
        # For OptimizedPoseLossV10, we pass Ms as a list
        Ms = Ms.cpu().tolist()

    if "v11" in poseloss_fn.__class__.__name__.lower():
        # For OptimizedPoseLossV10, we pass Ms as a list
        Ms = Ms.cpu().tolist()

    # warmup & benchmark
    benchmark_v2.benchmark(run, args=(pred_dT, gt_dT, Ms), n_warmup=100, n_runs=300)

    # keep only timing columns
    benchmark_v2.drop_columns(exclude=["time/min (ms)", "time/max (ms)"])


# ---------------------------------------------------
# Equivalence test
# ---------------------------------------------------


def test_poseloss_equivalence():
    """Ensure base, index and scatter produce the same loss (in float64)."""
    B = 8  # Smaller values for quicker test
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Generate Ms such that sum(Ms) = M
    gV = 2
    gM = 49
    lV = 10
    lM = 9
    M = gV * gM + lV * lM  # 188

    Ms = torch.tensor(
        [gM] * gV + [lM] * lV, dtype=torch.int64, device=device
    )  # [49, 49, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
    Ms_device = Ms.to(device)

    # use double precision for exactness
    pred_dT = torch.randn(B, M, M, 4, device=device, dtype=torch.double)
    gt_dT = torch.randn(B, M, M, 4, device=device, dtype=torch.double)
    # gt_dT = gt_dT + gt_dT.transpose(1, 2) # make it symmetric

    # cast all modules to double
    m_base = PoseLoss().to(device).double()
    m_optimized = OptimizedPoseLoss().to(device).double()
    m_optimized_v4 = OptimizedPoseLossV4().to(device).double()
    m_optimized_v5 = OptimizedPoseLossV5().to(device).double()
    m_optimized_v6 = OptimizedPoseLossV6().to(device).double()
    m_optimized_v7 = OptimizedPoseLossV7().to(device).double()
    m_optimized_v10 = OptimizedPoseLossV10().to(device).double()
    m_optimized_v11 = OptimizedPoseLossV11().to(device).double()

    loss_base = m_base(pred_dT, gt_dT, Ms_device)
    loss_optimized = m_optimized(pred_dT, gt_dT, Ms_device)
    loss_optimized_v4 = m_optimized_v4(pred_dT, gt_dT, Ms_device)
    loss_optimized_v5 = m_optimized_v5(pred_dT, gt_dT, Ms_device)
    loss_optimized_v6 = m_optimized_v6(pred_dT, gt_dT, Ms_device)
    loss_optimized_v7 = m_optimized_v7(pred_dT, gt_dT, Ms_device)
    loss_optimized_v10 = m_optimized_v10(pred_dT, gt_dT, Ms_device.cpu().tolist())
    loss_optimized_v11 = m_optimized_v11(pred_dT, gt_dT, Ms_device.cpu().tolist())

    # now they’ll agree to within 1e-9 or so
    for key in loss_base:
        torch.testing.assert_close(
            loss_base[key],
            loss_optimized[key],
            rtol=1e-9,
            atol=1e-9,
            msg=f"Base vs Optimized mismatch for {key}",
        )
        torch.testing.assert_close(
            loss_base[key],
            loss_optimized_v4[key],
            rtol=1e-9,
            atol=1e-9,
            msg=f"Base vs OptimizedV4 mismatch for {key}",
        )
        torch.testing.assert_close(
            loss_base[key],
            loss_optimized_v5[key],
            rtol=1e-9,
            atol=1e-9,
            msg=f"Base vs OptimizedV5 mismatch for {key}",
        )
        torch.testing.assert_close(
            loss_base[key],
            loss_optimized_v6[key],
            rtol=1e-9,
            atol=1e-9,
            msg=f"Base vs OptimizedV6 mismatch for {key}",
        )
        torch.testing.assert_close(
            loss_base[key],
            loss_optimized_v7[key],
            rtol=1e-9,
            atol=1e-9,
            msg=f"Base vs OptimizedV7 mismatch for {key}",
        )
        torch.testing.assert_close(
            loss_base[key],
            loss_optimized_v10[key],
            rtol=1e-9,
            atol=1e-9,
            msg=f"Base vs OptimizedV10 mismatch for {key}",
        )
        torch.testing.assert_close(
            loss_base[key],
            loss_optimized_v11[key],
            rtol=1e-9,
            atol=1e-9,
            msg=f"Base vs OptimizedV11 mismatch for {key}",
        )
