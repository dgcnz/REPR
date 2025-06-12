from torch import nn, Tensor
import itertools
from jaxtyping import Float
from torch.nn.functional import mse_loss, l1_loss


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
        Ms: list[int],  # sum(Ms) == M, and each “view” is a contiguous block
    ) -> dict[str, Tensor]:
        B, M = pred_dT.shape[0], pred_dT.shape[1]
        loss_full = self.criterion(pred_dT, gt_dT, reduction="none")
        total_all = loss_full.sum(dim=(0, 1, 2))
        total_all_t, total_all_s = total_all[0:2].sum(), total_all[2:4].sum()
        view_ends = list(itertools.accumulate(Ms))
        view_begins = [0] + view_ends[:-1]
        total_intra = sum(
            [
                loss_full[:, s:e, s:e, :].sum(dim=(0, 1, 2))
                for s, e in zip(view_begins, view_ends)
            ]
        )
        total_intra_t, total_intra_s = total_intra[0:2].sum(), total_intra[2:4].sum()

        sum_Ms_sq = sum(m**2 for m in Ms)
        diag_count = sum_Ms_sq * B
        offdiag_count = (M * M - sum_Ms_sq) * B

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