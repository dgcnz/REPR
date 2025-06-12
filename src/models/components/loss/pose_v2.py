import torch
from torch import nn
import torch.nn.functional as F
import itertools


class PoseHead(nn.Module):
    """
    Prediction head for pairwise pose differences with optional variance/scale.
    """

    def __init__(
        self,
        embed_dim: int,
        num_targets: int,
        apply_tanh: bool = False,
        predict_uncertainty: bool = False,
        eps: float = 1e-3,
    ):
        super().__init__()
        self.num_targets = num_targets
        self.apply_tanh = apply_tanh
        self.predict_uncertainty = predict_uncertainty
        self.eps = eps
        self.tanh = nn.Tanh() if apply_tanh else nn.Identity()
        out_dim = num_targets * (2 if predict_uncertainty else 1)
        self.linear = nn.Linear(embed_dim, out_dim, bias=False)

    def forward(self, z: torch.Tensor):
        """
        z: [B, M, D]
        returns:
          mu_dT: [B, M, M, K]
          logvar_dT (optional): [B, M, M, K]
        """
        out = self.linear(z)
        if not self.predict_uncertainty:
            return (self.tanh(out.unsqueeze(2) - out.unsqueeze(1)), None)  # [B, M, M, K]
        
        mu_pred, var_pred = out.chunk(2, dim=-1)
        mu_pred = self.tanh(mu_pred.unsqueeze(2) - mu_pred.unsqueeze(1))  # [B, M, M, K]
        var_pred = F.softplus(var_pred) + self.eps
        var_pred = var_pred.unsqueeze(2) + var_pred.unsqueeze(1)
        var_pred = torch.log(var_pred)
        return (mu_pred, var_pred)


class PoseLoss(nn.Module):
    """
    Pose loss with optional heteroscedastic component; criterion decides distribution.
    """

    def __init__(
        self,
        criterion: str = "mse",  # 'mse' for Gaussian, 'l1' for Laplace
        alpha_t: float = 0.5,
        alpha_s: float = 0.75,
        alpha_ts: float = 0.5,
        predict_uncertainty: bool = False,
    ):
        super().__init__()
        self.predict_uncertainty = predict_uncertainty
        self.criterion_str = criterion
        self.alpha_t = alpha_t
        self.alpha_s = alpha_s
        self.alpha_ts = alpha_ts
        self.criterion = torch.square if criterion == "mse" else torch.abs
        self.nll_scalar = 0.5 if criterion == "mse" else 1.0

    def forward(
        self, pred: tuple, gt_dT: torch.Tensor, Ms: list[int]
    ) -> dict[str, torch.Tensor]:
        B = gt_dT.shape[0]
        mu, logvar = pred
        nll = self.criterion(mu - gt_dT)
        if self.predict_uncertainty:
            nll = self.nll_scalar * ((nll / torch.exp(logvar)) + logvar)

        total = nll.sum(dim=(0, 1, 2))  # [K]
        t_all = total[0:2].sum()
        s_all = total[2:4].sum()

        view_ends = list(itertools.accumulate(Ms))
        view_begins = [0] + view_ends[:-1]
        intra = sum(
            nll[:, s:e, s:e, :].sum(dim=(0, 1, 2))
            for s, e in zip(view_begins, view_ends)
        )
        t_intra = intra[0:2].sum()
        s_intra = intra[2:4].sum()

        M = nll.shape[1]
        sum_Ms_sq = sum(m * m for m in Ms)
        diag_count = sum_Ms_sq * B
        offdiag_count = (M * M - sum_Ms_sq) * B

        t_inter = t_all - t_intra
        s_inter = s_all - s_intra

        loss_intra_t = t_intra / diag_count
        loss_inter_t = t_inter / offdiag_count
        loss_intra_s = s_intra / diag_count
        loss_inter_s = s_inter / offdiag_count

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