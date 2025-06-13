import torch
from torch import nn
import torch.nn.functional as F
import itertools
import math


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
    ):
        super().__init__()
        self.num_targets = num_targets
        self.apply_tanh = apply_tanh
        self.predict_uncertainty = predict_uncertainty
        self.tanh = nn.Tanh() if apply_tanh else nn.Identity()
        self.mu = nn.Linear(embed_dim, num_targets, bias=False)
        if self.predict_uncertainty:
            self.logvar = nn.Linear(embed_dim, num_targets, bias=True)
            self.min_logvar = -14 # this is for 512x512 images
            self.max_logvar = 2

        self.initialize_weights()  # this will be called manually in the model

    def initialize_weights(self):
        if self.predict_uncertainty:
            nn.init.zeros_(self.logvar.weight)
            init_logvar = math.log((0.1**2)) * 0.5
            nn.init._no_grad_fill_(self.logvar.bias, init_logvar)
        nn.init.zeros_(self.mu.weight)

    def forward(self, z: torch.Tensor):
        """
        z: [B, M, D]
        """
        mu_z = self.mu(z)
        mu_dT = self.tanh(mu_z.unsqueeze(2) - mu_z.unsqueeze(1))
        if not self.predict_uncertainty:
            return (mu_dT, None)

        # 1) per-token raw logvar, clamped

        raw_lv = self.logvar(z).clamp(self.min_logvar, self.max_logvar)  # [B, M, K]

        # 2) pairwise var = σ_i² + σ_j² in log-space
        #    logvar_dT = log(exp(lv_i) + exp(lv_j))
        logvar_dT = torch.logaddexp(raw_lv.unsqueeze(2), raw_lv.unsqueeze(1))
        # problem, logvar_dT.mean() is too high at the start of training (~1.7-2.1)
        # and then goes down rapidly to 0.2
        return (mu_dT, logvar_dT)


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
        self.nll_a = 0.5 if criterion == "mse" else 1.0
        self.nll_c = math.log(2 * math.pi) if criterion == "mse" else math.log(2)
        assert criterion in ["mse", "l1"], f"Unsupported criterion: {criterion}"

    def forward(
        self, pred: tuple, gt_dT: torch.Tensor, Ms: list[int]
    ) -> dict[str, torch.Tensor]:
        B = gt_dT.shape[0]
        mu, logvar = pred
        nll = self.criterion(mu - gt_dT)
        if self.predict_uncertainty:
            nll = self.nll_a * ((nll / torch.exp(logvar)) + logvar + self.nll_c)

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
