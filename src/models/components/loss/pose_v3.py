import torch
from torch import nn
from torch import Tensor
import itertools
import math
from typing import Literal


class PoseHead(nn.Module):
    """
    Predict pairwise pose means and (optionally) uncertainties.
    uncertainty_mode:
      'none'       = mu_ij = W(z_i - z_j), no disp
      'additive'   = + per-token disp σ_i²: disp_ij = σ_i² + σ_j²
      'correlated' = additive + corr term: disp_ij = σ_i²+σ_j² - 2ρ_ij σ_i σ_j

    For correlated mode we want:
    - At initialization, E[p_ij] ≈ 0 and only  p_ij = 1 when i=j.
    """

    def __init__(
        self,
        embed_dim: int,
        proj_embed_dim: int,
        num_targets: int,
        apply_tanh: bool = False,
        uncertainty_mode: Literal[
            "none", "additive", "correlated", "correlated_proj"
        ] = "none",
        min_logdisp: float = -14.0,
        max_logdisp: float = +2.0,
    ):
        super().__init__()
        assert uncertainty_mode in ("none", "additive", "correlated", "correlated_proj")
        self.num_targets = num_targets
        self.apply_tanh = apply_tanh
        self.embed_dim = embed_dim
        self.proj_embed_dim = proj_embed_dim
        self.uncertainty_mode = uncertainty_mode
        self.tanh = nn.Tanh() if apply_tanh else nn.Identity()

        # mean head (no bias since mu_ij = Wz_i - Wz_j)
        self.mu_proj = nn.Linear(embed_dim, num_targets, bias=False)

        # dispersion head(s)
        self.min_logdisp = min_logdisp
        self.max_logdisp = max_logdisp
        if uncertainty_mode in ("additive", "correlated", "correlated_proj"):
            # per-token log-dispersion
            self.disp_proj = nn.Linear(embed_dim, num_targets, bias=True)
        if uncertainty_mode == "correlated":
            self.gate_dim = 16
            self.gate_proj = nn.Linear(self.embed_dim, self.gate_dim, bias=False)
        if uncertainty_mode == "correlated_proj":
            self.gate_dim = 16
            self.gate_proj = nn.Linear(self.proj_embed_dim, self.gate_dim, bias=False)

        self.initialize_weights()

    def initialize_weights(self):
        # zero-init mu so starting mean=0
        nn.init.zeros_(self.mu_proj.weight)
        init_lv = math.log(0.1**2) * 0.5
        assert self.min_logdisp < init_lv < self.max_logdisp
        if self.uncertainty_mode == "additive":
            nn.init.zeros_(self.disp_proj.weight)
            nn.init._no_grad_fill_(self.disp_proj.bias, init_lv)
        if self.uncertainty_mode in ["correlated", "correlated_proj"]:
            # nn.init.normal_(self.disp_proj.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.disp_proj.weight)
            nn.init._no_grad_fill_(self.disp_proj.bias, init_lv)
            nn.init._no_grad_normal_(self.gate_proj.weight, mean=0.0, std=0.001)

    def forward(self, z: Tensor, proj: Tensor):
        """
        Args:
          z: [B, M, D]  token embeddings after encoder+proj+decoder
        Returns:
          mu_dT:    [B, M, M, K]
          logdisp_dT or None: [B, M, M, K]
        """
        out = dict()

        h = self.mu_proj(z)  # [B, M, K]
        out["pred_dT"] = self.tanh(h.unsqueeze(2) - h.unsqueeze(1))  # [B, M, M, K]

        if self.uncertainty_mode == "none":
            return out

        logdisp_tok = self.disp_proj(z).clamp(self.min_logdisp, self.max_logdisp)
        logdisp_pair = torch.logaddexp(
            logdisp_tok.unsqueeze(2), logdisp_tok.unsqueeze(1)
        )
        out["disp_T"] = logdisp_tok.detach()
        out["loss_pose_disp_T_mean"] = out["disp_T"].mean()
        out["loss_pose_disp_T_std"] = out["disp_T"].std()

        if self.uncertainty_mode in ["correlated", "correlated_proj"]:
            # --- Correlated Case ---
            gate_in = z if self.uncertainty_mode == "correlated" else proj
            h_gate = self.gate_proj(gate_in)  # [B, M, gate_dim]
            # Use bmm for explicit batched matrix multiplication
            similarity_logits = torch.bmm(h_gate, h_gate.transpose(-1, -2))  # [B, M, M]
            similarity_logits = similarity_logits.unsqueeze(-1)
            # Broadcast for K: [B, M, M, 1]
            logdisp_pair = logdisp_pair - similarity_logits
            out["loss_pose_simlog_mean"] = similarity_logits.detach().mean()
            out["loss_pose_simlog_std"] = similarity_logits.detach().std()

        out["disp_dT"] = torch.exp(logdisp_pair)  # [B, M, M, K]
        return out


class PoseLoss(nn.Module):
    """
    Pose loss with optional heteroscedastic NLL.
    criterion: 'mse' → Gaussian, 'l1' → Laplace-style.
    uncertainty_mode must match the head’s.
    """

    def __init__(
        self,
        criterion: str = "l1",
        alpha_t: float = 0.5,
        alpha_s: float = 0.75,
        alpha_ts: float = 0.5,
        uncertainty_mode: Literal["none", "additive", "correlated", "correlated_proj"] = "none",
    ):
        super().__init__()
        assert criterion in ("mse", "l1")
        assert uncertainty_mode in ("none", "additive", "correlated", "correlated_proj")
        self.criterion_str = criterion
        self.error_fn = torch.square if criterion == "mse" else torch.abs
        # scalar in front of NLL: ½ for Gaussian, 1 for Laplace
        self.nll_a = 0.5 if criterion == "mse" else 1.0
        # optional const term for true NLL (can be dropped)
        self.nll_c = math.log(2 * math.pi) if criterion == "mse" else math.log(2.0)

        self.alpha_t = alpha_t
        self.alpha_s = alpha_s
        self.alpha_ts = alpha_ts
        self.uncertainty_mode = uncertainty_mode

    def forward(
        self,
        pred: dict[str, Tensor],
        gt_dT: Tensor,
        Ms: list[int],
    ) -> dict[str, Tensor]:
        """
        pred: (mu_dT, logdisp_dT or None)
        gt_dT: [B, M, M, K]
        Ms: list of per-view token counts
        """
        B = gt_dT.shape[0]
        mu_dT = pred["pred_dT"]

        # 1) elementwise error or NLL
        nll = self.error_fn(mu_dT - gt_dT)  # [B,M,M,K]

        if self.uncertainty_mode != "none":
            disp_pair = pred["disp_dT"]
            logdisp = torch.log(disp_pair)
            nll = self.nll_a * (nll / disp_pair + logdisp + self.nll_c)
            # set diagonal to 0
            nll.diagonal(dim1=1, dim2=2).fill_(0.0)

        # 2) decompose into intra/inter & spatial/temporal exactly as before
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
