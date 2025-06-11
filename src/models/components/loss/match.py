import torch


class PatchMatchingLoss(torch.nn.Module):
    def __init__(
        self,
        beta_f: float,
        beta_w: float,
        sigma_yx: float,
        sigma_hw: float,
        gN: int,
    ):
        super().__init__()
        if beta_f <= 0:
            raise ValueError("beta_f must be positive.")
        if beta_w < 0:
            raise ValueError("beta_w should be non-negative.")

        self.beta_f = beta_f
        self.beta_w = beta_w
        self.gN = gN
        # since gt_dT[b, i, j, k] is always in [-1, 1] the maximum l2 or l1 magnitude is 4.0
        # assuming sigma's are in [0, 1]
        self.inf = 4.0 + 1e-2
        if not (0 <= sigma_yx <= 1) or not (0 <= sigma_hw <= 1):
            raise ValueError(
                "sigma_yx and sigma_hw must be between 0 and 1 (inclusive)."
            )

        sigma = torch.tensor([sigma_yx, sigma_yx, sigma_hw, sigma_hw]).float()
        self.register_buffer("sigma", sigma, persistent=False)

    def forward(
        self,
        z: torch.Tensor,  # Shape: [B, N, D]
        gt_dT: torch.Tensor,  # Shape: [B, N, N, 4]
    ) -> torch.Tensor:
        """
        Calculates loss by finding one global best non-self match for each anchor.
        :param z: l2-normalized features from all views, shape [B, gV*gN + lV * lN, D]
        """
        B, N, D = z.shape

        # --- 1. Extract Anchor Features ---
        anchor_feat = z[:, : self.gN, :]

        # --- 2. Calculate Geometric Costs from anchors to ALL patches ---
        dist_from_anchor = gt_dT[:, : self.gN, :, :].pow(2).mul(self.sigma).sum(dim=-1)
        # Shape: [B, gN, N]

        # --- 3. Mask Intra-Anchor View Costs (set to infinity) ---
        # For each anchor, we don't want it to match with another patch from the *same* anchor view.
        dist_from_anchor[:, :, 0 : self.gN] = self.inf

        # --- 4. Find Global Best Match (j*) and its cost for each anchor ---
        # For each anchor (dim 1), find the min cost across all N (dim 2)
        min_dist, min_idx = dist_from_anchor.min(dim=2)

        # --- 5. Calculate Weights based on minimum geometric costs ---
        weights = torch.exp(-self.beta_w * min_dist)

        # --- 6. Gather Positive Features from all_views using global indices ---
        idx_to_gather = min_idx.unsqueeze(-1).expand(-1, -1, D)
        # Shape: [B, gN, D]

        positive_feat = torch.gather(z, dim=1, index=idx_to_gather)
        # Shape: [B, gN, D]

        # --- 7. Calculate Feature Similarity Loss for (anchor, gathered_positive) ---
        cos_sim = (anchor_feat * positive_feat).sum(dim=-1).clamp(-1.0, 1.0)
        # Shape: [B, gN]

        # --- 8. Weighted Loss Terms ---
        exp_cos_sim = self.beta_f * torch.log1p(torch.exp(-cos_sim / self.beta_f))
        weighted_feature_losses = weights * exp_cos_sim
        # Shape: [B, gN]

        # --- 9. Final Loss: Average over all B*gN terms ---
        # This effectively averages over all anchors in the batch.
        final_loss = weighted_feature_losses.mean()
        return {
            "loss_pmatch": final_loss,
            "loss_pmatch_cos_sim": cos_sim.detach().mean(),
            "loss_pmatch_mindist_max": min_dist.detach().max(),
            "loss_pmatch_mindist_mean": min_dist.detach().mean(),
        }


if __name__ == "__main__":
    # Example usage
    B, N, D = 1, 4, 1
    C = 1
    beta = 0.1
    beta_w = 0.5
    gN = 1

    features_all_views = torch.randn(B, N, D)
    gt_dT_full = torch.randn(B, N, N, C)

    loss_fn = PatchMatchingLoss(beta, beta_w, gN)
    print(gt_dT_full[0].shape)
    print(gt_dT_full[0].squeeze(-1))
    loss = loss_fn(features_all_views, gt_dT_full)
    print(gt_dT_full[0].squeeze(-1))
    print("Loss:", loss.item())
