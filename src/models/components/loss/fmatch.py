import torch
from torch import nn
from jaxtyping import Float, Tensor

class FeatureMatchSimpleLoss(nn.Module):
    """
    Simplified VicRegL-style feature match loss (invariance only),
    with a static top-γ selection using per-image similarity (O(B·P²·D)).

    view_ids: Tensor of shape [P] indicating view index per patch.
    """
    def __init__(self, gamma: int = 20, lambda_inv: float = 25.0):
        super().__init__()
        assert gamma > 0, "gamma must be positive"
        self.gamma = gamma
        self.lambda_inv = lambda_inv

    def forward(self, z: Float[Tensor, "B P D"], view_ids: Tensor) -> dict[str, Tensor]:
        B, P, D = z.shape
        device = z.device

        # Compute per-image patch similarities [B, P, P]
        sim = torch.bmm(z, z.transpose(1, 2))  # cosine since z is l2-norm

        # Mask out same-view pairs and self-pairs
        # view_ids: [P]
        mask_view = view_ids.unsqueeze(0) != view_ids.unsqueeze(1)  # [P, P]
        eye = torch.eye(P, device=device, dtype=torch.bool)
        mask = mask_view & ~eye                         # [P, P]
        sim.masked_fill_(~mask.unsqueeze(0), -1.0)

        # For each patch, find best neighbour index and similarity
        # best_sim: [B, P], best_j: [B, P]
        best_sim, best_j = sim.max(dim=2)

        # Flatten across batch: total N = B*P anchors
        z_flat = z.view(-1, D)  # [N, D]
        anchors = torch.arange(B * P, device=device)

        # Compute flattened best_j indices (shift per image)
        offsets = torch.arange(B, device=device).unsqueeze(1) * P  # [B,1]
        best_j_flat = (best_j + offsets).view(-1)                # [N]
        best_sim_flat = best_sim.view(-1)                        # [N]

        # Static top-γ selection
        topk_vals, topk_idx = torch.topk(best_sim_flat, self.gamma, largest=True, sorted=False)
        sel_anchors = anchors[topk_idx]                          # [γ]
        sel_js = best_j_flat[topk_idx]                           # [γ]

        # Invariance loss: MSE between matched features
        z1 = z_flat[sel_anchors]                                  # [γ, D]
        z2 = z_flat[sel_js]                                       # [γ, D]
        loss = self.lambda_inv * (z1 - z2).pow(2).mean()

        return {
            "loss_fmatch": loss,
            "fmatch_avg_cos": topk_vals.mean().detach(),
        }
