import torch
from torch import nn, Tensor

class PatchStressLoss(nn.Module):
    """
    Top-k tempered stress loss for patch embeddings. Only the k
    nearest‐neighbours (by ground-truth distance) for each patch i
    contribute to the loss.

    Loss formula (for each batch b, patch i, neighbour j in top-k):
        dist_ij = sqrt(2 - 2 * (z_i · z_j))                   # spherical distance
        d_gt_ij = ||dt_gt_ij||                                 # ground-truth pixel distance
        w_ij    = exp( -0.5 * (dist_ij / beta)^2 )              # tempered weight
        L = (1 / (B * T * k)) * ∑_{b,i,j∈top-k(i)} w_ij * (dist_ij - d_gt_ij)^2

    where top-k(i) are the k smallest d_gt_ij values for each anchor i.

    Usage:
        loss_fn = PatchTopKTemperedStressLoss(beta=0.02, k=32)
        loss    = loss_fn(z, dt_gt)
    """

    def __init__(self, beta: float = 0.2, k: int = 8, eps: float = 1e-12):
        """
        :param beta:  same β as the Hummingbird soft-max decoder (usually 0.02)
        :param k:     number of nearest‐neighbours to consider per patch
        :param eps:   small constant to avoid sqrt(0) and division by zero
        """
        super().__init__()
        self.beta = beta
        self.k = k
        self.eps = eps

    def forward(self, z: Tensor, dt_gt: Tensor) -> Tensor:
        """
        :param z:     [B, T, D]    L2-normalized patch embeddings
        :param dt_gt: [B, T, T, 4] ground-truth pose deltas (dy, dx, dlogh, dlogw)
                        – each element is already normalized to roughly [-1, +1]
                        – so d_gt_ij = ||dt_gt[b,i,j]|| is ~pixel distance / canonical_size
        :returns:     scalar top-k tempered stress loss
        """
        B, T, D = z.shape

        # 1) Compute pairwise spherical distances:
        #    sim[b,i,j] = z[b,i] · z[b,j]   (cosine similarity, since z is L2‐normed)
        sim = torch.bmm(z, z.transpose(1, 2))  # [B, T, T]

        #    spherical distance: d_ij = sqrt(2 − 2·sim_ij)
        dist = (2.0 - 2.0 * sim).clamp(min=self.eps).sqrt()  # [B, T, T]

        # 2) Ground-truth pixel distance:
        #    d_gt_ij = ||dt_gt[b,i,j]||   (norm over the 4D pose delta)
        d_gt = dt_gt.norm(dim=-1).clamp(min=self.eps)  # [B, T, T]

        # 3) Find the k smallest ground-truth distances for each anchor:
        #    torch.topk(d_gt, k, largest=False) returns the values/indices of the k smallest d_gt[b, i, *].
        #    idx[b, i, :] are the indices j of the k nearest neighbours (in pixel-space) for patch i.
        _, idx = torch.topk(d_gt, self.k, dim=2, largest=False)  # idx: [B, T, k]

        # 4) Build a boolean mask M[b,i,j] = 1 if j ∈ top-k(i), else 0
        mask = torch.zeros_like(d_gt)  # [B, T, T]
        mask.scatter_(2, idx, 1.0)  # set mask[b,i, idx[b,i,m]] = 1 for m in [0..k−1]

        # 5) Compute the tempered weights w_ij = exp[−½ (dist_ij / β)^2]
        w = torch.exp(-0.5 * (dist / self.beta).pow(2))  # [B, T, T]

        # 6) Weighted squared error on those top-k pairs:
        se = (dist - d_gt).pow(2)  # [B, T, T]
        loss = (w * se * mask).sum() / (B * T * self.k)

        return loss