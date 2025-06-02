import torch
from torch import nn, Tensor

class PatchSmoothnessLoss(nn.Module):
    """
    Graph-Laplacian smoothness, with per-node normalization and optimized energy computation.
    DEPRECATED: using PatchStressLoss instead.

    params:
        sigma_yx: bandwidth for spatial offsets (dy, dx)
        sigma_hw: bandwidth for log-scale offsets (dlogh, dlogw)
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
    def _laplacian_energy_per_node(z: Tensor, w: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute per-node Laplacian energy and degree without forming full distance matrix.
        Equivalent to:  ∑_ij w_ij ||z_i - z_j||^2

        :param z: [B, T, D] L2-normalized patch embeddings
        :param w: [B, T, T] affinity matrix
        :return: (e, d) where
                 e: [B, T] per-node energy ∑_j w_ij ||z_i - z_j||^2,
                 d: [B, T] per-node degree ∑_j w_ij
        """
        # degree per node
        d = w.sum(dim=-1).clamp(min=1e-6)  # [B, T]
        # squared norms of embeddings
        norms = (z * z).sum(dim=-1)  # [B, T]
        # affinity-weighted norm sum: ∑_j w_ij * ||z_j||^2
        w_norms = torch.bmm(w, norms.unsqueeze(-1)).squeeze(-1)  # [B, T]
        # affinity-weighted embedding sum: ∑_j w_ij * z_j
        wz = torch.bmm(w, z)  # [B, T, D]

        # energy: ||z_i||^2 * d_i + ∑_j w_ij ||z_j||^2 - 2 <z_i, ∑_j w_ij z_j>
        e = norms * d + w_norms - 2.0 * (wz * z).sum(dim=-1)  # [B, T]
        return e, d

    def forward(
        self,
        z: Tensor,  # [B, T, D] patch embeddings
        gt_dT: Tensor,  # [B, T, T, 4] normalized [dy, dx, dlogh, dlogw]
    ) -> Tensor:
        """
        :param z:    Patch embeddings (already L2-normalized in projector)
        :param gt_dT: Ground-truth pose deltas
        :returns:     Scalar smoothness loss (per-node normalized)
        """
        # 1) compute affinity matrix w_ij from pose deltas
        dist2_pose = (gt_dT.pow(2) / (2 * self.sigma2)).sum(dim=-1)  # [B, T, T]
        w = torch.exp(-dist2_pose) + 1e-6  # [B, T, T]

        # 2) compute per-node energy and degree via optimized routine
        e, d = self._laplacian_energy_per_node(z, w)  # [B, T], [B, T]

        # 3) per-node Laplacian then average
        lap = (e / d).mean()  # scalar
        return lap