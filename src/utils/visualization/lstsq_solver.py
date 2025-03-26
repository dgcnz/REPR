import torch
from jaxtyping import Float
from torch import Tensor

def lstsq_dT_solver(
    dT: Float[Tensor, "M M 2"],   # [M, M, 2] with predicted differences (x[i] - x[j])
    weights: Float[Tensor, "M M"], # [M, M] weight for each pair
    anchor_index: int,
    anchor_value: Float[Tensor, "2"]  # [2]
) -> torch.Tensor:
    """
    Vectorized LS solver for x in R^(M x 2) from pairwise differences.
    In this version, diffs are assumed to be of the form:
         diffs[a, b] = x[a] - x[b].
    We require equations in the form:
         x[b] - x[a] = -diffs[a, b],
    and we fix x[anchor_index] = anchor_value.

    Args:
        diffs: Tensor of shape [M, M, 2] with predicted differences (x[i] - x[j]).
        weights: Tensor of shape [M, M] with weights for each pair.
        M: Total number of patches.
        anchor_index: The index of the patch to fix.
        anchor_value: Tensor of shape [2] giving the fixed value at anchor_index.

    Returns:
        x: Tensor of shape [M, 2] with x[anchor_index] exactly equal to anchor_value.
    """
    M = dT.shape[0]
    device = dT.device
    # Determine free indices (all indices except the anchor).
    free_idx = torch.tensor([i for i in range(M) if i != anchor_index], device=device)
    num_free = free_idx.numel()  # M - 1

    # === Build LS system ===

    # --- Free-Free Equations ---
    # For free indices a and b, we want:
    #   x[b] - x[a] = - (x[a] - x[b]) = -diffs[a, b].
    F = free_idx
    I, J = torch.meshgrid(F, F, indexing='ij')  # each of shape [num_free, num_free]
    mask = I != J
    I_vec = I[mask]  # global index for a
    J_vec = J[mask]  # global index for b
    N_ff = I_vec.numel()
    w_ff = torch.sqrt(weights[I_vec, J_vec])  # [N_ff]
    rel_a = torch.searchsorted(F, I_vec)
    rel_b = torch.searchsorted(F, J_vec)

    # Build design matrices for x-dimension.
    A_ff_x = torch.zeros((N_ff, 2 * num_free), device=device)
    A_ff_x[torch.arange(N_ff), 2 * rel_b] = 1.0
    A_ff_x[torch.arange(N_ff), 2 * rel_a] = -1.0
    # Since diffs are x[a]-x[b], we set the right-hand side to be -diffs.
    b_ff_x = - dT[I_vec, J_vec, 0]
    A_ff_x = A_ff_x * w_ff.unsqueeze(1)
    b_ff_x = b_ff_x * w_ff

    A_ff_y = torch.zeros((N_ff, 2 * num_free), device=device)
    A_ff_y[torch.arange(N_ff), 2 * rel_b + 1] = 1.0
    A_ff_y[torch.arange(N_ff), 2 * rel_a + 1] = -1.0
    b_ff_y = - dT[I_vec, J_vec, 1]
    A_ff_y = A_ff_y * w_ff.unsqueeze(1)
    b_ff_y = b_ff_y * w_ff

    # --- Anchor-Free Equations ---
    # For a = anchor and b in free, we want:
    #   x[b] - x[anchor] = - (x[anchor] - x[b]) = -diffs[anchor, b].
    F_rep = F
    w_af = torch.sqrt(weights[anchor_index, F_rep])
    A_af_x = torch.zeros((num_free, 2 * num_free), device=device)
    A_af_x[torch.arange(num_free), 2 * torch.arange(num_free)] = 1.0
    b_af_x = - dT[anchor_index, F_rep, 0] + anchor_value[0]
    A_af_x = A_af_x * w_af.unsqueeze(1)
    b_af_x = b_af_x * w_af

    A_af_y = torch.zeros((num_free, 2 * num_free), device=device)
    A_af_y[torch.arange(num_free), 2 * torch.arange(num_free) + 1] = 1.0
    b_af_y = - dT[anchor_index, F_rep, 1] + anchor_value[1]
    A_af_y = A_af_y * w_af.unsqueeze(1)
    b_af_y = b_af_y * w_af

    # --- Free-Anchor Equations ---
    # For a in free and b = anchor, we want:
    #   x[anchor] - x[a] = - (x[a] - x[anchor]) = -diffs[a, anchor].
    w_fa = torch.sqrt(weights[F_rep, anchor_index])
    A_fa_x = torch.zeros((num_free, 2 * num_free), device=device)
    A_fa_x[torch.arange(num_free), 2 * torch.arange(num_free)] = -1.0
    b_fa_x = - dT[F_rep, anchor_index, 0] - anchor_value[0]
    A_fa_x = A_fa_x * w_fa.unsqueeze(1)
    b_fa_x = b_fa_x * w_fa

    A_fa_y = torch.zeros((num_free, 2 * num_free), device=device)
    A_fa_y[torch.arange(num_free), 2 * torch.arange(num_free) + 1] = -1.0
    b_fa_y = - dT[F_rep, anchor_index, 1] - anchor_value[1]
    A_fa_y = A_fa_y * w_fa.unsqueeze(1)
    b_fa_y = b_fa_y * w_fa

    # === Stack all equations ===
    A_total = torch.cat([A_ff_x, A_ff_y, A_af_x, A_af_y, A_fa_x, A_fa_y], dim=0)
    b_total = torch.cat([b_ff_x, b_ff_y, b_af_x, b_af_y, b_fa_x, b_fa_y], dim=0)

    # Solve the LS system.
    y_sol = torch.linalg.lstsq(A_total, b_total).solution
    y_sol = y_sol.view(num_free, 2)

    # Reconstruct full solution: assign solved values and fix the anchor.
    x = torch.empty((M, 2), device=device)
    x[free_idx] = y_sol
    x[anchor_index] = anchor_value
    return x
