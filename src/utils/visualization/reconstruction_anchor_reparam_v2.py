import torch
import math
import torch.nn.functional as F

def ls_solver_fixed_anchor_reparam_vectorized(
    diffs: torch.Tensor, 
    weights: torch.Tensor, 
    M: int, 
    anchor_index: int, 
    anchor_value: torch.Tensor
) -> torch.Tensor:
    """
    Vectorized version of LS solver for x in R^(M x 2) from pairwise differences.
    Fixes x[anchor_index] = anchor_value and solves for the rest.
    
    Args:
        diffs: Tensor of shape [M, M, 2] with predicted differences (x[b] - x[a]).
        weights: Tensor of shape [M, M] with weights for each pair.
        M: Total number of patches.
        anchor_index: The index to fix.
        anchor_value: Tensor of shape [2] for the fixed value at anchor_index.
    
    Returns:
        x: Tensor of shape [M, 2] with x[anchor_index] exactly equal to anchor_value.
    """
    device = diffs.device
    # Get free indices (all indices except the anchor)
    free_idx = torch.tensor([i for i in range(M) if i != anchor_index], device=device)
    num_free = free_idx.numel()  # M - 1

    # Unknown vector y is of shape [num_free, 2] flattened to [2*num_free]
    # We'll build three sets of equations in a vectorized fashion.

    # === Free-Free Equations ===
    # For free indices, let F = free_idx. We want all pairs (a, b) with a,b in F and a != b.
    F = free_idx
    I, J = torch.meshgrid(F, F, indexing='ij')  # both shape [num_free, num_free]
    mask = I != J
    I_vec = I[mask]  # global index for a
    J_vec = J[mask]  # global index for b
    # Number of free-free equations per dimension:
    N_ff = I_vec.numel()  
    # Weight factors for these equations:
    w_ff = torch.sqrt(weights[I_vec, J_vec])  # shape [N_ff]

    # To map free global indices to relative positions, we use the fact that F is sorted.
    # For each equation, we need the relative index of I_vec and J_vec:
    # (Since F was built from range(M) excluding anchor, we can use searchsorted.)
    rel_a = torch.searchsorted(F, I_vec)
    rel_b = torch.searchsorted(F, J_vec)

    # Build A_ff for x-dim equations: for each free-free pair, row = e_{rel_b} - e_{rel_a}.
    A_ff_x = torch.zeros((N_ff, 2 * num_free), device=device)
    A_ff_x[torch.arange(N_ff), 2 * rel_b] = 1.0
    A_ff_x[torch.arange(N_ff), 2 * rel_a] = -1.0
    b_ff_x = diffs[I_vec, J_vec, 0]
    A_ff_x = A_ff_x * w_ff.unsqueeze(1)
    b_ff_x = b_ff_x * w_ff

    # And similarly for y-dim:
    A_ff_y = torch.zeros((N_ff, 2 * num_free), device=device)
    A_ff_y[torch.arange(N_ff), 2 * rel_b + 1] = 1.0
    A_ff_y[torch.arange(N_ff), 2 * rel_a + 1] = -1.0
    b_ff_y = diffs[I_vec, J_vec, 1]
    A_ff_y = A_ff_y * w_ff.unsqueeze(1)
    b_ff_y = b_ff_y * w_ff

    # === Anchor-Free Equations ===
    # For a = anchor and b in free:
    F_rep = F  # free indices for b
    w_af = torch.sqrt(weights[anchor_index, F_rep])  # shape [num_free]
    A_af_x = torch.zeros((num_free, 2 * num_free), device=device)
    A_af_x[torch.arange(num_free), 2 * torch.arange(num_free)] = 1.0
    b_af_x = diffs[anchor_index, F_rep, 0] + anchor_value[0]
    A_af_x = A_af_x * w_af.unsqueeze(1)
    b_af_x = b_af_x * w_af

    A_af_y = torch.zeros((num_free, 2 * num_free), device=device)
    A_af_y[torch.arange(num_free), 2 * torch.arange(num_free) + 1] = 1.0
    b_af_y = diffs[anchor_index, F_rep, 1] + anchor_value[1]
    A_af_y = A_af_y * w_af.unsqueeze(1)
    b_af_y = b_af_y * w_af

    # === Free-Anchor Equations ===
    # For a in free and b = anchor.
    w_fa = torch.sqrt(weights[F_rep, anchor_index])  # shape [num_free]
    A_fa_x = torch.zeros((num_free, 2 * num_free), device=device)
    A_fa_x[torch.arange(num_free), 2 * torch.arange(num_free)] = -1.0
    b_fa_x = diffs[F_rep, anchor_index, 0] - anchor_value[0]
    A_fa_x = A_fa_x * w_fa.unsqueeze(1)
    b_fa_x = b_fa_x * w_fa

    A_fa_y = torch.zeros((num_free, 2 * num_free), device=device)
    A_fa_y[torch.arange(num_free), 2 * torch.arange(num_free) + 1] = -1.0
    b_fa_y = diffs[F_rep, anchor_index, 1] - anchor_value[1]
    A_fa_y = A_fa_y * w_fa.unsqueeze(1)
    b_fa_y = b_fa_y * w_fa

    # === Stack All Equations ===
    A_total = torch.cat([A_ff_x, A_ff_y, A_af_x, A_af_y, A_fa_x, A_fa_y], dim=0)
    b_total = torch.cat([b_ff_x, b_ff_y, b_af_x, b_af_y, b_fa_x, b_fa_y], dim=0)

    # Solve the least-squares system
    y_sol = torch.linalg.lstsq(A_total, b_total).solution
    y_sol = y_sol.view(num_free, 2)

    # Reconstruct the full solution: assign free values and set the anchor exactly.
    x = torch.empty((M, 2), device=device)
    x[free_idx] = y_sol
    x[anchor_index] = anchor_value
    return x

def reconstruction_from_gt_dT_with_anchor(
    image_crops: torch.Tensor,      # [V, C, H, W] augmented crop images.
    patch_positions: torch.Tensor,  # [V, N, 2] patch positions in crop coordinates.
    patch_size: int,                # e.g., 16.
    img_size: int,                  # crop size, e.g., 224.
    canonical_img_size: int,        # e.g., 512.
    crop_params: torch.Tensor,      # [V, 4] each row is [y, x, h, w] in canonical space.
    gt_dT: torch.Tensor,            # [V, V, N, N, 4] ground-truth normalized differences.
    max_scale_ratio: float          # e.g., 4.0.
) -> tuple:
    """
    Reconstruct a canonical image using ground-truth differences (gt_dT) by solving for absolute
    translations and log-scale values via a reparameterized LS system. The anchor is fixed to the
    first patch of view 0, whose canonical value is computed from the crop parameters.
    
    The gt differences are assumed normalized:
      - Translations divided by canonical_img_size.
      - Scales divided by log(max_scale_ratio).
    We undo these normalizations when building the LS system.
    
    Returns:
        A tuple of:
         - reconstructed_img: [C, canonical_img_size, canonical_img_size] image.
         - translations: List (length V) of tensors [N, 2] for each view.
         - log_scales: List (length V) of tensors [N, 2] for each view.
    """
    V, C, H, W = image_crops.shape
    _, N, _ = patch_positions.shape
    device = image_crops.device
    M = V * N  # total number of patches

    # Reshape gt_dT: from [V, V, N, N, 4] to global [M, M, 4]
    gt_dT_global = gt_dT.permute(0, 2, 1, 3, 4).reshape(M, M, 4)
    # Undo normalization:
    # Translations were divided by canonical_img_size,
    # Scales by log(max_scale_ratio)
    dT_global = gt_dT_global[:, :, :2] * canonical_img_size
    dS_global = gt_dT_global[:, :, 2:] * math.log(max_scale_ratio)

    # Use uniform weights.
    weight_global = torch.ones((M, M), device=device)
    
    # Choose the anchor: first patch of view 0 (global index 0).
    # Compute its canonical absolute translation:
    # T_anchor = crop_params[0][:2] + (patch_positions[0,0] / img_size) * crop_params[0][2:4]
    T_anchor = crop_params[0, :2] + (patch_positions[0, 0] / img_size) * crop_params[0, 2:4]
    # Compute its canonical log-scale.
    # Here S_anchor is defined as log(patch_size * scale) relative to the crop.
    S_anchor = torch.log((patch_size * crop_params[0, 2:4] / img_size))
    
    # Solve the LS systems using the optimized (vectorized) solver.
    T_global = ls_solver_fixed_anchor_reparam_vectorized(
        dT_global, weight_global, M, anchor_index=0, anchor_value=T_anchor
    )
    S_global = ls_solver_fixed_anchor_reparam_vectorized(
        dS_global, weight_global, M, anchor_index=0, anchor_value=S_anchor
    )
    
    # Reshape global solutions into per-view lists.
    translations = [T_global[v * N : (v + 1) * N] for v in range(V)]
    log_scales = [S_global[v * N : (v + 1) * N] for v in range(V)]
    
    # === Reconstruct the Canonical Image ===
    # Instead of a nested loop over views and patches, flatten to a single loop.
    canvas = torch.zeros((C, canonical_img_size, canonical_img_size), device=device)
    count_map = torch.zeros((1, canonical_img_size, canonical_img_size), device=device)
    
    # Flatten global translations and log_scales.
    T_global_flat = T_global  # [M, 2]
    S_global_flat = S_global  # [M, 2]
    
    for idx in range(M):
        # Determine view and patch index.
        v = idx // N
        # Extract patch position from patch_positions (for view v and patch idx_in_view)
        i = idx % N
        y_crop, x_crop = patch_positions[v, i]
        y_crop = int(round(y_crop.item()))
        x_crop = int(round(x_crop.item()))
        y_crop = max(0, min(H - patch_size, y_crop))
        x_crop = max(0, min(W - patch_size, x_crop))
        patch = image_crops[v][:, y_crop:y_crop+patch_size, x_crop:x_crop+patch_size].unsqueeze(0)
        
        # Compute target patch dimensions.
        # Here, S_global_flat[idx] is a log-scale; convert via exp.
        scale_factor = torch.exp(S_global_flat[idx])
        new_patch_h = int(round(scale_factor[0].item()))
        new_patch_w = int(round(scale_factor[1].item()))
        # Resize patch.
        patch_resized = F.interpolate(patch, size=(new_patch_h, new_patch_w),
                                      mode='bilinear', align_corners=False).squeeze(0)
        
        # Place the patch using its canonical translation.
        T_patch = T_global_flat[idx]
        y_can = int(round(T_patch[0].item()))
        x_can = int(round(T_patch[1].item()))
        y_can = max(0, min(canonical_img_size - new_patch_h, y_can))
        x_can = max(0, min(canonical_img_size - new_patch_w, x_can))
        
        canvas[:, y_can:y_can+new_patch_h, x_can:x_can+new_patch_w] += patch_resized
        count_map[:, y_can:y_can+new_patch_h, x_can:x_can+new_patch_w] += 1

    count_map[count_map == 0] = 1
    reconstructed_img = canvas / count_map

    return reconstructed_img, translations, log_scales
