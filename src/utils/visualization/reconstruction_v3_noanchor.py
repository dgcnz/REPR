import torch
import torch.nn.functional as F
import math


def reconstruct_4d_poses_centered_torch_vectorized(
    pred_dT: torch.Tensor, 
    lambda_center: float = 0.1,
    target_scale_ratio: float = 1.0,
    max_scale_ratio: float = 4.0,
) -> torch.Tensor:
    """
    Recover absolute 4D patch poses (translation and log-scale) from predicted pairwise differences.
    Optionally, the centering constraints are weighted by lambda_center.

    Args:
        pred_dT: Tensor of shape [M, M, 4] where ideally pred_dT[i, j] ~ pose[j] - pose[i].
        lambda_center: Weight for the centering constraints.
        target_scale_ratio: Desired average scale ratio (default 1.0).
        max_scale_ratio: Maximum scale ratio used for normalization.
        
    Returns:
        poses: Tensor of shape [M, 4] with recovered poses.
    """
    M = pred_dT.shape[0]
    device = pred_dT.device

    # Build indices for all (i, j) pairs (excluding i == j)
    idx = torch.arange(M, device=device)
    i_idx, j_idx = torch.meshgrid(idx, idx, indexing="ij")
    i_idx = i_idx.reshape(-1)
    j_idx = j_idx.reshape(-1)
    valid_mask = i_idx != j_idx
    i_idx = i_idx[valid_mask]
    j_idx = j_idx[valid_mask]
    num_constraints = i_idx.shape[0]

    total_rows = 4 * num_constraints  # 4 equations per pair
    total_cols = 4 * M

    # Construct system matrix A
    A = torch.zeros((total_rows, total_cols), device=device)
    dims = torch.arange(4, device=device).repeat(num_constraints)
    i_idx_rep = i_idx.repeat_interleave(4)
    j_idx_rep = j_idx.repeat_interleave(4)
    row_idx = torch.arange(total_rows, device=device)
    neg_cols = 4 * i_idx_rep + dims
    pos_cols = 4 * j_idx_rep + dims
    A[row_idx, neg_cols] = -1.0
    A[row_idx, pos_cols] = 1.0

    # Build right-hand side vector b
    pred_dT_flat = pred_dT.reshape(-1, 4)  # [M*M, 4]
    b = pred_dT_flat[valid_mask].reshape(-1)

    # Construct centering constraint matrix: for each pose dimension d,
    # enforce sum_i pose[i,d] = target.
    # For translation dims (0,1) target = 0.5 * M.
    # For scale dims (2,3), target = log(target_scale_ratio)/log(max_scale_ratio)
    centering_A = torch.zeros((4, total_cols), device=device)
    for d in range(4):
        centering_A[d, d:total_cols:4] = 1.0

    b_center = torch.zeros(4, device=device)
    b_center[0] = 0.5 * M  # Translation dimension 0.
    b_center[1] = 0.5 * M  # Translation dimension 1.
    raw_target = math.log(target_scale_ratio) / math.log(max_scale_ratio) if max_scale_ratio != 1 else 0.0
    b_center[2] = raw_target  # Scale dimension 2.
    b_center[3] = raw_target  # Scale dimension 3.

    # Combine constraints with centering weight.
    A_full = torch.cat([A, lambda_center * centering_A], dim=0)
    b_full = torch.cat([b, lambda_center * b_center], dim=0)

    solution = torch.linalg.lstsq(A_full, b_full).solution
    poses = solution.view(M, 4)
    return poses



def reconstruction_lstsq(
    patch_positions_nopos: torch.Tensor,  # [V, N_vis, 2] in crop coordinates
    pred_dT: torch.Tensor,  # [V, V, N_mask, N_mask, 4]
    original_images: torch.Tensor,  # [V, C, H, W] augmented crop images
    patch_size: int,  # e.g., 16
    img_size: int,  # e.g., 224 (crop size)
    canonical_img_size: int = 512,  # size of canonical canvas
    max_scale_ratio: float = 4.0,
    target_scale_ratio: float = 1.0,
) -> torch.Tensor:
    """
    Reconstruct a canonical image from predicted differences.

    Assumes that the recovered translation components are normalized in (0, 1),
    where 0 maps to 0 and 1 maps to canonical_img_size. For scales, the LS system
    enforces an average scale ratio given by target_scale_ratio (via its log-space).
    The log-scale components are recovered by multiplying by math.log(max_scale_ratio).

    Args:
        patch_positions_nopos: [V, N_vis, 2] ground-truth patch positions in crop coordinates.
        pred_dT: [V, V, N_mask, N_mask, 4] predicted pairwise differences.
        original_images: [V, C, H, W] augmented crop images.
        patch_size: Patch size in the crop.
        img_size: Crop size.
        canonical_img_size: Size of canonical canvas.
        max_scale_ratio: Maximum scale ratio used during training.
        target_scale_ratio: Desired average scale ratio (default 1.0 means no bias).
        
    Returns:
        Reconstructed canonical image: [C, canonical_img_size, canonical_img_size].
    """
    device = original_images.device
    V, C, H, W = original_images.shape
    N_mask = patch_positions_nopos.shape[1]
    M = V * N_mask

    patch_positions_nopos = patch_positions_nopos.view(-1, 2)

    # --- Recover absolute 4D poses from predicted differences ---
    pred_dT_flat = pred_dT.permute(0, 2, 1, 3, 4).reshape(M, M, 4)
    refined_poses = reconstruct_4d_poses_centered_torch_vectorized(
        pred_dT_flat, 
        lambda_center=1, 
        target_scale_ratio=target_scale_ratio,
        max_scale_ratio=max_scale_ratio,
    )  # [M, 4]

    # --- Map recovered translations ---
    T_norm = refined_poses[:, :2]
    refined_trans_canonical = T_norm * canonical_img_size

    # --- Process scales ---
    refined_log_scale = refined_poses[:, 2:] * math.log(max_scale_ratio)

    # Final recovered token pose.
    refined_final = torch.cat(
        [refined_trans_canonical, refined_log_scale], dim=1
    )  # [M, 4]

    # --- Build token info ---
    token_infos = []
    idx = 0
    for v in range(V):
        for _ in range(N_mask):
            token_infos.append((v, refined_final[idx], patch_positions_nopos[idx]))
            idx += 1

    # --- Create canonical canvas ---
    canvas = torch.zeros((C, canonical_img_size, canonical_img_size), device=device)
    count_map = torch.zeros((1, canonical_img_size, canonical_img_size), device=device)

    # --- Paste patches ---
    for info in token_infos:
        v, token_pose, pos_in_crop = info
        final_translation = token_pose[:2]
        scale_factor_y = torch.exp(token_pose[2])
        scale_factor_x = torch.exp(token_pose[3])
        new_patch_h = int(torch.ceil(patch_size * scale_factor_y))
        new_patch_w = int(torch.ceil(patch_size * scale_factor_x))
        if new_patch_h < 1 or new_patch_w < 1:
            continue

        # Extract patch from the crop.
        orig_img = original_images[v]
        y_crop, x_crop = int(round(pos_in_crop[0].item())), int(round(pos_in_crop[1].item()))
        y_crop = max(0, min(H - patch_size, y_crop))
        x_crop = max(0, min(W - patch_size, x_crop))
        patch = orig_img[:, y_crop:y_crop+patch_size, x_crop:x_crop+patch_size].unsqueeze(0)
        patch_resized = F.interpolate(
            patch,
            size=(new_patch_h, new_patch_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        # Place patch on canonical canvas.
        y_can = int(round(final_translation[0].item()))
        x_can = int(round(final_translation[1].item()))
        y_can = max(0, min(canonical_img_size - new_patch_h, y_can))
        x_can = max(0, min(canonical_img_size - new_patch_w, x_can))
        canvas[:, y_can:y_can+new_patch_h, x_can:x_can+new_patch_w] += patch_resized
        count_map[:, y_can:y_can+new_patch_h, x_can:x_can+new_patch_w] += 1
    count_map[count_map == 0] = 1
    reconstructed_img = canvas / count_map
    return reconstructed_img
