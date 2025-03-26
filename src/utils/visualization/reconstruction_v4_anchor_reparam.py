import torch
import math
import torch.nn.functional as F
from src.utils.visualization.lstsq_solver import lstsq_dT_solver
from jaxtyping import Float, Int
from torch import Tensor

def reconstruction_lstsq_with_anchor_reparam(
    g_crops: Float[Tensor, "gV C H W"],             # global augmented crop images
    l_crops: Float[Tensor, "lV C H W"],             # local augmented crop images
    g_patch_positions: Float[Tensor, "gV*N_g 2"],   # patch positions (top-left) in global crop coordinates
    l_patch_positions: Float[Tensor, "lV*N_l 2"],   # patch positions (top-left) in local crop coordinates
    patch_size: int,                                # patch size in the crop (e.g., 16)
    g_shapes: tuple[int, int, int],                 # (global crop size, N_g, gV)
    l_shapes: tuple[int, int, int],                 # (local crop size, N_l, lV)
    canonical_img_size: int,                        # size of the canonical canvas (e.g., 512)
    g_crop_params: Float[Tensor, "gV 4"],           # each row is [y, x, h, w] for global crops in canonical space
    l_crop_params: Float[Tensor, "lV 4"],           # each row is [y, x, h, w] for local crops in canonical space
    pred_dT: Float[Tensor, "M M 4"],                # predicted pairwise differences (translation and log-scale)
    max_scale_ratio: float                          # e.g., 4.0
) -> tuple[
    Float[Tensor, "C canonical_img_size canonical_img_size"],  # reconstructed image
    list[Float[Tensor, "N_g 2"]],                              # global translations
    list[Float[Tensor, "N_g 2"]],                              # global log scales
    list[Float[Tensor, "N_l 2"]],                              # local translations
    list[Float[Tensor, "N_l 2"]]                               # local log scales
]:
    """
    Reconstruct a canonical image using least-squares estimation of absolute translations
    and log-scale values from predicted pairwise differences.
    
    Parameters:
        g_crops: Global augmented crop images with shape [gV, C, H, W]
        l_crops: Local augmented crop images with shape [lV, C, H, W]
        g_patch_positions: Patch positions (top-left) in global crop coordinates with shape [gV*N_g, 2]
        l_patch_positions: Patch positions (top-left) in local crop coordinates with shape [lV*N_l, 2]
        patch_size: Size of each patch in pixels (e.g., 16)
        g_shapes: Tuple containing (global crop size, N_g, gV)
        l_shapes: Tuple containing (local crop size, N_l, lV)
        canonical_img_size: Size of the canonical canvas in pixels (e.g., 512)
        g_crop_params: Crop parameters for global crops in canonical space, each row is [y, x, h, w]
        l_crop_params: Crop parameters for local crops in canonical space, each row is [y, x, h, w]
        pred_dT: Predicted pairwise differences with shape [M, M, 4] where M = M_g + M_l
                 The first 2 channels are translations, next 2 are log-scales
        max_scale_ratio: Maximum scale ratio for normalization (e.g., 4.0)
    
    Notes:
        The predicted differences are assumed normalized:
          - Translations were divided by canonical_img_size
          - Scales were divided by log(max_scale_ratio)
        These normalizations are undone when building the LS system.
        
        An anchor is fixed to the first patch of the first global crop (its canonical
        value computed from the crop parameters) so that the LS system is well posed.
    
    Returns:
        A tuple containing:
          - reconstructed_img: Reconstructed image with shape [C, canonical_img_size, canonical_img_size]
          - g_translations: List of tensors for global view translations, each with shape [N_g, 2]
          - g_log_scales: List of tensors for global view log-scales, each with shape [N_g, 2]
          - l_translations: List of tensors for local view translations, each with shape [N_l, 2]
          - l_log_scales: List of tensors for local view log-scales, each with shape [N_l, 2]
    """
    device = g_crops.device

    # Get numbers for each branch.
    g_crop_size, N_g, gV = g_shapes
    l_crop_size, N_l, lV = l_shapes
    g_patch_positions=  g_patch_positions.unflatten(0, (gV, N_g))
    l_patch_positions=  l_patch_positions.unflatten(0, (lV, N_l))
    M_g = gV * N_g
    M_l = lV * N_l
    M = M_g + M_l

    # Undo normalization:
    # - Translations were divided by canonical_img_size.
    # - Scales were divided by log(max_scale_ratio).
    dT = pred_dT[..., :2] * canonical_img_size
    dS = pred_dT[..., 2:] * math.log(max_scale_ratio)

    # Use uniform weights.
    weight = torch.ones((M, M), device=device)

    # Choose the anchor: first patch of the first global crop (global index 0).
    # Compute its canonical absolute translation:
    #   T_anchor = global_crop_params[0][:2] + (patch_pos / g_crop_size)*global_crop_params[0][2:4]
    T_anchor = g_crop_params[0, :2] + (g_patch_positions[0, 0] / g_crop_size) * g_crop_params[0, 2:4]
    # Compute its canonical log-scale:
    # Here S_anchor is defined as log(patch_size * (global_crop_params[0][2:4] / g_crop_size))
    S_anchor = torch.log((patch_size * g_crop_params[0, 2:4] / g_crop_size))

    # Solve the LS systems for translation and log-scale.
    T_global = lstsq_dT_solver(dT, weight, anchor_index=0, anchor_value=T_anchor)
    S_global = lstsq_dT_solver(dS, weight, anchor_index=0, anchor_value=S_anchor)

    # Split the LS solutions back into global and local branches.
    g_translations = [T_global[v * N_g : (v + 1) * N_g] for v in range(gV)]
    g_log_scales = [S_global[v * N_g : (v + 1) * N_g] for v in range(gV)]
    l_translations = [T_global[M_g + v * N_l : M_g + (v + 1) * N_l] for v in range(lV)]
    l_log_scales = [S_global[M_g + v * N_l : M_g + (v + 1) * N_l] for v in range(lV)]

    # Reconstruct the canonical image by pasting each patch.
    # Assume both g_crops and l_crops have the same number of channels.
    C = g_crops.shape[1]
    canvas = torch.zeros((C, canonical_img_size, canonical_img_size), device=device)
    count_map = torch.zeros((1, canonical_img_size, canonical_img_size), device=device)

    # Helper: paste patches from one branch.
    def paste_patches(crops, patch_positions, crop_params, crop_size, translations_list, log_scales_list):
        V_branch, N = patch_positions.shape[:2]
        _, _, H, W = crops.shape
        for v in range(V_branch):
            # (For LS reconstruction, we rely solely on the LS translations.)
            for i in range(patch_positions.shape[1]):
                # Get LS predicted translation and log-scale for this patch.
                T_patch = translations_list[v][i]  # [2]
                S_patch = log_scales_list[v][i]      # [2]
                scale_factor = torch.exp(S_patch)    # [2] (per-dimension scaling)
                new_patch_h = int(round(scale_factor[0].item()))
                new_patch_w = int(round(scale_factor[1].item()))
                # Extract patch from crop.
                pos = patch_positions[v, i]
                y_crop, x_crop = int(round(pos[0].item())), int(round(pos[1].item()))
                y_crop = max(0, min(H - patch_size, y_crop))
                x_crop = max(0, min(W - patch_size, x_crop))
                patch = crops[v][:, y_crop:y_crop + patch_size, x_crop:x_crop + patch_size].unsqueeze(0)
                # Resize patch according to LS predicted scale.
                patch_resized = F.interpolate(patch, size=(new_patch_h, new_patch_w),
                                              mode='bilinear', align_corners=False).squeeze(0)
                # Use the LS translation directly as canonical coordinates.
                y_can = int(round(T_patch[0].item()))
                x_can = int(round(T_patch[1].item()))
                y_can = max(0, min(canonical_img_size - new_patch_h, y_can))
                x_can = max(0, min(canonical_img_size - new_patch_w, x_can))
                canvas[:, y_can:y_can + new_patch_h, x_can:x_can + new_patch_w] += patch_resized
                count_map[:, y_can:y_can + new_patch_h, x_can:x_can + new_patch_w] += 1

    # Paste global branch patches.
    paste_patches(g_crops, g_patch_positions, g_crop_params, g_crop_size, g_translations, g_log_scales)
    # Paste local branch patches.
    paste_patches(l_crops, l_patch_positions, l_crop_params, l_crop_size, l_translations, l_log_scales)

    count_map[count_map == 0] = 1
    reconstructed_img = canvas / count_map

    return reconstructed_img, g_translations, g_log_scales, l_translations, l_log_scales
