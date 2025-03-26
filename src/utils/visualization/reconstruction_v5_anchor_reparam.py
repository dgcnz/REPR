import torch
import math
import torch.nn.functional as F
from src.utils.visualization.lstsq_solver import lstsq_dT_solver
from src.utils.visualization.general import paste_patch
from jaxtyping import Float, Int
from torch import Tensor


def reconstruction_lstsq_with_anchor_reparam(
    x: list[Float[Tensor, "C gH gW"] | Float[Tensor, "C lH lW"]],
    patch_positions_nopos: Float[Tensor, "M 2"],
    num_tokens: list[int],  # [gM, ..., lM, lM, ..., lM] (gM #gV times, lM #lV times)
    crop_params: list[Float[Tensor, "4"]],
    patch_size: int,
    canonical_img_size: int,
    max_scale_ratio: float,  # e.g., 4.0
    pred_dT: Float[Tensor, "M M 4"],
    # predicted pairwise differences (translation and log-scale)
) -> tuple[
    Float[Tensor, "C canonical_img_size canonical_img_size"],  # reconstructed image
    list[Float[Tensor, "N_g 2"]],  # global translations
    list[Float[Tensor, "N_g 2"]],  # global log scales
    list[Float[Tensor, "N_l 2"]],  # local translations
    list[Float[Tensor, "N_l 2"]],  # local log scales
]:
    device = x[0].device
    C = x[0].shape[0]
    M = pred_dT.shape[0]

    # Undo normalization:
    # - Translations were divided by canonical_img_size.
    # - Scales were divided by log(max_scale_ratio).
    dT = pred_dT[..., :2] * canonical_img_size
    dS = pred_dT[..., 2:] * math.log(max_scale_ratio)

    # Use uniform weights.
    weight = torch.ones((M, M), device=device)

    # Choose the anchor: first patch of the first global crop (global index 0).
    T_anchor = (
        crop_params[0][:2]
        + (patch_positions_nopos[0] / x[0].shape[1]) * crop_params[0][2:4]
    )
    S_anchor = torch.log((patch_size * crop_params[0][2:4] / x[0].shape[1]))

    # Solve the LS systems for translation and log-scale.
    T_global = lstsq_dT_solver(dT, weight, anchor_index=0, anchor_value=T_anchor)
    S_global = lstsq_dT_solver(dS, weight, anchor_index=0, anchor_value=S_anchor)

    T_global_grouped = torch.split(T_global, num_tokens)
    S_global_grouped = torch.split(S_global, num_tokens)
    patch_positions_nopos_grouped = torch.split(patch_positions_nopos, num_tokens)

    # Reconstruct the canonical image by pasting each patch.
    # Assume both g_crops and l_crops have the same number of channels.
    canvas = torch.zeros((C, canonical_img_size, canonical_img_size), device=device)
    count_map = torch.zeros((1, canonical_img_size, canonical_img_size), device=device)

    for crop, patch_positions, canonical_pos, log_size in zip(
        x,
        patch_positions_nopos_grouped,
        T_global_grouped,
        S_global_grouped,
    ):
        N = patch_positions.shape[0]
        for i in range(N):
            paste_patch(
                crop=crop,
                pos=patch_positions[i].float(),
                pos_canonical=canonical_pos[i],  # [y, x]
                patch_size_canonical=torch.exp(log_size[i]),  # [h, w]
                canvas=canvas,
                count_map=count_map,
                patch_size=patch_size,
                canonical_size=canonical_img_size,
            )

    count_map[count_map == 0] = 1
    reconstructed_img = canvas / count_map

    return reconstructed_img, T_global, S_global
