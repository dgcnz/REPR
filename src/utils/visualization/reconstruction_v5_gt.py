from jaxtyping import Float, Int
from torch import Tensor
import torch
import torch.nn.functional as F
from src.utils.visualization.general import paste_patch


def reconstruction_gt(
    x: list[Float[Tensor, "C gH gW"] | Float[Tensor, "C lH lW"]],
    patch_positions_nopos: Float[Tensor, "M 2"],
    crop_params: list[Float[Tensor, "4"]],
    num_tokens: list[int],  # [gM, ..., lM, lM, ..., lM] (gM #gV times, lM #lV times)
    patch_size: int,
    canonical_img_size: int,
) -> Float[Tensor, "B C canonical_img_size canonical_img_size"]:
    device = x[0].device
    C = x[0].shape[0]

    patch_positions_nopos_grouped = torch.split(patch_positions_nopos, num_tokens)

    # Initialize output canvas and count maps.
    canvas = torch.zeros((C, canonical_img_size, canonical_img_size), device=device)
    count_map = torch.zeros((1, canonical_img_size, canonical_img_size), device=device)

    assert (
        len(x) == len(crop_params) == len(patch_positions_nopos_grouped)
    ), f"Input lists must have the same length: {len(x)}, {len(crop_params)}, {len(patch_positions_nopos_grouped)}"

    for crop, params, patch_positions in zip(
        x, crop_params, patch_positions_nopos_grouped
    ):
        N = patch_positions.shape[0]
        H, W = crop.shape[1:3]
        crop_size = H
        cp = params.float()  # [y, x, h, w] in canonical coordinates
        origin = cp[:2]
        size = cp[2:4]
        patch_size_canonical = patch_size * (size / crop_size)

        # For each patch position in the grid.
        for i in range(N):
            pos = patch_positions[i].float()  # in crop coordinates
            pos_canonical = origin + (pos / crop_size) * size
            paste_patch(
                crop=crop,
                pos=pos,
                pos_canonical=pos_canonical,
                patch_size_canonical=patch_size_canonical,
                canvas=canvas,
                count_map=count_map,
                patch_size=patch_size,
                canonical_size=canonical_img_size,
            )

    count_map[count_map == 0] = 1
    return canvas / count_map
