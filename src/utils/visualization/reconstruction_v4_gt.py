import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor


def reconstruction_gt(
    g_crops: Float[Tensor, "gV C gH gW"],  # global crop images
    l_crops: Float[Tensor, "lV C lH lW"],  # local crop images
    g_patch_positions_nopos: Float[Tensor, "gV*gN 2"],  # global patch positions (y,x) in crop coordinates
    l_patch_positions_nopos: Float[Tensor, "lV*lN 2"],  # local patch positions (y,x) in crop coordinates
    patch_size: int,  # patch size in pixels (square patch)
    g_shapes: tuple[int, int, int],  # (crop_size, gN, gV) sizes for global crops
    l_shapes: tuple[int, int, int],  # (crop_size, lN, lV) sizes for local crops
    canonical_img_size: int,  # size of the output canonical image
    g_crop_params: Float[Tensor, "gV 4"],  # global crop params (y, x, h, w) in canonical coordinates
    l_crop_params: Float[Tensor, "lV 4"],  # local crop params (y, x, h, w) in canonical coordinates
) -> Float[Tensor, "C canonical_img_size canonical_img_size"]:
    """Reconstructs an image from global and local crops by projecting patches back to a canonical canvas.
    
    This function takes crops from different views and their associated patch positions, then maps these
    patches to their proper locations on a canonical image canvas. It handles the scaling between crop
    coordinates and canonical image coordinates using the provided crop parameters.
    
    :param Float[Tensor, "gV C gH gW"] g_crops: Global crop images, where gV is the number of global views,
        C is the number of channels, and gH/gW are the height/width of global crops
    :param Float[Tensor, "lV C lH lW"] l_crops: Local crop images, where lV is the number of local views,
        C is the number of channels, and lH/lW are the height/width of local crops
    :param Float[Tensor, "gV*gN 2"] g_patch_positions_nopos: Positions of global patches in crop coordinates,
        where gN is the number of patches per global crop
    :param Float[Tensor, "lV*lN 2"] l_patch_positions_nopos: Positions of local patches in crop coordinates,
        where lN is the number of patches per local crop
    :param int patch_size: Size of each square patch in pixels
    :param tuple[int, int, int] g_shapes: Tuple (crop_size, gN, gV) containing global crop size, 
        number of patches per crop, and number of global views
    :param tuple[int, int, int] l_shapes: Tuple (crop_size, lN, lV) containing local crop size, 
        number of patches per crop, and number of local views
    :param int canonical_img_size: Size of the square canonical output image in pixels
    :param Float[Tensor, "gV 4"] g_crop_params: Global crop parameters [y, x, h, w] in canonical coordinates
    :param Float[Tensor, "lV 4"] l_crop_params: Local crop parameters [y, x, h, w] in canonical coordinates
    
    :return: A reconstructed image tensor with shape [C, canonical_img_size, canonical_img_size].
        The reconstruction is formed by averaging overlapping patches in the canonical space.
    :rtype: Float[Tensor, "C canonical_img_size canonical_img_size"]
    """
    device = g_crops.device
    # Initialize a blank canonical canvas and a count map.
    C = g_crops.shape[1]
    canvas = torch.zeros((C, canonical_img_size, canonical_img_size), device=device)
    count_map = torch.zeros((1, canonical_img_size, canonical_img_size), device=device)

    for crops, patch_positions, crop_params, (crop_size, N, V) in [
        (g_crops, g_patch_positions_nopos, g_crop_params, g_shapes),
        (l_crops, l_patch_positions_nopos, l_crop_params, l_shapes),
    ]:
        V, C, H, W = crops.shape
        patch_positions = patch_positions.unflatten(0, (V, N))
        for v in range(V):
            # For view v, retrieve the crop parameters: [y, x, h, w]
            cp = crop_params[v].float()  # [y, x, h, w]
            origin = cp[:2]  # top-left coordinate of the crop in canonical space
            size = cp[2:4]  # the crop's size in canonical space

            # Compute vertical and horizontal scale factors.
            scale_y = size[0] / crop_size
            scale_x = size[1] / crop_size

            new_patch_h = int(round(float(patch_size * scale_y)))
            new_patch_w = int(round(float(patch_size * scale_x)))

            for i in range(N):
                # Get patch position in crop coordinates.
                pos = patch_positions[v, i].float()  # [y, x]
                # Map to canonical coordinates using the crop parameters.
                canonical_pos = origin + (pos / crop_size) * size
                y_can, x_can = int(round(canonical_pos[0].item())), int(
                    round(canonical_pos[1].item())
                )
                # Ensure the patch fits within the canonical canvas.
                y_can = max(0, min(canonical_img_size - new_patch_h, y_can))
                x_can = max(0, min(canonical_img_size - new_patch_w, x_can))

                # Extract the patch from the crop.
                y_crop, x_crop = int(round(pos[0].item())), int(round(pos[1].item()))
                y_crop = max(0, min(H - patch_size, y_crop))
                x_crop = max(0, min(W - patch_size, x_crop))
                patch = crops[v][
                    :, y_crop : y_crop + patch_size, x_crop : x_crop + patch_size
                ].unsqueeze(0)

                # Resize the patch using the exact per-dimension scaling.
                patch_resized = F.interpolate(
                    patch,
                    size=(new_patch_h, new_patch_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

                # Paste the resized patch onto the canonical canvas.
                canvas[
                    :, y_can : y_can + new_patch_h, x_can : x_can + new_patch_w
                ] += patch_resized
                count_map[
                    :, y_can : y_can + new_patch_h, x_can : x_can + new_patch_w
                ] += 1

    count_map[count_map == 0] = 1
    reconstructed_img = canvas / count_map
    return reconstructed_img
