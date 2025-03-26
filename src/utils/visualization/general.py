from jaxtyping import Float
from torch import Tensor
import torch.nn.functional as F


def paste_patch(
    crop: Float[Tensor, "C h w"],
    pos: Float[Tensor, "2"],
    pos_canonical: Float[Tensor, "2"],
    patch_size_canonical: Float[Tensor, "2"],
    canvas: Float[Tensor, "C H W"],
    count_map: Float[Tensor, "1 H W"],
    patch_size: int,
    canonical_size: int,
):
    """
    Extract a patch from a crop at pos and paste it onto a canvas at pos_canonical with appropriate rescaling.

    Args:
        crop: Source image crop of shape [C, h, w]
        pos: Patch position in crop coordinates [y, x]
        pos_canonical: Target position in canonical coordinates [y, x]
        patch_size_canonical: Size of patch in canonical space [height, width]
        canvas: Target canvas to paste onto [C, H, W]
        count_map: Counter for averaging overlapping patches [1, H, W]
        patch_size: Size of patch in crop space
        canonical_size: Size of the canonical image
    """
    crop_h, crop_w = crop.shape[1:3]

    # Convert to integer coordinates for the canonical position
    y_canonical, x_canonical = int(round(pos_canonical[0].item())), int(
        round(pos_canonical[1].item())
    )

    # Get integer patch size for the canonical space
    patch_h_canonical, patch_w_canonical = patch_size_canonical.round().int()

    # Ensure the patch fits within the canonical canvas
    y_canonical = max(0, min(canonical_size - patch_h_canonical, y_canonical))
    x_canonical = max(0, min(canonical_size - patch_w_canonical, x_canonical))

    # Get source patch coordinates, ensuring they're within the crop boundaries
    y_crop, x_crop = int(round(pos[0].item())), int(round(pos[1].item()))
    y_crop = max(0, min(crop_h - patch_size, y_crop))
    x_crop = max(0, min(crop_w - patch_size, x_crop))

    # Extract the patch from the source crop
    patch = crop[
        :, y_crop : y_crop + patch_size, x_crop : x_crop + patch_size
    ].unsqueeze(0)

    # Resize the patch to the canonical size
    patch_resized = F.interpolate(
        patch,
        size=(patch_h_canonical, patch_w_canonical),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    # Add the patch to the canvas and update the count map
    canvas[
        :,
        y_canonical : y_canonical + patch_h_canonical,
        x_canonical : x_canonical + patch_w_canonical,
    ] += patch_resized
    count_map[
        :,
        y_canonical : y_canonical + patch_h_canonical,
        x_canonical : x_canonical + patch_w_canonical,
    ] += 1
