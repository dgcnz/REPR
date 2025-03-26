import torch
import torch.nn.functional as F


def reconstruction_gt(
    g_crops: torch.Tensor,      # [V, C, H, W] augmented crop images
    g_patch_positions_nopos: torch.Tensor,  # [V, N, 2] patch positions in crop coordinates (assumed top-left)
    patch_size: int,                # patch size in the crop (e.g., 16)
    g_crop_size: int,                  # size of the crop (e.g., 224)
    canonical_img_size: int,        # size of the canonical canvas (e.g., 512)
    g_crop_params: torch.Tensor       # [V, 4] each row is [y, x, h, w] used to generate the crop from canonical space
) -> torch.Tensor:
    """
    Reconstruct a canonical image from a set of augmented crops using ground truth patch positions and the crop parameters.

    For each view v:
      - Compute vertical and horizontal scale factors from crop_params:
            scale_y = crop_params[v,2] / img_size,
            scale_x = crop_params[v,3] / img_size.
      - The new patch dimensions will be:
            new_patch_h = patch_size * scale_y,
            new_patch_w = patch_size * scale_x.
      - Each patch’s top-left coordinate (in crop space) is mapped to canonical space via:
            canonical_pos = crop_params[v][:2] + (patch_pos / img_size) * crop_params[v][2:4].

    The function extracts patches from each crop, resizes them using the exact per-dimension scaling,
    and pastes them into a canonical canvas of size [C, canonical_img_size, canonical_img_size]. Overlapping areas are averaged.

    Args:
        image_crops: Tensor of shape [V, C, H, W] with augmented crop images.
        patch_positions: Tensor of shape [V, N, 2] with patch positions (top-left) in crop coordinates.
        patch_size: The patch size in the crop (e.g., 16).
        img_size: The size of the crop (e.g., 224).
        canonical_img_size: The size of the canonical canvas (e.g., 512).
        crop_params: Tensor of shape [V, 4] where each row is [y, x, h, w] describing the crop's location
                     and size in canonical space.

    Returns:
        A reconstructed canonical image of shape [C, canonical_img_size, canonical_img_size].
    """
    V, C, H, W = g_crops.shape
    _, N, _ = g_patch_positions_nopos.shape
    device = g_crops.device

    # Initialize a blank canonical canvas and a count map.
    canvas = torch.zeros((C, canonical_img_size, canonical_img_size), device=device)
    count_map = torch.zeros((1, canonical_img_size, canonical_img_size), device=device)

    for v in range(V):
        # For view v, retrieve the crop parameters: [y, x, h, w]
        cp = g_crop_params[v].float()  # [y, x, h, w]
        origin = cp[:2]  # top-left coordinate of the crop in canonical space
        size = cp[2:4]   # the crop's size in canonical space

        # Compute vertical and horizontal scale factors.
        scale_y = size[0] / g_crop_size
        scale_x = size[1] / g_crop_size

        new_patch_h = int(round(float(patch_size * scale_y)))
        new_patch_w = int(round(float(patch_size * scale_x)))

        for i in range(N):
            # Get patch position in crop coordinates.
            pos = g_patch_positions_nopos[v, i].float()  # [y, x]
            # Map to canonical coordinates using the crop parameters.
            canonical_pos = origin + (pos / g_crop_size) * size
            y_can, x_can = int(round(canonical_pos[0].item())), int(round(canonical_pos[1].item()))
            # Ensure the patch fits within the canonical canvas.
            y_can = max(0, min(canonical_img_size - new_patch_h, y_can))
            x_can = max(0, min(canonical_img_size - new_patch_w, x_can))

            # Extract the patch from the crop.
            y_crop, x_crop = int(round(pos[0].item())), int(round(pos[1].item()))
            y_crop = max(0, min(H - patch_size, y_crop))
            x_crop = max(0, min(W - patch_size, x_crop))
            patch = g_crops[v][:, y_crop:y_crop + patch_size, x_crop:x_crop + patch_size].unsqueeze(0)  # [1, C, patch_size, patch_size]

            # Resize the patch using the exact per-dimension scaling.
            patch_resized = F.interpolate(patch, size=(new_patch_h, new_patch_w),
                                          mode='bilinear', align_corners=False).squeeze(0)  # [C, new_patch_h, new_patch_w]

            # Paste the resized patch onto the canonical canvas.
            canvas[:, y_can:y_can + new_patch_h, x_can:x_can + new_patch_w] += patch_resized
            count_map[:, y_can:y_can + new_patch_h, x_can:x_can + new_patch_w] += 1

    count_map[count_map == 0] = 1
    reconstructed_img = canvas / count_map
    return reconstructed_img