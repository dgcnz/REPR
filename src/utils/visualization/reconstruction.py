import torch
from torch import Tensor
from jaxtyping import Float, Int
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from copy import copy


def get_center_refpatch_id(
    patch_positions: Int[Tensor, "B n_patches 2"],
    image_size: tuple[int, int],
) -> Int[Tensor, "B"]:
    """
    Get the reference patch id which is the closest to the center of the image
    """
    H, W = image_size
    center_y, center_x = H // 2, W // 2
    distances = (patch_positions[:, :, 0] - center_y) ** 2 + (
        patch_positions[:, :, 1] - center_x
    ) ** 2
    return torch.argmin(distances, dim=1)


def _get_refpatch_id_batch(
    patch_positions: Int[Tensor, "B n_patches 2"], img_size: tuple[int, int]
) -> Int[Tensor, "B"]:
    # Create ideal refpatch positions for all batches at once
    ideal_refpatch_yx = torch.tensor(img_size, device=patch_positions.device) // 2
    
    # Compute L1 distances in batched manner
    distances = (patch_positions - ideal_refpatch_yx.view(1, 1, 2)).abs().sum(2)
    
    # Get closest patch index for each batch
    refpatch_ids = torch.argmin(distances, dim=1)
    
    return refpatch_ids


def create_image_from_transforms(
    ref_transforms: dict[int, Tensor],
    patch_positions: Int[Tensor, "n_patches 2"],
    patch_size: int,
    img: Float[Tensor, "3 H W"],
    refpatch_id: int,
):
    new_img = torch.zeros_like(img)
    _, H, W = img.shape
    for node, t in ref_transforms.items():
        y, x = patch_positions[node]
        t_rounded = t.round().to(torch.int64)
        new_y, new_x = patch_positions[refpatch_id] + t_rounded
        # TODO: relax this constraint:
        # Instead of rejecting the entire patch if there's at least a pixel outside the image,
        # we can just paint the part of the patch that is inside the image.
        if 0 <= new_y <= H - patch_size and 0 <= new_x <= W - patch_size:
            new_img[:, new_y : new_y + patch_size, new_x : new_x + patch_size] = img[
                :, y : y + patch_size, x : x + patch_size
            ]
    return new_img


def reconstruct_image_from_sampling(
    patch_positions: Int[Tensor, "n_patches 2"],
    patch_size: int,
    img: Float[Tensor, "3 H W"],
):
    """
    Reconstruct the image using the predicted transformations.
    """
    n_patches, _ = patch_positions.size()
    assert tuple(patch_positions.shape) == (n_patches, 2)
    assert img.size(0) == 3
    assert img.dim() == 3

    new_img = torch.zeros_like(img)
    _, H, W = img.shape
    for node in range(n_patches):
        y, x = patch_positions[node]
        new_img[:, y : y + patch_size, x : x + patch_size] = img[
            :, y : y + patch_size, x : x + patch_size
        ]

    return new_img


def plot_reconstructions(
    refpatch_ids: list[int],
    ref_transforms: list[dict[int, Tensor]],
    patch_positions: Int[Tensor, "B n_patches 2"],
    img_original: Float[Tensor, "B 3 H W"],
    img_input: Float[Tensor, "B 3 H W"],
    patch_size: int,
):
    B = len(refpatch_ids)
    fig, axes = plt.subplots(B, 4, figsize=(20, 5 * B), squeeze=False)
    for i, ax in enumerate(axes):
        reconstructed_image = create_image_from_transforms(
            ref_transforms=ref_transforms[i],
            patch_positions=patch_positions[i],
            patch_size=patch_size,
            img=img_original[i],
            refpatch_id=refpatch_ids[i],
        )

        img_sampled = reconstruct_image_from_sampling(
            patch_positions=patch_positions[i],
            patch_size=patch_size,
            img=img_original[i],
        )
        y, x = patch_positions[i][refpatch_ids[i]]
        rect = Rectangle(
            (x, y),
            patch_size,
            patch_size,
            linewidth=1,
            edgecolor="red",
            facecolor="none",
        )

        ax[0].imshow(img_original[i].permute(1, 2, 0))
        ax[0].set_title("Original Image")
        ax[0].add_patch(copy(rect))
        ax[1].imshow(img_sampled.permute(1, 2, 0))
        ax[1].set_title("Sampled Image")
        ax[1].add_patch(copy(rect))

        ax[2].imshow(reconstructed_image.permute(1, 2, 0))
        ax[2].set_title("Reconstructed Image")
        ax[2].add_patch(copy(rect))

        ax[3].imshow(img_input[i].permute(1, 2, 0))
        ax[3].set_title("Input Image")
    return fig
