import pytest
from PIL import Image
import torchvision.transforms.v2.functional as TTFv2
import torch
from src.models.components.part_vit import (
    compute_gt_transform,
)
from utils.visualization.visualization import (
    compute_connected_components,
    reconstruct_image,
    compute_reconstruction_graph,
)

import matplotlib.pyplot as plt


@pytest.fixture
def img():
    img = Image.open("artifacts/img.jpg")
    img = img.resize((224, 224))
    patch_size = 16
    num_patches = (224 // patch_size) ** 2
    return img, patch_size, num_patches


@pytest.fixture
def patch_pair_indices_linear(img):
    _, _, num_patches = img
    return torch.stack([torch.zeros(num_patches), torch.arange(num_patches)], dim=1).to(
        torch.int64
    )


@pytest.fixture
def patch_positions_linear(img):
    _, patch_size, num_patches = img
    img_size = 224
    return torch.stack(
        [
            torch.arange(num_patches) // (img_size // patch_size),
            torch.arange(num_patches) % (img_size // patch_size),
        ],
        dim=1,
    ).to(torch.int64) * patch_size


@pytest.fixture
def gt_transform_linear(patch_pair_indices_linear, patch_positions_linear):
    return compute_gt_transform(
        patch_pair_indices_linear[None], patch_positions_linear[None]
    )[0]


"""
0   1   2   3
4   5   6   7
8   9   10  11
12  13  14  15
"""


def test_compute_gt_transform(patch_pair_indices_linear, patch_positions_linear):
    """
    GIVEN   an image of size (H, W) with patch_size P,
            the set of patch pair indices [(0, 0), ..., (0, N)],
            where N = (H // P) * (W // P) is the number of patches.
    WHEN    we compute the ground truth transformations
    THEN    the ground truth transformations should be the same as the patch positions
    """
    # GIVEN
    # WHEN
    gt_T = compute_gt_transform(
        patch_pair_indices_linear[None], patch_positions_linear[None]
    )[0]
    # THEN
    assert torch.allclose(gt_T, patch_positions_linear.to(torch.float32), atol=1e-6)


def test_reconstruction_graph(img, patch_pair_indices_linear, gt_transform_linear):
    """
    GIVEN   the set of patch pair indices [(0, 0), ..., (0, N)],
            where N is the number of patches,
            and the set of ground truth transformations [T_00, T_01, ..., T_0N]
    WHEN    we compute the reconstruction graph
    THEN    the graph should be a connected component
    """

    _, _, num_patches = img
    g = compute_reconstruction_graph(patch_pair_indices_linear, gt_transform_linear)
    components = compute_connected_components(g)
    assert len(components) == 1
    largest_component = max(components, key=lambda x: x[1])
    assert largest_component[0] == 0
    assert largest_component[1] == num_patches


def test_reconstruction_sanity_check(
    img, patch_pair_indices_linear, patch_positions_linear, gt_transform_linear
):
    """
    GIVEN   an image of size (H, W),
            patch size,
            the set of pair indices [(0, 0), (0, 1), ..., (0, N)],
            where N is the number of patches.
            and the set of ground truth transformations [T_00, T_01, ..., T_0N]
    WHEN    we compute the reconstructed image
    THEN    the reconstructed image should be the same as the original image
    """

    # GIVEN
    img, patch_size, num_patches = img
    img = TTFv2.to_tensor(img)
    img_size = img.shape[-1]

    # WHEN
    reconstructed_img = reconstruct_image(
        gt_transform_linear,
        patch_pair_indices_linear,
        patch_positions_linear,
        patch_size,
        img,
    )
    # THEN
    # plot 2 images side by side
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(TTFv2.to_pil_image(img))
    ax[0].set_title("Original Image")
    ax[0].axis("off")
    ax[1].imshow(TTFv2.to_pil_image(reconstructed_img))
    ax[1].set_title("Reconstructed Image")
    ax[1].axis("off")
    fig.savefig("artifacts/reconstructed_img.jpg")
    assert img.shape == reconstructed_img.shape
    assert torch.allclose(img, reconstructed_img, atol=1e-6)


def test_reconstruction_uncomplete_pairs(
    img, patch_pair_indices_linear, patch_positions_linear
):
    """
    GIVEN   an image of size (H, W),
            patch size,
            the set of pair indices [(0, 0), (0, 1), ..., (0, N)],
            where N is the number of patches.
            and the set of ground truth transformations [T_00, T_01, ..., T_0N]
    WHEN    we compute the reconstructed image
    THEN    the reconstructed image should be the same as the original image
    """

    # GIVEN
    img, patch_size, num_patches = img
    img = TTFv2.to_tensor(img)
    img_size = img.shape[-1]
    # remove half ot the pairs
    patch_pair_indices_linear = patch_pair_indices_linear[::2]

    gt_transform_linear = compute_gt_transform(
        patch_pair_indices_linear[None], patch_positions_linear[None]
    )[0]

    # WHEN
    reconstructed_img = reconstruct_image(
        gt_transform_linear,
        patch_pair_indices_linear,
        patch_positions_linear,
        patch_size,
        img,
    )
    # THEN
    # plot 2 images side by side
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(TTFv2.to_pil_image(img))
    ax[0].set_title("Original Image")
    ax[0].axis("off")
    ax[1].imshow(TTFv2.to_pil_image(reconstructed_img))
    ax[1].set_title("Reconstructed Image")
    ax[1].axis("off")
    fig.savefig("artifacts/reconstructed_img_uncomplete.jpg")


