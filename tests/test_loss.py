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



if __name__ == "__main__":
    backbone = timm.create_model(
        "vit_base_patch16_384",
        pretrained=False,
        num_classes=0,
        global_pool="",
        class_token=False,
        img_size=64,
    )
    model = PARTViT(
        backbone=backbone,
        n_pairs=4,
        embed_dim=backbone.embed_dim,
        num_patches=backbone.patch_embed.num_patches,
        head_type="cross_attention",
        num_targets=2,
        cross_attention_num_heads=4,
        cross_attention_query_type="positional",
    )
    x = torch.randn(2, 3, 64, 64)
    x, patch_positions = sample_and_stitch(x, 16, "offgrid")
    y_pred, patch_pair_indices = model(x)
    y_gt = compute_gt_transform(patch_pair_indices, patch_positions)

    # plot reconstructed image given the predicted patch pair indices

    mse = nn.MSELoss()(y_pred, y_gt)
    print(mse)
