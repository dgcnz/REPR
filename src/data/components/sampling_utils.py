import torch
from jaxtyping import Float, Int
from torch import Tensor
from typing import Literal


def _create_sampled_grid_flattened(
    patch_size: int, H: int, W: int, ys: Int[Tensor, "B N"], xs: Int[Tensor, "B N"]
) -> Int[Tensor, "B H*W"]:
    """
    Create full indices for each patch index specified by `ys` and `xs`.

    For example, let:
        - ys[0] = 0, xs[0] = 0,  patch_size = 4, H = 32, W = 32

    Then the indices corresponding to patch 0 would be:
    [0, 1, 2, 3, 32, 33, 34, 35, 64, 65, 66, 67, 96, 97, 98, 99]

    This is equivalent to the 2D (y, x) indices:
    [
        [(0, 0), (0, 1), (0, 2), (0, 3)],
        [(1, 0), (1, 1), (1, 2), (1, 3)],
        [(2, 0), (2, 1), (2, 2), (2, 3)],
        [(3, 0), (3, 1), (3, 2), (3, 3)]
    ]

    :param patch_size: Size of the patches to sample
    :param H: Height of the image
    :param W: Width of the image
    :param ys: y coordinates of the top-left corner of the patches
    :param xs: x coordinates of the top-left corner of the patches
    """
    B, nH, nW = ys.size(0), H // patch_size, W // patch_size
    zs = ys * W + xs
    dx = dy = torch.arange(patch_size, device=ys.device)
    d: Int[Tensor, "P P"] = W * dy.unsqueeze(-1) + dx
    indices: Int[Tensor, "B N H W"] = zs[:, :, None, None] + d
    indices = indices.view(B, nH, nW, patch_size, patch_size)
    indices = indices.permute(0, 1, 3, 2, 4).reshape(B, H * W)
    return indices


def sample_and_stitch(
    img: Float[Tensor, "B C H W"],
    patch_size: int,
    mode: Literal["offgrid", "ongrid", "canonical"] = "offgrid",
) -> tuple[Float[Tensor, "B C H W"], Int[Tensor, "B N 2"]]:
    """
    Sample patches from an image and stitch them back together.

    For more implementations see `tests/test_benchmark_sample.py`.

    :param img: Input image of shape (B, C, H, W)
    :param patch_size: Size of the patches to sample
    :param mode: Sampling mode, choose from ['offgrid', 'ongrid', 'canonical']

    """
    B, C, H, W = img.size()

    # Sampling patches (top left corners)
    if mode == "offgrid":
        ys, xs = _sample_offgrid(B, H, W, patch_size, device=img.device)
    elif mode == "ongrid":
        ys, xs = _sample_ongrid(B, H, W, patch_size, device=img.device)
    elif mode == "canonical":
        ys, xs = _sample_ongrid(B, H, W, patch_size, canonical=True, device=img.device)
    else:
        raise ValueError("Invalid mode, choose from ['offgrid', 'ongrid', 'canonical']")

    idx = torch.stack([ys, xs], dim=-1)

    # Creating sampling grid
    indices = _create_sampled_grid_flattened(patch_size, H, W, ys, xs)
    indices: Int[Tensor, "B C H*W"] = indices.unsqueeze(1).expand(-1, C, -1)
    new_img = img.flatten(-2).gather(2, indices).unflatten(-1, (H, W))
    return new_img, idx


def _sample_offgrid(
    B: int, H: int, W: int, patch_size: int, device: torch.device = "cpu"
) -> tuple[Int[Tensor, "B N"], Int[Tensor, "B N"]]:
    """
    Sample random patches off-grid.

    Note: Each patch is represented by the top-left corner.

    :param B: Batch size
    :param H: Height of the image
    :param W: Width of the image
    :param patch_size: Size of the patches to sample
    :param device: Device to use
    """
    nW = W // patch_size
    nH = H // patch_size
    xs = torch.randint(0, H - patch_size, (B, nH * nW), device=device)
    ys = torch.randint(0, W - patch_size, (B, nH * nW), device=device)
    return ys, xs


def _sample_ongrid(
    B: int,
    H: int,
    W: int,
    patch_size: int,
    canonical: bool = False,
    device: torch.device = "cpu",
) -> tuple[Int[Tensor, "B N"], Int[Tensor, "B N"]]:
    """
    Sample patches on-grid, that is, all corners are multiples of `patch_size`.

    :param B: Batch size
    :param H: Height of the image
    :param W: Width of the image
    :param patch_size: Size of the patches to sample
    :param canonical: If true, the patches are just retrieved in canonical order
    :param device: Device to use
    """
    nW = W // patch_size
    nH = H // patch_size
    idx = torch.arange(nH * nW, device=device)
    if not canonical:
        idx = idx[torch.randperm(nH * nW, device=device)]
    idx = idx.repeat(B, 1)
    xs = idx % nW
    ys = idx // nW
    xs = xs * patch_size
    ys = ys * patch_size
    return ys, xs


if __name__ == "__main__":
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import ToTensor
    import matplotlib.pyplot as plt

    patch_size = 4
    dataset = CIFAR10(root="data/", download=True, train=False)
    img, _ = dataset[0]
    img = ToTensor()(img).unsqueeze(0)

    sampler3 = sample_and_stitch
    out = sampler3(img.cpu(), patch_size, "offgrid")

    fig, ax = plt.subplots(1)
    ax.imshow(out.squeeze().permute(1, 2, 0))
    fig.savefig("test.png")
