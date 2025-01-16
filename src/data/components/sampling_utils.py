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
    :return: New image of shape (B, C, H, W) and positions (y, x) of each patch
    """
    B, C, H, W = img.size()

    # Sampling patches (top left corners)
    if mode == "offgrid":
        ys, xs = _sample_offgrid(B, H, W, patch_size, device=img.device)
    elif mode == "ongrid":
        ys, xs = _sample_ongrid(B, H, W, patch_size, device=img.device)
    elif mode == "canonical":
        ys, xs = _sample_ongrid(B, H, W, patch_size, canonical=True, device=img.device)
    elif mode == "ongrid_close_2":
        ys, xs = _sample_ongrid_close(
            B, H, W, patch_size, region_size=2, device=img.device
        )
    elif mode == "ongrid_close_4":
        ys, xs = _sample_ongrid_close(
            B, H, W, patch_size, region_size=4, device=img.device
        )
    else:
        raise ValueError("Invalid mode, choose from ['offgrid', 'ongrid', 'canonical']")

    patch_positions = torch.stack([ys, xs], dim=-1)

    # Creating sampling grid
    indices = _create_sampled_grid_flattened(patch_size, H, W, ys, xs)
    indices: Int[Tensor, "B C H*W"] = indices.unsqueeze(1).expand(-1, C, -1)
    new_img = img.flatten(-2).gather(2, indices).unflatten(-1, (H, W))
    return new_img, patch_positions


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


def extract_k_by_k_blocks_nonoverlapping(x: Tensor, k: int) -> Tensor:
    H, W = x.shape
    assert H % k == 0 and W % k == 0, "H and W must be multiples of k"
    x = x.view(H // k, k, W // k, k)
    x = x.permute(0, 2, 1, 3)
    x = x.contiguous().view(-1, k, k)
    return x


def extract_k_by_k_blocks(x: Tensor, k: int, stride: int) -> Tensor:
    """
    Extract k x k blocks from input tensor with given stride.
    For floating-point types check: https://github.com/pytorch/pytorch/issues/44989
    """
    H, W = x.shape
    assert k > 0 and stride > 0, "k and stride must be positive"
    assert H >= k and W >= k, "Input dimensions must be >= k"

    out_h = (H - k) // stride + 1
    out_w = (W - k) // stride + 1
    total_blocks = out_h * out_w

    x = x.contiguous()
    blocks = x.unfold(0, k, stride).unfold(1, k, stride).reshape(total_blocks, k, k)
    return blocks  


def _sample_ongrid_close(
    B: int,
    H: int,
    W: int,
    patch_size: int,
    region_size: int = 2,
    device: torch.device = "cpu",
) -> tuple[Int[Tensor, "B N"], Int[Tensor, "B N"]]:
    nH = H // patch_size
    nW = W // patch_size
    N = nH * nW
    idx = torch.arange(0, N, device=device).reshape(nH, nW)
    idx = extract_k_by_k_blocks_nonoverlapping(idx, region_size)
    # idx = idx.reshape(-1, region_size**2)
    idx = idx.flatten()
    # randperm?
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

    # patch_size = 4
    # dataset = CIFAR10(root="data/", download=True, train=False)
    # img, _ = dataset[0]
    # img = ToTensor()(img).unsqueeze(0)

    # sampler3 = sample_and_stitch
    # out = sampler3(img.cpu(), patch_size, "offgrid")

    # fig, ax = plt.subplots(1)
    # ax.imshow(out.squeeze().permute(1, 2, 0))
    # fig.savefig("test.png")

    # Test extract_k_by_k_blocks
    x = torch.arange(25).reshape(5, 5)
    print(x)
    blocks = extract_k_by_k_blocks(x, 4, 1)
    print(blocks)
