import torch
from jaxtyping import Float, Int
from typing import Callable
from torch import Tensor
from src.utils.fancy_indexing import index_pairs


def compute_gt_transform(
    patch_pair_indices: Int[Tensor, "b n_pairs 2"],
    patch_positions: Int[Tensor, "b n_patches 2"],
) -> Float[Tensor, "b n_pairs 2"]:
    true_positions: Float[Tensor, "b n_pairs 2 2"] = index_pairs(
        patch_positions, patch_pair_indices
    )
    true_transform: Float[Tensor, "b n_pairs 2"] = (
        true_positions[:, :, 1] - true_positions[:, :, 0]
    )
    return true_transform.float()


def get_all_pairs(b: int, n: int, device="cpu") -> Float[Tensor, "b n*n 2"]:
    """
    Get all pairs of indices from n patches.
    Example: (0, 0), (0, 1), ..., (1, 0), (1, 1), ..., (n, n)

    :param b: batch size
    :param n: number of patches
    """
    return (
        torch.stack(
            torch.meshgrid(
                torch.arange(n, device=device),
                torch.arange(n, device=device),
            ),
            dim=-1,
        )
        .reshape(1, -1, 2)
        .expand(b, -1, -1)
    )


def get_all_pairs_by_ongrid_regions(
    batch_size: int,
    n_patches: int,
    region_size: int,
    add_inter_region_pairs: bool = True,
    device="cpu",
):
    n_patches_per_region = region_size**2
    n_regions = n_patches // n_patches_per_region

    # Generate indices for one region
    region_indices = torch.arange(n_patches_per_region, device=device)
    region_pairs = torch.stack(
        torch.meshgrid(region_indices, region_indices), dim=-1
    ).reshape(-1, 2)

    # Offset indices for each region
    offsets = torch.arange(n_regions, device=device).unsqueeze(1) * n_patches_per_region
    all_pairs = (region_pairs.unsqueeze(0) + offsets.unsqueeze(2)).reshape(-1, 2)

    # Add inter-region pairs
    if add_inter_region_pairs:
        first_patch_in_each_region = torch.arange(0, n_patches, n_patches_per_region, device=device)
        # generate all pairs between all first patches in each region
        inter_region_pairs = torch.stack(
            torch.meshgrid(first_patch_in_each_region, first_patch_in_each_region), dim=-1
        ).reshape(-1, 2)
        all_pairs = torch.cat([all_pairs, inter_region_pairs], dim=0)


    # Expand to batch size
    all_pairs = all_pairs.expand(batch_size, -1, -1)
    return all_pairs


def random_pairs(
    b: int, n: int, n_pairs: int, device="cpu"
) -> Float[Tensor, "b n_pairs 2"]:
    """
    Randomly sample n_pairs pairs of indices from n patches.
    :param b: batch size
    :param n: number of patches
    :param n_pairs: number of pairs to sample
    """
    idx = torch.randint(n, (b, n_pairs), device=device)
    jdx = torch.randint(n, (b, n_pairs), device=device)
    return torch.stack([idx, jdx], dim=2)


def pivot_to_all_pairs(b: int, n: int, device="cpu", pivot=0) -> Float[Tensor, "b n 2"]:
    """
    Sample all pairs with the first element being pivot (default:0) in order.
    Example: (0, 0), (0, 1), ..., (0, n)

    :param b: batch size
    :param n: number of patches
    """
    idx = torch.full((b, n), pivot, dtype=torch.long, device=device)
    jdx = torch.arange(n, device=device).unsqueeze(0).expand(b, -1)
    return torch.stack([idx, jdx], dim=2)

class PARTLoss(torch.nn.Module):
    def __init__(self, img_size: tuple[int, int], f: Callable | None = None):
        super().__init__()
        self.f = f
        self.img_size = img_size
        self.max_T = max(img_size)
        self.min_T = -self.max_T

    def forward(self, pred_T: Tensor, gt_T: Tensor):
        pred_T = (pred_T - self.min_T) / (self.max_T - self.min_T)
        gt_T = (gt_T - self.min_T) / (self.max_T - self.min_T)
        loss = self.f(torch.abs(pred_T - gt_T)).sum()
        return loss


def compute_symmetry_loss(
    patch_pair_indices: Int[Tensor, "B n_pairs 2"],
    transforms: Float[Tensor, "B n_pairs 2"],
    n_patches: int,
) -> Tensor:
    batch_size = patch_pair_indices.shape[0]
    T_matrix = torch.zeros(
        batch_size,
        n_patches,
        n_patches,
        2,
        device=transforms.device,
        dtype=transforms.dtype,
    )

    batch_idx = (
        torch.arange(batch_size, device=transforms.device)
        .unsqueeze(1)
        .repeat(1, patch_pair_indices.shape[1])
    )
    T_matrix[batch_idx, patch_pair_indices[..., 0], patch_pair_indices[..., 1]] = (
        transforms
    )

    symmetry_loss = (T_matrix + T_matrix.transpose(1, 2)).abs().mean()
    return symmetry_loss


if __name__ == "__main__":
    # Example usage
    b = 1
    n = 16
    region_size = 2
    device = "cpu"

    pairs = get_all_pairs_by_ongrid_regions(b, n, region_size, device)
    num_pairs_per_region = (region_size**2)**2
    print(pairs)
    # print(pairs.reshape(b, -1, num_pairs_per_region, 2))
