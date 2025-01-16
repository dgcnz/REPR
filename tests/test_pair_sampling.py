import torch
from jaxtyping import Float
from torch import Tensor



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


def test_get_all_pairs():
    b = 2
    n = 3
    expected = torch.tensor(
        [
            [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]],
            [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]],
        ]
    )
    assert torch.equal(get_all_pairs(b, n), expected)


def test_random_pairs():
    b = 2
    n = 3
    n_pairs = 4
    pairs = random_pairs(b, n, n_pairs)
    assert pairs.shape == (b, n_pairs, 2)
    assert pairs.min() >= 0
    assert pairs.max() < n


def test_zero_to_all_pairs():
    b = 2
    n = 3
    pairs = pivot_to_all_pairs(b, n)
    assert pairs.shape == (b, n, 2)
    assert torch.equal(pairs[:, 0, :], torch.zeros(b, 2, dtype=torch.long))
    assert torch.equal(pairs[:, :, 1], torch.arange(n).unsqueeze(0).expand(b, -1))
