from jaxtyping import Float, Int
import torch
from torch import Tensor
import pytest

def index_select_2d(z: Float[Tensor, "b n d"], idx: Int[Tensor, "b k"], idy: Int[Tensor, "b k"]) -> Float[Tensor, "b k 2 d"]:
    b, n, dim = z.shape
    _, k = idx.shape
    indices = torch.stack([idy, idx], dim=2).view(b, -1)
    indices = indices + torch.arange(b, device=z.device).unsqueeze(-1) * n
    indices = indices.flatten()
    return z.flatten(0,1).index_select(0, indices).view(b, k, 2,  dim)

def index_select_2d_v4(z: Float[Tensor, "b n d"], idx: Int[Tensor, "b k"], idy: Int[Tensor, "b k"]) -> Float[Tensor, "b k 2 d"]:
    b, n, dim = z.shape
    _, k = idx.shape
    batch_offsets = torch.arange(b, device=z.device) * n
    indices = torch.stack([idy, idx], dim=2) + batch_offsets[:, None, None]
    indices = indices.flatten()
    return z.flatten(0,1).index_select(0, indices).view(b, k, 2,  dim)

def index_select_2d_v5(z: Float[Tensor, "b n d"], idx: Int[Tensor, "b k"], idy: Int[Tensor, "b k"]) -> Float[Tensor, "b k 2 d"]:
    b, n, dim = z.shape
    _, k = idx.shape

    batch_offsets = torch.arange(b, device=z.device)[:, None, None] * n  # Shape: (b, 1, 1)
    indices = torch.stack([idy, idx], dim=2) + batch_offsets

    # Flatten the indices for advanced indexing
    indices = indices.flatten()  # Shape: (b * k * 2,)

    # Flatten z for indexing and reshape the result
    selected = z.flatten(0, 1)[indices]  # Shape: (b * k * 2, d)

    # Reshape to desired output shape
    return selected.view(b, k, 2, dim)  # Shape: (b, k, 2, d) 

def index_select_2d_v3(z: Float[Tensor, "b n d"], idx: Int[Tensor, "b k"], idy: Int[Tensor, "b k"]) -> Float[Tensor, "b k 2 d"]:
    b, n, dim = z.shape
    
    # Efficient batch indexing calculation
    batch_offsets = torch.arange(b, device=z.device)[: None, None] * n
    
    # Create combined indices with batch offsets
    combined_indices = torch.stack([idy, idx], dim=2)  # Shape: (b, k, 2)
    global_indices = (combined_indices + batch_offsets).view(-1)  # Shape: (b * k * 2,)
    
    # Flatten input tensor for efficient indexing
    flat_z = z.flatten(0, 1)  # Shape: (b * n, d)
    
    # Select and reshape
    return flat_z[global_indices].view(b, -1, 2, dim)

def index_select_2d_v2(z: Float[Tensor, "b n d"], idx: Int[Tensor, "b k"], idy: Int[Tensor, "b k"]) -> Float[Tensor, "b k 2 d"]:
    b, n, dim = z.shape
    _, k = idx.shape

    # Combine idy and idx into a single index tensor
    combined_indices = torch.stack([idy, idx], dim=2)  # Shape: (b, k, 2)

    # Add batch offsets to the indices
    batch_offsets = torch.arange(b, device=z.device)[:, None, None] * n  # Shape: (b, 1, 1)
    combined_indices = combined_indices +  batch_offsets  # Broadcasting: Shape: (b, k, 2)

    # Flatten the indices for advanced indexing
    flat_indices = combined_indices.view(-1)  # Shape: (b * k * 2,)

    # Flatten z for indexing and reshape the result
    flat_z = z.flatten(0, 1)  # Shape: (b * n, d)
    selected = flat_z[flat_indices]  # Shape: (b * k * 2, d)

    # Reshape to desired output shape
    return selected.view(b, k, 2, dim)  # Shape: (b, k, 2, d)

def func_gather(z: Tensor, idx, idy):
    b, n, dim = z.shape
    zy = z.gather(1, idy.unsqueeze(-1).expand(-1, -1, dim))
    zx = z.gather(1, idx.unsqueeze(-1).expand(-1, -1, dim))
    return torch.stack([zy, zx], dim=2)


@pytest.mark.parametrize("batch_size", [64, 128])
@pytest.mark.parametrize("image_size", [224]) 
@pytest.mark.parametrize("n_pairs", [2500, 5000]) 
def test_2d_index_select(batch_size: int, image_size: int, n_pairs: int, benchmark_v2):
    device = "cuda"
    patch_size, dim = 16, 768
    num_patches = int((image_size // patch_size) ** 2)

    z: Float[Tensor, "b n d"] = torch.randn(batch_size, num_patches, dim).to(device)
    idy = torch.randint(num_patches, (batch_size, n_pairs)).to(z.device)
    idx = torch.randint(num_patches, (batch_size, n_pairs)).to(z.device)

    

    group_name = f"Benchmark[B: {batch_size}, I: {image_size}, P: {n_pairs}]"
    benchmark_v2.benchmark(index_select_2d, args=(z, idx, idy), n_warmup=2, n_runs=500)
    benchmark_v2.group_by(group_name)

@pytest.mark.parametrize("batch_size", [64, 128])
@pytest.mark.parametrize("image_size", [224]) 
@pytest.mark.parametrize("n_pairs", [2500, 5000]) 
def test_2d_index_select_v2(batch_size: int, image_size: int, n_pairs: int, benchmark_v2):
    device = "cuda"
    patch_size, dim = 16, 768
    num_patches = int((image_size // patch_size) ** 2)

    z: Float[Tensor, "b n d"] = torch.randn(batch_size, num_patches, dim).to(device)
    idy = torch.randint(num_patches, (batch_size, n_pairs)).to(z.device)
    idx = torch.randint(num_patches, (batch_size, n_pairs)).to(z.device)

    

    group_name = f"Benchmark[B: {batch_size}, I: {image_size}, P: {n_pairs}]"
    benchmark_v2.benchmark(index_select_2d_v2, args=(z, idx, idy), n_warmup=2, n_runs=500)
    benchmark_v2.group_by(group_name)

@pytest.mark.parametrize("batch_size", [64, 128])
@pytest.mark.parametrize("image_size", [224]) 
@pytest.mark.parametrize("n_pairs", [2500, 5000]) 
def test_2d_index_select_v3(batch_size: int, image_size: int, n_pairs: int, benchmark_v2):
    device = "cuda"
    patch_size, dim = 16, 768
    num_patches = int((image_size // patch_size) ** 2)

    z: Float[Tensor, "b n d"] = torch.randn(batch_size, num_patches, dim).to(device)
    idy = torch.randint(num_patches, (batch_size, n_pairs)).to(z.device)
    idx = torch.randint(num_patches, (batch_size, n_pairs)).to(z.device)

    

    group_name = f"Benchmark[B: {batch_size}, I: {image_size}, P: {n_pairs}]"
    benchmark_v2.benchmark(index_select_2d_v3, args=(z, idx, idy), n_warmup=2, n_runs=500)
    benchmark_v2.group_by(group_name)

@pytest.mark.parametrize("batch_size", [64, 128])
@pytest.mark.parametrize("image_size", [224]) 
@pytest.mark.parametrize("n_pairs", [2500, 5000]) 
def test_2d_index_select_v4(batch_size: int, image_size: int, n_pairs: int, benchmark_v2):
    device = "cuda"
    patch_size, dim = 16, 768
    num_patches = int((image_size // patch_size) ** 2)

    z: Float[Tensor, "b n d"] = torch.randn(batch_size, num_patches, dim).to(device)
    idy = torch.randint(num_patches, (batch_size, n_pairs)).to(z.device)
    idx = torch.randint(num_patches, (batch_size, n_pairs)).to(z.device)

    

    group_name = f"Benchmark[B: {batch_size}, I: {image_size}, P: {n_pairs}]"
    benchmark_v2.benchmark(index_select_2d_v4, args=(z, idx, idy), n_warmup=2, n_runs=500)
    benchmark_v2.group_by(group_name)

@pytest.mark.parametrize("batch_size", [64, 128])
@pytest.mark.parametrize("image_size", [224]) 
@pytest.mark.parametrize("n_pairs", [2500, 5000]) 
def test_2d_index_select_v5(batch_size: int, image_size: int, n_pairs: int, benchmark_v2):
    device = "cuda"
    patch_size, dim = 16, 768
    num_patches = int((image_size // patch_size) ** 2)

    z: Float[Tensor, "b n d"] = torch.randn(batch_size, num_patches, dim).to(device)
    idy = torch.randint(num_patches, (batch_size, n_pairs)).to(z.device)
    idx = torch.randint(num_patches, (batch_size, n_pairs)).to(z.device)

    

    group_name = f"Benchmark[B: {batch_size}, I: {image_size}, P: {n_pairs}]"
    benchmark_v2.benchmark(index_select_2d_v5, args=(z, idx, idy), n_warmup=2, n_runs=500)
    benchmark_v2.group_by(group_name)


@pytest.mark.parametrize("batch_size", [64, 128])
@pytest.mark.parametrize("image_size", [224]) 
@pytest.mark.parametrize("n_pairs", [2500, 5000]) 
def test_2d_gather(batch_size: int, image_size: int, n_pairs: int, benchmark_v2):
    device = "cuda"
    patch_size, dim = 16, 768
    num_patches = int((image_size // patch_size) ** 2)

    z: Float[Tensor, "b n d"] = torch.randn(batch_size, num_patches, dim).to(device)
    idy = torch.randint(num_patches, (batch_size, n_pairs)).to(z.device)
    idx = torch.randint(num_patches, (batch_size, n_pairs)).to(z.device)

    group_name = f"Benchmark[B: {batch_size}, I: {image_size}, P: {n_pairs}]"
    benchmark_v2.benchmark(func_gather, args=(z, idx, idy), n_warmup=2, n_runs=500)
    benchmark_v2.group_by(group_name)



@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("image_size, n_pairs", [(32, 2), (64, 4), (128, 8)])
def test_equivalence(batch_size: int, image_size: int, n_pairs):
    device = "cpu"
    patch_size, dim = 8, 5
    num_patches = int((image_size // patch_size) ** 2)

    z: Float[Tensor, "b n d"] = torch.randint(0, 10, (batch_size, num_patches, dim)).to(device)
    idx = torch.randint(num_patches, (batch_size, n_pairs)).to(z.device)
    jdx = torch.randint(num_patches, (batch_size, n_pairs)).to(z.device)

    res1 = func_gather(z, idx, jdx)
    res2 = index_select_2d(z, idx, jdx)
    res3 = index_select_2d_v2(z, idx, jdx)
    res4 = index_select_2d_v3(z, idx, jdx)
    res5 = index_select_2d_v4(z, idx, jdx)
    res5 = index_select_2d_v5(z, idx, jdx)
    assert torch.allclose(res1, res2)
    assert torch.allclose(res1, res3)
    assert torch.allclose(res1, res4)
    assert torch.allclose(res1, res5)