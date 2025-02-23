import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Fused Triton kernel to compute 2D sincos positional embeddings with autotuning.
#
# This kernel fuses the computation for the x and y sin/cos embeddings.
# For an input tensor "coords_ptr" of shape [N, 2] (with x at index 0 and y at index 1)
# the output is a tensor of shape [N, D]. The first half of each row (D//2)
# corresponds to the embedding for x and the second half to y.
#
# For each half (of length D1 = D//2), for each inner index i:
#   if i is even: value = sin(coord * exp(-log(10000)*(2*(i//2)/D1)))
#   if i is odd : value = cos(coord * exp(-log(10000)*(2*((i-1)//2)/D1)))
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_R': 16, 'BLOCK_C': 64}, num_warps=2),
        triton.Config({'BLOCK_R': 32, 'BLOCK_C': 64}, num_warps=2),
        triton.Config({'BLOCK_R': 16, 'BLOCK_C': 128}, num_warps=2),
        triton.Config({'BLOCK_R': 32, 'BLOCK_C': 128}, num_warps=2),
        triton.Config({'BLOCK_R': 64, 'BLOCK_C': 64}, num_warps=4),
    ],
    key=['N', 'D']
)
@triton.jit
def sincos_2d_fused_kernel(coords_ptr, output_ptr,
                             N: tl.constexpr, D: tl.constexpr, D1: tl.constexpr,
                             BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    LOG_10000 = 9.210340371976184  # precomputed math.log(10000.0)

    # Determine tile indices.
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    row_offsets = row_block * BLOCK_R + tl.arange(0, BLOCK_R)
    col_offsets = col_block * BLOCK_C + tl.arange(0, BLOCK_C)

    # Boundary masks.
    row_mask = row_offsets < N
    col_mask = col_offsets < D

    # Load x and y coordinates from the input tensor.
    # The coordinates are stored in row-major order as [x, y] pairs.
    x = tl.load(coords_ptr + row_offsets * 2 + 0, mask=row_mask, other=0.0)
    y = tl.load(coords_ptr + row_offsets * 2 + 1, mask=row_mask, other=0.0)

    # For each output column, decide if it belongs to x or y embedding.
    is_x = col_offsets < D1
    inner_idx = tl.where(is_x, col_offsets, col_offsets - D1)

    # Determine sin (for even indices) or cos (for odd indices).
    is_even = (inner_idx % 2) == 0
    div_idx = tl.where(is_even, inner_idx // 2, (inner_idx - 1) // 2)

    # Compute the frequency factor for each inner index.
    freq = tl.exp(-LOG_10000 * (2 * tl.cast(div_idx, tl.float32) / tl.cast(D1, tl.float32)))

    # Select coordinate: use x for first half and y for second half.
    coord = tl.where(is_x[None, :], x[:, None], y[:, None])
    angle = coord * freq[None, :]

    # Compute output value: sin for even, cos for odd.
    val = tl.where(is_even[None, :], tl.sin(angle), tl.cos(angle))

    # Calculate linear output indices and store the computed values.
    out_offsets = row_offsets[:, None] * D + col_offsets[None, :]
    tl.store(output_ptr + out_offsets, val, mask=(row_mask[:, None] & col_mask[None, :]))

# ---------------------------------------------------------------------------
# Python wrapper for the fused 2D sincos positional embedding kernel.
# ---------------------------------------------------------------------------
def get_2d_sincos_pos_embed_triton_fused(coords: torch.Tensor, embed_dim: int) -> torch.Tensor:
    """
    coords: Tensor of shape (N, 2) on the target device (e.g., cuda), containing x and y.
    embed_dim: Total embedding dimension (must be even).
    Returns: Tensor of shape (N, embed_dim) with the positional embeddings.
    """
    assert embed_dim % 2 == 0, "Embedding dimension must be even."
    D1 = embed_dim // 2  # Half dimension for each coordinate.
    N = coords.shape[0]
    device = coords.device
    pos_embed = torch.empty((N, embed_dim), device=device, dtype=torch.float32)
    # Define grid dimensions; using candidate BLOCK_R=16 and BLOCK_C=64 as starting points.
    grid_r = triton.cdiv(N, 16)
    grid_c = triton.cdiv(embed_dim, 64)
    grid = (grid_r, grid_c)
    # Call the kernel without explicitly passing BLOCK_R and BLOCK_C.
    sincos_2d_fused_kernel[grid](coords, pos_embed, N, embed_dim, D1)
    return pos_embed

# ---------------------------------------------------------------------------
# Dynamic positional embedding module using the fused Triton kernel.
# ---------------------------------------------------------------------------
class PosEmbedDynamicDiffTriton(nn.Module):
    """
    Computes 2D sinusoidal positional embeddings on-the-fly for each batch element,
    using a fused Triton kernel with autotuning.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        assert embed_dim % 2 == 0, "Embedding dimension must be even."
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x      : Tensor of shape (B, N, embed_dim)
            coords : Tensor of shape (B, N, 2) containing positional coordinates.
        Returns:
            Tensor of shape (B, N, embed_dim) after adding positional embeddings.
        """
        B, N, _ = coords.shape
        coords_flat = coords.reshape(-1, 2)
        pos_embed = get_2d_sincos_pos_embed_triton_fused(coords_flat, self.embed_dim)
        pos_embed = pos_embed.view(B, N, self.embed_dim)
        return x + pos_embed

# ---------------------------------------------------------------------------
# Example usage:
# For a ViT originally using 14x14 tokens, masking reduces the effective token count to ~40.
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    device = torch.device('cuda')
    B = 1
    N = 40  # Effective token count after masking.
    embed_dim = 256  # Must be even.
    
    # Create dummy coordinates (e.g., canonical grid positions or post-masking coordinates).
    coords = torch.rand((N, 2), device=device, dtype=torch.float32)
    
    # Test the fused 2D sincos positional embedding with autotuning.
    pos_embed = get_2d_sincos_pos_embed_triton_fused(coords, embed_dim)
    print("Positional embedding shape:", pos_embed.shape)  # Expected: [N, embed_dim]
    
    # Test the dynamic module.
    x = torch.zeros((B, N, embed_dim), device=device, dtype=torch.float32)
    module = PosEmbedDynamicDiffTriton(embed_dim)
    out = module(x, coords.unsqueeze(0))  # Add batch dimension to coords.
    print("Module output shape:", out.shape)  # Expected: [B, N, embed_dim]
