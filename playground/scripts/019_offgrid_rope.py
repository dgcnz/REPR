import torch
import math
from offgrid_rope import RoPEAttention
import time

# Settings
B = 64                         # Batch size
# Assume first token is special; total tokens = 1 + grid tokens, e.g. 1 + 196 = 197.
N = 197                        
embed_dim = 768
num_heads = 8
n_iter = 100  # number of iterations per benchmark
max_coord = 224.0  # maximum coordinate value (example)

# Dummy input tokens
dummy_x = torch.randn(B, N, embed_dim).cuda()

# Instantiate and compile the RoPEAttention module
rope = RoPEAttention(dim=embed_dim, num_heads=num_heads, rope_theta=10.0, rope_mixed=True).cuda()
rope = torch.compile(rope)

def benchmark(module, coords, label):
    # Warmup
    _ = module(dummy_x, coords)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iter):
        _ = module(dummy_x, coords)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    avg_time = elapsed / n_iter * 1e3  # in milliseconds
    print(f"{label}: {avg_time:.2f} ms per iteration")

# Ongrid: no coordinates provided (uses default grid positions)
benchmark(rope, None, "Ongrid RoPE")

# Offgrid: pass random offgrid coordinates for all tokens except the first (shape: [B, N-1, 2])
offgrid_coords = torch.empty(B, N-1, 2).uniform_(0, max_coord).cuda()
benchmark(rope, offgrid_coords, "Offgrid RoPE")

# Optional: use torch.cuda.synchronize() between tests if needed.
