import time
import torch
import math

# ...existing imports...
from rope_complex import RoPEAttention as RoPEComplex
from rope_sincos import RoPEAttention as RoPENoComplex

# Settings: axial mode (rope_mixed=False)
B = 32              # batch size
N = 197             # total tokens (including a special token)
embed_dim = 768
num_heads = 8
rope_theta = 10.0
n_iter = 100

# Dummy input tokens
x = torch.randn(B, N, embed_dim).cuda()

# Instantiate both modules with axial mode (rope_mixed disabled).
rope_complex = RoPEComplex(dim=embed_dim, num_heads=num_heads, rope_theta=rope_theta, rope_mixed=False).cuda()
rope_nocomplex = RoPENoComplex(dim=embed_dim, num_heads=num_heads, rope_theta=rope_theta, rope_mixed=False).cuda()

# Run a forward pass to compare outputs.
with torch.no_grad():
    out_complex = rope_complex(x)
    out_nocomplex = rope_nocomplex(x)
    diff = torch.abs(out_complex - out_nocomplex).max().item()
    print(f"Axial mode - Max absolute difference: {diff:.6f}")

# Benchmark function
def benchmark(module, label):
    # Warmup
    _ = module(x)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iter):
        _ = module(x)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    avg_time = elapsed / n_iter * 1e3  # ms per iteration
    print(f"{label} latency (axial): {avg_time:.2f} ms per iteration")

benchmark(rope_complex, "RoPEComplex")
benchmark(rope_nocomplex, "RoPENoComplex")
