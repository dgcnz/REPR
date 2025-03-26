import time
import torch
import math

# Import the two implementations
from rope_complex import RoPEAttention as RoPEComplex
from rope_sincos import RoPEAttention as RoPENoComplex

# Settings: ensure same dimensions and parameters
B = 32              # batch size
N = 197             # total tokens (including a special token)
embed_dim = 768
num_heads = 8
rope_theta = 10.0
n_iter = 100

# Dummy input tokens
x = torch.randn(B, N, embed_dim).cuda()

# Instantiate both modules with rope_mixed enabled.
rope_complex = RoPEComplex(dim=embed_dim, num_heads=num_heads, rope_theta=rope_theta, rope_mixed=True).cuda()
rope_nocomplex = RoPENoComplex(dim=embed_dim, num_heads=num_heads, rope_theta=rope_theta, rope_mixed=True).cuda()

# Copy state dict with reshaping of the freqs parameter.
state_dict = rope_complex.state_dict()
# Expected shape: [2, num_heads, (embed_dim//num_heads)//2] (i.e., [2,8,48])
if "freqs" in state_dict:
    state_dict["freqs"] = state_dict["freqs"].view(2, num_heads, (embed_dim // num_heads) // 2)
rope_nocomplex.load_state_dict(state_dict)

# Run a forward pass to compare outputs.
with torch.no_grad():
    out_complex = rope_complex(x)
    out_nocomplex = rope_nocomplex(x)
    diff = torch.abs(out_complex - out_nocomplex).max().item()
    print(f"Max absolute difference between implementations: {diff:.6f}")

# Benchmark function
def benchmark(module, label):
    # Warmup
    _ = module(x)
    torch.cuda.synchronize()
    start = time.time()
    for i in range(n_iter):
        _ = module(x)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    avg_time = elapsed / n_iter * 1e3  # ms per iteration
    print(f"{label} latency: {avg_time:.2f} ms per iteration")

benchmark(rope_complex, "RoPEComplex")
benchmark(rope_nocomplex, "RoPENoComplex")
