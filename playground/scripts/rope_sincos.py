import torch
import torch.nn as nn
import math
from functools import partial

def init_2d_freqs(dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
    # This function initializes learnable frequencies for each head.
    freqs_x = []
    freqs_y = []
    # Instead of using dim directly, note that only dim//2 unique frequencies are used per head.
    mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    for i in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)
        fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi/2 + angles)], dim=-1)
        fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi/2 + angles)], dim=-1)
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)  # [num_heads, dim]
    freqs_y = torch.stack(freqs_y, dim=0)  # [num_heads, dim]
    # The two parts are stacked so that later we can use them to compute rotation angles.
    freqs = torch.stack([freqs_x, freqs_y], dim=0)  # [2, num_heads, dim]
    return freqs

def init_t_xy(end_x: int, end_y: int):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode='floor').float()
    return t_x, t_y

def compute_mixed_cos_sin(freqs: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor, num_heads: int):
    """
    Instead of computing complex numbers via torch.polar,
    compute cos and sin parts for each head.
    freqs: Tensor of shape [2, num_heads, d] where d is head_dim.
    t_x, t_y: tensors of shape [N] (where N=end_x*end_y)
    Returns: (cos, sin), each of shape [num_heads, N, d]
    """
    N = t_x.shape[0]
    # Replace matrix multiplication with elementwise multiplication:
    freqs_x = t_x.unsqueeze(-1).unsqueeze(-1) * freqs[0].unsqueeze(0)  # [N, num_heads, d]
    freqs_y = t_y.unsqueeze(-1).unsqueeze(-1) * freqs[1].unsqueeze(0)  # [N, num_heads, d]
    angle = freqs_x + freqs_y  # [N, num_heads, d]
    angle = angle.permute(1, 0, 2)  # [num_heads, N, d]
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    return cos, sin

def compute_axial_cos_sin(dim: int, end_x: int, end_y: int, theta: float = 100.0):
    # Compute separate frequency tables for x and y.
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    t_x, t_y = init_t_xy(end_x, end_y)  # [N] each
    # Compute outer products
    ax_x = torch.outer(t_x, freqs_x)
    ax_y = torch.outer(t_y, freqs_y)
    # FIX: Flip the y-axis contributions to match complex implementation
    ax_y = ax_y.flip(0)
    # Instead of concatenating, interleave the results
    N = ax_x.shape[0]
    half = ax_x.shape[1]
    total = half * 2
    cos = torch.empty(N, total, device=ax_x.device)
    sin = torch.empty(N, total, device=ax_x.device)
    cos[:, 0::2] = torch.cos(ax_x)
    cos[:, 1::2] = torch.cos(ax_y)
    sin[:, 0::2] = torch.sin(ax_x)
    sin[:, 1::2] = torch.sin(ax_y)
    return cos, sin

def reshape_for_broadcast(tensor: torch.Tensor, x: torch.Tensor):
    # Reshape the precomputed cos or sin tensor for proper broadcasting over x.
    ndim = x.ndim
    if tensor.shape == (x.shape[-2], x.shape[-1]):
        shape = [1] * (ndim - 2) + [x.shape[-2], x.shape[-1]]
    elif tensor.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [1] * (ndim - 3) + [x.shape[-3], x.shape[-2], x.shape[-1]]
    else:
        shape = tensor.shape
    return tensor.view(*shape)

def apply_rotary_emb_no_complex(xq: torch.Tensor, xk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    Applies rotary embeddings to xq and xk without using complex numbers.
    Both xq and xk are assumed to have shape [B, num_heads, seq_len, head_dim],
    where head_dim is even.
    The rotation is applied to all tokens except possibly a special token (if desired).
    """
    def rotate(x, cos, sin):
        # Split the last dimension into two halves (each pair corresponds to one complex number)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        # Compute the rotated coordinates:
        out_even = x_even * cos - x_odd * sin
        out_odd  = x_even * sin + x_odd * cos
        # Now interleave the two halves along the last dimension.
        # One way is to stack and then reshape.
        out = torch.stack((out_even, out_odd), dim=-1)
        return out.flatten(-2)
    xq_out = rotate(xq, cos, sin)
    xk_out = rotate(xk, cos, sin)
    return xq_out, xk_out

class Attention(nn.Module):
    # A standard multi-head attention block.
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (self.qkv(x)
               .reshape(B, N, 3, self.num_heads, C // self.num_heads)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class RoPEAttention(Attention):
    """
    Multi-head attention block with rotary position embeddings,
    implemented without using complex numbers.
    """
    def __init__(self, *args, rope_theta=10.0, rope_mixed=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.rope_mixed = rope_mixed        
        head_dim = self.dim // self.num_heads
        
        if self.rope_mixed:
            # For the mixed-frequency (learnable) version,
            # precompute a parameter tensor of frequencies.
            self.compute_cos_sin = partial(compute_mixed_cos_sin, num_heads=self.num_heads)
            freqs = init_2d_freqs(dim=head_dim, num_heads=self.num_heads, theta=rope_theta, rotate=True)
            # Flatten the two parts so that freqs has shape [2, num_heads, head_dim]
            self.freqs = nn.Parameter(freqs, requires_grad=True)
            t_x, t_y = init_t_xy(end_x=14, end_y=14)
            self.register_buffer('freqs_t_x', t_x)
            self.register_buffer('freqs_t_y', t_y)
        else:
            # For the axial version (non-learnable), precompute cos/sin once.
            self.compute_cos_sin = partial(compute_axial_cos_sin, dim=head_dim, theta=rope_theta)
            cos, sin = self.compute_cos_sin(end_x=14, end_y=14)
            self.register_buffer('cos', cos)
            self.register_buffer('sin', sin)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = (self.qkv(x)
               .reshape(B, N, 3, self.num_heads, C // self.num_heads)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Remove initial scaling - will be applied after rotary
        # q = q * self.scale  

        q_tokens, k_tokens = q[:, :, 1:], k[:, :, 1:]
        w = h = math.sqrt(x.shape[1] - 1)
        if self.rope_mixed:
            t_x, t_y = self.freqs_t_x, self.freqs_t_y
            if t_x.shape[0] != x.shape[1] - 1:
                t_x, t_y = init_t_xy(end_x=int(w), end_y=int(h))
                t_x, t_y = t_x.to(x.device), t_y.to(x.device)
            cos, sin = self.compute_cos_sin(self.freqs, t_x, t_y)
        else:
            cos, sin = self.cos, self.sin
            if cos.shape[0] != x.shape[1] - 1:
                cos, sin = self.compute_cos_sin(end_x=int(w), end_y=int(h))
                cos, sin = cos.to(x.device), sin.to(x.device)

        cos = reshape_for_broadcast(cos, q_tokens)
        sin = reshape_for_broadcast(sin, q_tokens)
        q_rot, k_rot = apply_rotary_emb_no_complex(q_tokens, k_tokens, cos, sin)
        q = torch.cat((q[:, :, :1], q_rot), dim=2)
        k = torch.cat((k[:, :, :1], k_rot), dim=2)

        # Apply scaling after rotary, matching complex implementation
        q = q * self.scale
        
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
