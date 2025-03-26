# TODO: maybe migrate to pytest
import torch
from models.components.utils.offgrid_pos_embed import (
    get_2d_sincos_pos_embed,
    get_canonical_coords,
    PosEmbedDynamicDiff,
    PosEmbedStaticDiff,
    PosEmbedDynamicSame,
    PosEmbedStaticSame,
)

def main():
    # Settings
    patch_size = 16
    grid_size = (14, 14)
    B = 2
    embed_dim = 768
    N = grid_size[0] * grid_size[1]
    device = 'cpu'

    # Generate canonical coordinates using get_canonical_coords
    canonical_coords = get_canonical_coords(grid_size, patch_size, device)  # shape (N,2)
    
    # Compute baseline embedding using the original method
    baseline = get_2d_sincos_pos_embed(canonical_coords, embed_dim)  # shape (N, embed_dim)
    baseline = baseline.expand(B, -1, -1) 

    # Create input with zeros so output = positional encoding
    x = torch.zeros(B, N, embed_dim)

    # Prepare coordinates for homogeneous (same) methods and heterogeneous (diff) methods
    coords_same = canonical_coords                # shape (N,2)
    coords_diff = canonical_coords.unsqueeze(0).expand(B, -1, -1)  # shape (B, N,2)

    # Instantiate each module
    modules = {
        "PosEmbedDynamicSame": (PosEmbedDynamicSame(embed_dim), coords_same),
        "PosEmbedStaticSame": (PosEmbedStaticSame(embed_dim, patch_size, grid_size), coords_same),
        "PosEmbedDynamicDiff": (PosEmbedDynamicDiff(embed_dim, use_cache=False), coords_diff),
        "PosEmbedStaticDiff": (PosEmbedStaticDiff(embed_dim, patch_size, grid_size), coords_diff),
    }
    
    for name, (module, coords) in modules.items():
        module.eval()
        with torch.no_grad():
            out = module(x, coords)  # output = x + pos_emb; here x==0 so pos_emb = out
        # For heterogeneous methods, compare only the first element in batch.
        # assert torch.allclose(res, baseline, atol=1e-5), f"{name} test failed!"
        if torch.allclose(out, baseline, atol=1e-5):
            print(f"{name} test passed.")
        else:
            print(f"{name} test failed!")

    
if __name__ == "__main__":
    main()
