import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


def to_2tuple(x):
    return (x, x) if isinstance(x, int) else x


# -------------------------------
# Provided Patch Embedding Modules
# -------------------------------
class Conv2DPatchEmbed(nn.Module):
    """2D Image to Patch Embedding using Conv2d."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        bias=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        x = self.norm(x)
        return x


class LinearPatchEmbed(nn.Module):
    """
    Linear patch embedding using nn.Unfold.
    Expects flattened patches of shape [B, num_patches, C*patch_size*patch_size].
    """

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Linear(patch_size * patch_size * in_channels, embed_dim)

    def forward(self, x):
        # x is expected to be of shape [B, num_patches, C*patch_size*patch_size]
        return self.proj(x)


# -------------------------------
# Original Methods
# -------------------------------
class OffGridPatchEmbedStitching(nn.Module):
    """
    Original method using off-grid sampling with restitching into an image,
    then applying Conv2DPatchEmbed.
    """

    def __init__(
        self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, mode="offgrid"
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.mode = mode
        self.patch_embed = Conv2DPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

    def _sample_offgrid(self, B, H, W, patch_size, device):
        nW = W // patch_size
        nH = H // patch_size
        xs = torch.randint(0, W - patch_size, (B, nH * nW), device=device)
        ys = torch.randint(0, H - patch_size, (B, nH * nW), device=device)
        return ys, xs

    def _create_sampled_grid_flattened(self, patch_size, H, W, ys, xs):
        B, nH, nW = ys.size(0), H // patch_size, W // patch_size
        zs = ys * W + xs  # base indices for each patch.
        dx = dy = torch.arange(patch_size, device=ys.device)
        d = W * dy.unsqueeze(-1) + dx  # [patch_size, patch_size]
        indices = zs[:, :, None, None] + d  # [B, n_patches, patch_size, patch_size]
        indices = indices.view(B, nH, nW, patch_size, patch_size)
        indices = indices.permute(0, 1, 3, 2, 4).reshape(B, H * W)
        return indices

    def sample_and_stitch(self, img):
        B, C, H, W = img.shape
        ys, xs = self._sample_offgrid(B, H, W, self.patch_size, img.device)
        patch_positions = torch.stack([ys, xs], dim=-1)  # [B, n_patches, 2]
        indices = self._create_sampled_grid_flattened(self.patch_size, H, W, ys, xs)
        indices = indices.unsqueeze(1).expand(-1, C, -1)  # [B, C, H*W]
        stitched_img = img.flatten(-2).gather(2, indices).unflatten(-1, (H, W))
        return stitched_img, patch_positions

    def forward(self, x):
        stitched_img, patch_positions = self.sample_and_stitch(x)
        patch_tokens = self.patch_embed(stitched_img)
        return patch_tokens, patch_positions


class OffGridPatchEmbedGather(nn.Module):
    """
    Original method using off-grid sampling via gather and a linear projection.
    """

    def __init__(
        self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, mode="offgrid"
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.mode = mode
        self.patch_embed = LinearPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_chans,
            embed_dim=embed_dim,
        )

    def _sample_offgrid(self, B, H, W, patch_size, device):
        nW = W // patch_size
        nH = H // patch_size
        xs = torch.randint(0, W - patch_size, (B, nW * nH), device=device)
        ys = torch.randint(0, H - patch_size, (B, nW * nH), device=device)
        return ys, xs

    def forward(self, x):
        B, C, H, W = x.shape
        ys, xs = self._sample_offgrid(B, H, W, self.patch_size, x.device)
        patch_positions = torch.stack([ys, xs], dim=-1)  # [B, n_patches, 2]
        num_patches = ys.shape[1]
        device = x.device
        dy = torch.arange(self.patch_size, device=device).unsqueeze(1)
        dx = torch.arange(self.patch_size, device=device).unsqueeze(0)
        patch_delta = (dy * W + dx).view(-1)  # [patch_area]
        base_indices = ys * W + xs  # [B, n_patches]
        patch_indices = base_indices.unsqueeze(-1) + patch_delta.unsqueeze(0).unsqueeze(
            0
        )
        channel_offsets = torch.arange(C, device=device) * (H * W)
        patch_indices = patch_indices.unsqueeze(2) + channel_offsets.view(1, 1, C, 1)
        patch_indices = patch_indices.view(
            B, num_patches, C * self.patch_size * self.patch_size
        )
        img_flat = x.view(B, C * H * W)
        img_flat_expanded = img_flat.unsqueeze(1).expand(B, num_patches, -1)
        patches_flat = torch.gather(img_flat_expanded, 2, patch_indices)
        patch_tokens = self.patch_embed.proj(patches_flat)
        return patch_tokens, patch_positions


# -------------------------------
# Variant 1: Stitching with Linear Projection
# Instead of reshaping into an image and using Conv2DPatchEmbed,
# we gather flattened patches and feed them to LinearPatchEmbed.
# -------------------------------
class OffGridPatchEmbedStitchingLinear(nn.Module):
    def __init__(
        self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, mode="offgrid"
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.mode = mode
        self.patch_embed = LinearPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_chans,
            embed_dim=embed_dim,
        )

    def _sample_offgrid(self, B, H, W, patch_size, device):
        nW = W // patch_size
        nH = H // patch_size
        xs = torch.randint(0, W - patch_size, (B, nH * nW), device=device)
        ys = torch.randint(0, H - patch_size, (B, nH * nW), device=device)
        return ys, xs

    def _create_sampled_grid_flattened(self, patch_size, H, W, ys, xs):
        # Fix ordering by computing correct row-major offsets.
        B, n_patches = ys.shape[0], ys.shape[1]
        dy = torch.arange(patch_size, device=ys.device).unsqueeze(1)
        dx = torch.arange(patch_size, device=ys.device).unsqueeze(0)
        d = (dy * W + dx).view(-1)  # correct patch offsets
        base = ys * W + xs  # starting index for each patch
        indices = base.unsqueeze(-1) + d.unsqueeze(0)  # [B, n_patches, patch_area]
        indices = indices.view(B, n_patches * patch_size * patch_size)
        return indices

    def forward(self, x):
        B, C, H, W = x.shape
        ys, xs = self._sample_offgrid(B, H, W, self.patch_size, x.device)
        patch_positions = torch.stack([ys, xs], dim=-1)
        n_patches = ys.shape[1]
        patch_area = self.patch_size * self.patch_size
        indices = self._create_sampled_grid_flattened(self.patch_size, H, W, ys, xs)
        indices = indices.unsqueeze(1).expand(B, C, -1)
        patches_flat = x.flatten(-2).gather(2, indices)
        patches_flat = patches_flat.view(B, C, n_patches, patch_area)
        patches_flat = patches_flat.permute(0, 2, 1, 3).reshape(
            B, n_patches, C * patch_area
        )
        patch_tokens = self.patch_embed(patches_flat)
        return patch_tokens, patch_positions


# -------------------------------
# Variant 2: Gather with Conv2D Projection
# Instead of flattening and using LinearPatchEmbed, we reshape the gathered patches
# into an image and feed it to Conv2DPatchEmbed.
# -------------------------------
class OffGridPatchEmbedGatherConv2D(nn.Module):
    def __init__(
        self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, mode="offgrid"
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.mode = mode
        self.patch_embed = Conv2DPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

    def _sample_offgrid(self, B, H, W, patch_size, device):
        nW = W // patch_size
        nH = H // patch_size
        xs = torch.randint(0, W - patch_size, (B, nW * nH), device=device)
        ys = torch.randint(0, H - patch_size, (B, nW * nH), device=device)
        return ys, xs

    def forward(self, x):
        B, C, H, W = x.shape
        ys, xs = self._sample_offgrid(B, H, W, self.patch_size, x.device)
        patch_positions = torch.stack([ys, xs], dim=-1)
        num_patches = ys.shape[1]
        patch_area = self.patch_size * self.patch_size
        base_indices = ys * W + xs
        device = x.device
        dy = torch.arange(self.patch_size, device=device).unsqueeze(1)
        dx = torch.arange(self.patch_size, device=device).unsqueeze(0)
        patch_delta = (dy * W + dx).view(-1)
        patch_indices = base_indices.unsqueeze(-1) + patch_delta.unsqueeze(0).unsqueeze(
            0
        )
        channel_offsets = torch.arange(C, device=device) * (H * W)
        patch_indices = patch_indices.unsqueeze(2) + channel_offsets.view(1, 1, C, 1)
        patch_indices = patch_indices.view(B, num_patches, C * patch_area)
        img_flat = x.view(B, C * H * W)
        img_flat_expanded = img_flat.unsqueeze(1).expand(B, num_patches, -1)
        patches_flat = torch.gather(img_flat_expanded, 2, patch_indices)
        nH = H // self.patch_size
        nW = W // self.patch_size
        patches_img = patches_flat.view(B, nH, nW, C, patch_area)
        patches_img = patches_img.view(B, nH, nW, C, self.patch_size, self.patch_size)
        patches_img = patches_img.permute(0, 3, 1, 4, 2, 5).contiguous()
        restitched_img = patches_img.view(
            B, C, nH * self.patch_size, nW * self.patch_size
        )
        patch_tokens = self.patch_embed(restitched_img)
        return patch_tokens, patch_positions


# -------------------------------
# Pytest Test
# -------------------------------
@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("img_size", [224])
@pytest.mark.parametrize("patch_size", [16])
@pytest.mark.parametrize("in_chans", [3])
@pytest.mark.parametrize("embed_dim", [768])
def test_offgrid_all_equivalence(B, img_size, patch_size, in_chans, embed_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = torch.randn(B, in_chans, img_size, img_size, device=device)

    # Instantiate all four models.
    model_stitching = OffGridPatchEmbedStitching(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        mode="offgrid",
    ).to(device)
    model_gather = OffGridPatchEmbedGather(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        mode="offgrid",
    ).to(device)
    model_stitching_linear = OffGridPatchEmbedStitchingLinear(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        mode="offgrid",
    ).to(device)
    model_gather_conv2d = OffGridPatchEmbedGatherConv2D(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        mode="offgrid",
    ).to(device)

    # Force all projection layers to use the same weights.
    with torch.no_grad():
        # Use model_stitching (conv2d) as reference.
        weight_conv = model_stitching.patch_embed.proj.weight.data.clone()
        bias_conv = (
            model_stitching.patch_embed.proj.bias.data.clone()
            if model_stitching.patch_embed.proj.bias is not None
            else None
        )
        # For conv2d modules:
        model_stitching.patch_embed.proj.weight.data.copy_(weight_conv)
        if bias_conv is not None:
            model_stitching.patch_embed.proj.bias.data.copy_(bias_conv)
        # For the linear module, flatten the conv2d weight.
        model_gather.patch_embed.proj.weight.data.copy_(weight_conv.view(embed_dim, -1))
        if bias_conv is not None:
            model_gather.patch_embed.proj.bias.data.copy_(bias_conv.view(-1))
        model_gather_conv2d.patch_embed.proj.weight.data.copy_(weight_conv)
        if bias_conv is not None:
            model_gather_conv2d.patch_embed.proj.bias.data.copy_(bias_conv)
        # For linear modules: flatten conv2d weight.
        weight_lin = weight_conv.view(embed_dim, -1).clone()
        model_stitching_linear.patch_embed.proj.weight.data.copy_(weight_lin)
        if bias_conv is not None:
            model_stitching_linear.patch_embed.proj.bias.data.copy_(bias_conv.view(-1))

    # Run each model with the same seed so that patch sampling is identical.
    torch.manual_seed(42)
    tokens1, pos1 = model_stitching(img)
    torch.manual_seed(42)
    tokens2, pos2 = model_gather(img)
    torch.manual_seed(42)
    tokens3, pos3 = model_stitching_linear(img)
    torch.manual_seed(42)
    tokens4, pos4 = model_gather_conv2d(img)

    # Check that patch positions are identical across methods.
    for p in [pos1, pos2, pos3, pos4]:
        assert torch.allclose(
            pos1.float(), p.float(), atol=1e-6
        ), "Patch positions differ among methods."

    # Check that the token outputs are nearly identical.
    diff12 = (tokens1 - tokens2).abs().max().item()
    diff13 = (tokens1 - tokens3).abs().max().item()
    diff14 = (tokens1 - tokens4).abs().max().item()
    print("Max token difference between stitching and gather (original):", diff12)
    print(
        "Max token difference between stitching and stitching_linear (variant):", diff13
    )
    print("Max token difference between stitching and gather_conv2d (variant):", diff14)
    tol = 1e-3
    assert torch.allclose(
        tokens1, tokens2, atol=tol
    ), f"Original methods differ; max diff: {diff12}"
    assert torch.allclose(
        tokens1, tokens4, atol=tol
    ), f"Stitching vs. gather_conv2d differ; max diff: {diff14}"
    assert torch.allclose(
        tokens1, tokens3, atol=tol
    ), f"Stitching vs. stitching_linear differ; max diff: {diff13}"


if __name__ == "__main__":
    pytest.main([__file__])
