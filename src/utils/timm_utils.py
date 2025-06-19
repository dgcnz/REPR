import torch
from torch import Tensor
from jaxtyping import Float


def timm_patch_size(model: torch.nn.Module) -> int:
    """Return effective patch size for a timm model.

    :param model: timm model.
    :returns: Patch size mapped to a single token.
    """
    name = model.default_cfg.get("architecture", "")
    if name.startswith("vit_"):
        size = getattr(model.patch_embed, "patch_size", None)
        if isinstance(size, tuple):
            return size[-1]
        if isinstance(size, int):
            return size
        return model.patch_embed.proj.kernel_size[-1]
    elif name.startswith("convnext_"):
        patch = model.stem[0].stride[0]
        for stage in model.stages:
            ds = getattr(stage, "downsample", None)
            if ds is None:
                continue
            for m in ds.modules():
                if isinstance(m, torch.nn.Conv2d):
                    patch *= m.stride[0]
        return patch
    raise ValueError(f"Unsupported timm model: {name}")

def timm_embed_dim(model: torch.nn.Module) -> int:
    """Return embedding dimension for a timm model.
    
    :param model: timm model.
    :returns: Embedding dimension.
    """
    name = model.default_cfg.get("architecture", "")
    if name.startswith("vit_"):
        return model.embed_dim
    elif name.startswith("convnext_"):
        return model.num_features
    raise ValueError(f"Unsupported timm model: {name}")


def forward_features_vit(
    model: torch.nn.Module, imgs: Float[Tensor, "B C H W"]
) -> Float[Tensor, "B D h w"]:
    """Return patch feature map for a timm model.

    :param model: A timm model supporting ``forward_intermediates``.
    :param imgs: Input batch of images.
    :returns: Final feature map in ``NCHW`` format.
    """

    with torch.no_grad():
        feats = model.forward_intermediates(
            imgs,
            indices=[-1],
            return_prefix_tokens=False,
            output_fmt="NCHW",
            intermediates_only=True,
        )[0]
    return feats


def forward_features_convnext(
    model: torch.nn.Module, imgs: Float[Tensor, "B C H W"]
) -> Float[Tensor, "B D h w"]:
    """Return patch feature map for a convnext timm model.
    
    :param model: A timm model supporting ``forward_intermediates``.
    :param imgs: Input batch of images.
    :returns: Final feature map in ``NCHW`` format.
    """ 
    with torch.no_grad():
        feats = model.forward_intermediates(
            imgs,
            indices=[-1],
            output_fmt="NCHW",
            intermediates_only=True,
        )[0]
        feats = model.norm_pre(feats)
    return feats

