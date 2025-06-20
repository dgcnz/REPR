from typing import Mapping, Any

from torch import Tensor


def preprocess(state_dict: Mapping[str, Any]) -> dict[str, Tensor]:
    """Filter out keys unrelated to the ViT encoder."""
    state_dict = state_dict.get("model", state_dict)
    filtered: dict[str, Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith(
            (
                "decoder_",
                "patch_embed_c",
                "blocks_c",
                "norm_c",
                "decoder_g_embed_c",
            )
        ):
            continue
        if k.endswith(".num_batches_tracked"):
            continue
        filtered[k] = v
    return filtered


if __name__ == "__main__":
    import timm
    import numpy as np
    from src.models.components.cim import cim_vit_small_patch16
    import torch

    # Patch deprecated numpy attribute for positional embedding initialization
    if not hasattr(np, "float"):
        np.float = np.float64  # type: ignore[attr-defined]

    src = cim_vit_small_patch16().eval()
    ckpt = preprocess({"model": src.state_dict()})

    tgt = timm.create_model(
        "vit_small_patch16_224",
        pretrained=False,
        num_classes=0,
    ).eval()
    tgt.load_state_dict(ckpt, strict=True)

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        # compute features with src model using only the encoder branch
        B, _, H, W = x.shape
        src_tokens = src.patch_embed(x)
        cls_tok = src.cls_token.expand(B, -1, -1)
        src_tokens = torch.cat((cls_tok, src_tokens), dim=1)
        src_tokens = src_tokens + src.interpolate_pos_encoding(src_tokens, W, H)
        for blk in src.blocks:
            src_tokens = blk(src_tokens)
        src_feat = src.norm(src_tokens)

        tgt_feat = tgt.forward_features(x)

    assert torch.allclose(src_feat, tgt_feat, atol=1e-5, rtol=1e-5)
