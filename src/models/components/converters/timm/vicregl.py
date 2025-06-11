from typing import Any
from timm.models.convnext import checkpoint_filter_fn
import timm
import torch


def preprocess(state_dict, model_name: str = "convnext_small") -> dict[str, Any]:
    state_dict = state_dict.get("model", state_dict)
    model = timm.create_model(
        model_name, pretrained=False, num_classes=0, ls_init_value=1
    )
    out = checkpoint_filter_fn(state_dict, model)
    renamed: dict[str, Any] = dict()
    for k, v in out.items():
        if k.startswith("head.norm."):
            renamed[k.replace("head.norm.", "norm_pre.")] = v
        else:
            renamed[k] = v
    return renamed


if __name__ == "__main__":
    import timm
    version = "base" # or "small"
    # url = "https://dl.fbaipublicfiles.com/vicregl/convnext_small_alpha0.75.pth"
    url = f"https://dl.fbaipublicfiles.com/vicregl/convnext_{version}_alpha0.75.pth"
    repo = torch.hub.load(
        "facebookresearch/VICRegL", f"convnext_{version}_alpha0p75", source="github"
    ).eval()
    state_dict = torch.hub.load_state_dict_from_url(
        url, map_location="cpu", check_hash=False
    )
    timm_model = timm.create_model(
        f"convnext_{version}",
        head_norm_first=True,
        num_classes=0,
        ls_init_value=1,  # Use default initialization
        pretrained_strict=True,
        pretrained=True,
        pretrained_cfg_overlay=dict(
            state_dict=preprocess(state_dict, f"convnext_{version}"),
        ),
    )
    timm_model.eval()

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        repo_tokens, _ = repo.forward_features(x)
        feats = timm_model.forward_features(x)
        b, c, h, w = feats.shape
        print(repo_tokens.shape, feats.shape)
        timm_tokens = feats.flatten(2, 3).transpose(1, 2)

    assert torch.allclose(repo_tokens, timm_tokens, atol=1e-5, rtol=1e-5)
