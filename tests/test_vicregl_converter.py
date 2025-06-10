
import torch
import timm

from src.models.components.converters.timm.vicregl import process_vicregl_convnext


def test_vicregl_process_convnext():
    model = timm.create_model(
        "convnext_small",
        pretrained=False,
        num_classes=0,
        head_norm_first=True,
    )
    sample_sd = {
        "model": {
            "visual.trunk.stem.0.weight": model.state_dict()["stem.0.weight"].clone(),
            "visual.head.proj.weight": torch.randn(1, model.num_features),
        }
    }
    out = process_vicregl_convnext(sample_sd)
    assert "stem.0.weight" in out
    assert out["stem.0.weight"].shape == model.state_dict()["stem.0.weight"].shape
    assert "head.fc.weight" in out
    assert out["head.fc.bias"].shape[0] == out["head.fc.weight"].shape[0]


def test_vicregl_process_convnext_real():
    url = "https://dl.fbaipublicfiles.com/vicregl/convnext_small_alpha0.75.pth"
    state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu", check_hash=False)
    out = process_vicregl_convnext(state_dict)
    model = timm.create_model(
        "convnext_small",
        pretrained=False,
        num_classes=0,
        head_norm_first=True,
    )
    missing, unexpected = model.load_state_dict(out, strict=False)
    assert not missing and not unexpected


import pytest


def test_vicregl_feature_equivalence():
    url = "https://dl.fbaipublicfiles.com/vicregl/convnext_small_alpha0.75.pth"
    repo = torch.hub.load(
        "facebookresearch/VICRegL", "convnext_small_alpha0p75", source="github"
    ).eval()
    timm_model = timm.create_model(
        "convnext_small",
        pretrained=False,
        num_classes=0,
        head_norm_first=True,
    )
    state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu", check_hash=False)
    timm_sd = process_vicregl_convnext(state_dict)
    timm_model.load_state_dict(timm_sd, strict=False)
    timm_model.eval()

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        repo_tokens, _ = repo.forward_features(x)
        feats = timm_model.forward_features(x)
        if isinstance(feats, tuple):
            feats = feats[0]
        b, c, h, w = feats.shape
        timm_tokens = feats.flatten(2).transpose(1, 2)

    assert torch.allclose(repo_tokens, timm_tokens, atol=1e-5, rtol=1e-5)
