from __future__ import annotations

from typing import Mapping

import timm
import torch
from jaxtyping import Float
from torch import Tensor

__all__ = ["preprocess"]


def preprocess(state_dict: Mapping[str, Float[Tensor, "*"]]) -> dict[str, Float[Tensor, "*"]]:
    """Convert a VICReg checkpoint to a timm-friendly format.

    :param state_dict: Original state dictionary from the VICReg repository.
    :returns: Dictionary compatible with timm ResNet models.
    """
    state_dict = state_dict.get("model", state_dict)
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


if __name__ == "__main__":
    repo = torch.hub.load("facebookresearch/vicreg:main", "resnet50").eval()
    timm_model = timm.create_model("resnet50", num_classes=0, pretrained=False)
    timm_model.load_state_dict(preprocess(repo.state_dict()))
    timm_model.eval()

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        repo_out = repo(x)
        timm_out = timm_model(x)

    assert repo_out.shape == timm_out.shape
    assert torch.allclose(repo_out, timm_out, atol=1e-5, rtol=1e-5)
