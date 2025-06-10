from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from timm.models.convnext import convnext_small, checkpoint_filter_fn
import torch
try:  # jaxtyping requires jax, which may be missing
    from jaxtyping import PyTree
except Exception:  # pragma: no cover - optional dependency
    from typing import Any as PyTree


def process_vicregl_convnext_v2(state_dict: Mapping[str, Any]) -> PyTree:
    """Convert a VICRegL ConvNeXt checkpoint to timm format.

    :param state_dict: Checkpoint dictionary from VICRegL.
    :returns: remapped state dictionary suitable for a timm ConvNeXt model.
    """

    state_dict = state_dict.get("model", state_dict)
    model = convnext_small(pretrained=False, num_classes=0, head_norm_first=True, ls_init_value=1)
    out = checkpoint_filter_fn(state_dict, model)
    renamed: dict[str, Any] = {}
    # renamed = out
    for k, v in out.items():
        if k.startswith("head.norm."):
            renamed[k.replace("head.norm.", "norm_pre.")] = v
        else:
            renamed[k] = v

    for k, v in model.state_dict().items():
        if k not in renamed:
            if k.endswith("gamma"):
                renamed[k] = torch.ones_like(v)
            else:
                renamed[k] = v
    return renamed


def process_vicregl_convnext(state_dict: Mapping[str, Any]) -> PyTree:
    """Convert a VICRegL ConvNeXt checkpoint to timm format.

    :param state_dict: Checkpoint dictionary from VICRegL.
    :returns: remapped state dictionary suitable for a timm ConvNeXt model.
    """

    return state_dict
