import os
os.environ['WANDB_SILENT'] = 'true'
os.environ['WANDB_DISABLED'] = 'true'

import torch

from src.models.components.converters.partmae_v6.dino import (
    PARTMaskedAutoEncoderViTFromDINO,
)


def test_dino_converter_loads():
    torch.manual_seed(0)
    model = PARTMaskedAutoEncoderViTFromDINO()
    before = model.patch_embed.proj.weight.clone()
    miss, unex = model.load_state_dict(None, strict=True)
    after = model.patch_embed.proj.weight
    assert not unex
    assert not torch.allclose(before, after)
