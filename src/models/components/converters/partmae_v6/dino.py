from __future__ import annotations

import logging
import timm
from src.models.components.partmae_v6 import PARTMaskedAutoEncoderViT


class PARTMaskedAutoEncoderViTFromDINO(PARTMaskedAutoEncoderViT):
    """Load a timm DINO ViT backbone into :class:`PARTMaskedAutoEncoderViT`.

    If ``state_dict`` is ``None`` this will download the pretrained
    ``vit_small_patch16_224.dino`` weights from ``timm`` and use them as
    initialization for the encoder.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("embed_dim", 384)
        kwargs.setdefault("depth", 12)
        kwargs.setdefault("num_heads", 6)
        super().__init__(*args, **kwargs)

    def load_state_dict(self, state_dict=None, strict: bool = True):
        if state_dict is None:
            backbone = timm.create_model(
                "vit_small_patch16_224.dino", pretrained=True, num_classes=0
            )
            state_dict = backbone.state_dict()
        miss, unex = super().load_state_dict(state_dict, strict=False)
        if self.dino_head is None:
            unex = [k for k in unex if not k.startswith("dino_head.")]
        allowed_prefixes = [
            "dino_head.",
            "decoder",
            "mask_pos_token",
            "segment_embed",
            "pose_head.linear.weight",
            "_patch_loss.sigma",
        ]
        other_miss = [
            k for k in miss if not any(k.startswith(p) for p in allowed_prefixes)
        ]
        if other_miss:
            logging.warning("Missing keys: %s", other_miss)
        if unex:
            logging.warning("Unexpected keys: %s", unex)
        if strict:
            assert not other_miss, f"missing keys: {other_miss}"
            assert not unex, f"unexpected keys: {unex}"
        return miss, unex
