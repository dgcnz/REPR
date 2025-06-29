import timm
from mmengine.model import BaseModule
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmseg.registry import MODELS

@MODELS.register_module()
class TIMMViT(BaseModule):
    """TIMM ViT wrapper using forward_intermediates (no manual CLS‐drop needed)."""
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        in_channels: int = 3,
        out_indices: tuple[int, ...] = (0, 1, 2, 3),
        freeze: bool = True,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg)

        # remap norm_layer if provided
        if "norm_layer" in kwargs and isinstance(kwargs["norm_layer"], str):
            kwargs["norm_layer"] = MMENGINE_MODELS.get(kwargs["norm_layer"])

        # instantiate the full ViT (no features_only)
        self.vit = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            **kwargs,
        )
        self._out_indices = self.vit.prune_intermediate_layers(
            indices=out_indices,
            prune_head=True,
        )

        # freeze backbone if requested
        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False

        if pretrained:
            self._is_init = True
        
        assert self._is_init, "In normal circumstances, TIMMViT should be initialized with a pretrained model."

    def forward(self, x):
        feats = self.vit.forward_intermediates(
            x,
            indices=self._out_indices,
            return_prefix_tokens=False,
            intermediates_only=True,
            output_fmt='NCHW',
        )
        return feats
