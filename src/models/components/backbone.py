import torch.nn as nn
import timm
from timm.models.vision_transformer import VisionTransformer

class Backbone(nn.Module):
    def __init__(self, timm_kwargs: dict):
        super().__init__()
        self.net = timm.create_model(**timm_kwargs)


# def get_pretrained_timm_backbone()