import random
from PIL import ImageOps
from src.data.components.transforms.parameter_free import make_parameter_free

# from: https://github.com/facebookresearch/dino/blob/main/utils.py#L594
class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


@make_parameter_free
class ParametrizedSolarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __call__(self, img):
        return ImageOps.solarize(img)