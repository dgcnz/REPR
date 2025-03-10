import random
from PIL import ImageFilter
from src.data.components.transforms.parameter_free import make_parameter_free
import numpy as np


@make_parameter_free
class ParametrizedPILGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, radius_min=0.1, radius_max=2.0):
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
