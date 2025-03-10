# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from logging import getLogger

import numpy as np
import torchvision.transforms as transforms
from torch import nn, Tensor
import parameterized_transforms.transforms as ptx
import parameterized_transforms.wrappers as ptw
import parameterized_transforms.core as ptc
from src.data.components.transforms.pil_gaussian_blur import (
    PILRandomGaussianBlur,
    ParametrizedPILGaussianBlur,
)
from jaxtyping import Int, Float
import torch

logger = getLogger()


class MultiCrop(nn.Module):
    def __init__(
        self,
        size_crops,
        nmb_crops,
        min_scale_crops,
        max_scale_crops,
    ):
        super().__init__()
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)

        color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]
        trans = []
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            trans.extend(
                [
                    transforms.Compose(
                        [
                            randomresizedcrop,
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.Compose(color_transform),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=mean, std=std),
                        ]
                    )
                ]
                * nmb_crops[i]
            )
        self.trans = trans

    def forward(self, image):
        multi_crops = list(map(lambda trans: trans(image), self.trans))
        return multi_crops


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


class ParametrizedMultiCrop(object):
    def __init__(
        self,
        size_crops: int,
        nmb_crops: int,
        min_scale_crops: float,
        max_scale_crops: float,
    ):
        super().__init__()
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)

        self.size_crops = size_crops
        self.nmb_crops = nmb_crops
        self.min_scale_crops = min_scale_crops
        self.max_scale_crops = max_scale_crops

        color_transform = [get_color_distortion(), ParametrizedPILGaussianBlur()]
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]
        trans = []

        color_transform = [get_color_distortion_pt()]
        for i in range(len(size_crops)):
            randomresizedcrop = ptx.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            trans.extend(
                ptw.CastParamsToTensor(
                    transform=ptx.Compose(
                        [
                            randomresizedcrop,
                            ptx.RandomHorizontalFlip(p=0.5),
                            ptx.Compose(color_transform),
                            ptx.ToTensor(),
                            ptx.Normalize(mean=mean, std=std),
                        ]
                        * nmb_crops[i]
                    )
                )
            )

    def __call__(self, image: Tensor) -> list[Tensor]:
        multi_crops = list(map(lambda trans: trans(image), self.trans))
        return multi_crops


def restore_coordinates(
    coords: Int[Tensor, "B 2"], params: Tensor, crop_size: int
) -> Float[Tensor, "B 2"]:
    """
    Convert coordinates from the transformed image to the original image.

    :param coords: Coordinates in the transformed image, shape (B, 2) as (y, x).
    :param params: A tensor containing at least 5 parameters [y, x, h, w, flip, ...].
    :param crop_size: The size of the crop (e.g., 160).
    :return: A tensor of shape (B, 2) containing the coordinates in the original image.
    """
    # Ensure float type and correct device.
    params = params.float()
    coords = coords.float()

    # Extract the top-left corner and size of the crop.
    crop_origin = params[:2]  # [y, x] of the crop in the original image.
    crop_hw = params[2:4]  # [h, w] of the crop.
    flip = params[4]  # 0: no flip, 1: horizontal flip.

    # Define the offset: if flip is applied, adjust x coordinate accordingly.
    # Only x is affected by horizontal flip.
    flip_offset = torch.tensor(
        [0.0, flip * crop_hw[1]], dtype=coords.dtype, device=coords.device
    )
    offset = crop_origin + flip_offset

    # Compute the sign: keep y as is and for x, flip if needed.
    sign = torch.tensor([1.0, 1.0], dtype=coords.dtype, device=coords.device)
    if flip.item() >= 0.5:
        sign[1] = -1.0

    # Scale coordinates from crop space to original crop dimensions.
    scaled = coords * (crop_hw / crop_size)

    # Apply flip (if any) and add the offset.
    restored = scaled * sign + offset
    return restored


def get_color_distortion_pt(s=1.0):
    # s is the strength of color distortion.
    color_jitter = ptx.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = ptx.RandomApply([color_jitter], p=0.8)
    rnd_gray = ptx.RandomGrayscale(p=0.2)
    color_distort = ptx.Compose([rnd_color_jitter, rnd_gray])
    return color_distort

