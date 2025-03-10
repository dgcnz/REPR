from PIL import Image
from jaxtyping import Float, Int
from torch import Tensor
import torch
from torchvision import transforms
from src.data.components.transforms.pil_gaussian_blur import (
    PILRandomGaussianBlur,
    ParametrizedPILGaussianBlur,
)
from src.data.components.transforms.solarization import (
    Solarization,
    ParametrizedSolarization,
)
import parameterized_transforms.transforms as ptx
import parameterized_transforms.wrappers as ptw
import parameterized_transforms.core as ptc


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )
        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        # first global crop
        self.global_transfo1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224, scale=global_crops_scale, interpolation=Image.BICUBIC
                ),
                flip_and_color_jitter,
                PILRandomGaussianBlur(1.0),
                normalize,
            ]
        )
        # second global crop
        self.global_transfo2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224, scale=global_crops_scale, interpolation=Image.BICUBIC
                ),
                flip_and_color_jitter,
                PILRandomGaussianBlur(0.1),
                Solarization(0.2),
                normalize,
            ]
        )
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    96, scale=local_crops_scale, interpolation=Image.BICUBIC
                ),
                flip_and_color_jitter,
                PILRandomGaussianBlur(p=0.5),
                normalize,
            ]
        )

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


class ParametrizedDataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = ptx.Compose(
            [
                ptx.RandomHorizontalFlip(p=0.5),
                ptx.RandomApply(
                    [
                        ptx.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                ptx.RandomGrayscale(p=0.2),
            ]
        )
        normalize = ptx.Compose(
            [
                ptx.ToTensor(),
                ptx.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        self.global_transfo1 = ptw.CastParamsToTensor(
            transform=ptx.Compose(
                [
                    ptx.RandomResizedCrop(
                        224, scale=global_crops_scale, interpolation=Image.BICUBIC
                    ),
                    flip_and_color_jitter,
                    ptx.RandomApply([ParametrizedPILGaussianBlur()], p=1.0),
                    normalize,
                ]
            )
        )
        # second global crop
        self.global_transfo2 = ptw.CastParamsToTensor(
            transform=ptx.Compose(
                [
                    ptx.RandomResizedCrop(
                        224, scale=global_crops_scale, interpolation=Image.BICUBIC
                    ),
                    flip_and_color_jitter,
                    ptx.RandomApply([ParametrizedPILGaussianBlur()], p=0.1),
                    ptx.RandomApply([ParametrizedSolarization()], p=0.2),
                    normalize,
                ]
            )
        )
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = ptw.CastParamsToTensor(
            transform=ptx.Compose(
                [
                    ptx.RandomResizedCrop(
                        96, scale=local_crops_scale, interpolation=Image.BICUBIC
                    ),
                    flip_and_color_jitter,
                    ptx.RandomApply([ParametrizedPILGaussianBlur()], p=0.5),
                    normalize,
                ]
            )
        )

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


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
