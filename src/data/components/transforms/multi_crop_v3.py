from PIL import Image
from jaxtyping import Float
from torch import Tensor
import torch
import parameterized_transforms.transforms as ptx
import parameterized_transforms.wrappers as ptw
from src.data.components.transforms.pil_gaussian_blur import (
    ParametrizedPILGaussianBlur,
)


class ParametrizedMultiCropV3(object):
    def __init__(
        self,
        canonical_size: int = 512,
        global_size: int = 224,
        local_size: int = 96,
        canonical_crop_scale=(0.9, 1.0),
        global_crops_scale: tuple[float, float] = (0.25, 1.0),
        local_crops_scale: tuple[float, float] = (0.05, 0.25),
        n_global_crops: int = 2,
        n_local_crops: int = 8,
        distort_color: bool = False,
    ):
        self.canonical_size = canonical_size
        self.canonical_crop_scale = canonical_crop_scale
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.n_global_crops = n_global_crops
        self.n_local_crops = n_local_crops

        color_transforms = [
            ptx.RandomApply(
                [
                    ptx.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                    )
                ],
                p=0.8,
            ),
            ptx.RandomGrayscale(p=0.2),
            ptx.RandomApply([ParametrizedPILGaussianBlur()], p=0.5),
        ]
        if not distort_color:
            color_transforms = []


        self.canonicalize = ptw.CastParamsToTensor(
            transform=ptx.RandomResizedCrop(
                canonical_size, scale=canonical_crop_scale, interpolation=Image.BICUBIC
            )
        )
        normalize = ptx.Compose(
            [
                ptx.ToTensor(),
                ptx.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        self.global_ttx = ptw.CastParamsToTensor(
            transform=ptx.Compose(
                [
                    ptx.RandomResizedCrop(
                        global_size,
                        scale=global_crops_scale,
                        interpolation=Image.BICUBIC,
                    ),
                    *color_transforms,
                    normalize,
                ]
            )
        )
        self.local_ttx = ptw.CastParamsToTensor(
            transform=ptx.Compose(
                [
                    ptx.RandomResizedCrop(
                        local_size,
                        scale=local_crops_scale,
                        interpolation=Image.BICUBIC,
                    ),
                    *color_transforms,
                    normalize,
                ]
            )
        )
        self.max_scale_ratio = self.compute_max_scale_ratio_aug()

    def __call__(self, image: Image.Image):
        # First, canonicalize the image.
        image, canon_params = self.canonicalize(image.convert("RGB"))
        # Make sure canon_params is a 2D tensor of shape [1, 4]
        canon_params = canon_params.unsqueeze(0)

        # Process global crops: each call returns a tuple (crop, crop_params)
        global_results = [self.global_ttx(image) for _ in range(self.n_global_crops)]
        global_crops = [result[0] for result in global_results]
        # For each crop, prepend the canonical parameters to yield an 8-dim vector.
        global_params = [
            torch.cat([canon_params.squeeze(0), result[1]], dim=0)
            for result in global_results
        ]

        # Process local crops similarly.
        local_results = [self.local_ttx(image) for _ in range(self.n_local_crops)]
        local_crops = [result[0] for result in local_results]
        local_params = [
            torch.cat([canon_params.squeeze(0), result[1]], dim=0)
            for result in local_results
        ]

        # Return lists concatenating global and local crops and parameters.
        crops_list = global_crops + local_crops
        params_list = global_params + local_params
        return crops_list, params_list

    def recreate_local(
        self, canonical_img: Float[Tensor, "C H W"], local_params: Float[Tensor, "N 4"]
    ) -> Float[Tensor, "C H W"]:
        return [
            self.local_ttx.transform.transforms[0].consume_transform(
                canonical_img, local_params[i].tolist()
            )[0]
            for i in range(local_params.shape[0])
        ]

    def recreate_global(
        self, canonical_img: Float[Tensor, "C H W"], global_params: Float[Tensor, "N 4"]
    ) -> Float[Tensor, "C H W"]:
        return [
            self.global_ttx.transform.transforms[0].consume_transform(
                canonical_img, global_params[i].tolist()
            )[0]
            for i in range(global_params.shape[0])
        ]

    def recreate_canonical(
        self, image: Float[Tensor, "C H W"], canonical_params: Float[Tensor, "4"]
    ) -> Float[Tensor, "C H W"]:
        return self.canonicalize.transform.consume_transform(
            image, canonical_params.tolist()
        )[0]

    def compute_max_scale_ratio_aug(self) -> float:
        grrc = self.global_ttx.transform.transforms[0]
        lrrc = self.local_ttx.transform.transforms[0]
        scale_min = min(*grrc.scale, *lrrc.scale)
        scale_max = max(*grrc.scale, *lrrc.scale)
        ratio_min = min(*grrc.ratio, *lrrc.ratio)
        ratio_max = max(*grrc.ratio, *lrrc.ratio)
        max_ratio = ((scale_max * ratio_max) / (scale_min * ratio_min)) ** 0.5
        return max_ratio


if __name__ == "__main__":
    gV, lV = 2, 0
    V = gV + lV
    t = ParametrizedMultiCropV3(n_global_crops=gV, n_local_crops=lV, distort_color=True)
    print(t.compute_max_scale_ratio_aug())  # <5.97

    class MockedDataset(torch.utils.data.Dataset):
        def __init__(self, transform=None):
            self.img = Image.open("artifacts/labrador.jpg")
            self.transform = transform

        def __getitem__(self, idx):
            return self.transform(self.img)

        def __len__(self):
            return 4

    dataset = MockedDataset(t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)
    x, params = next(iter(loader))

    for i in range(V):
        print(x[i].shape)

    for i in range(V):
        print(params[i].shape)

    # print without scientific notation
    torch.set_printoptions(sci_mode=False)

    print(params[0][0].shape)
    print(params[0][0])
