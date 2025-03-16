from PIL import Image
from jaxtyping import Float
from torch import Tensor
import torch
import parameterized_transforms.transforms as ptx
import parameterized_transforms.wrappers as ptw
from torch.utils.data import default_collate


class ParametrizedMultiCropV2(object):
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
    ):
        # scale params from
        # https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/args.txt
        self.canonical_size = canonical_size
        self.canonical_crop_scale = canonical_crop_scale
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.n_global_crops = n_global_crops
        self.n_local_crops = n_local_crops

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
                    normalize,
                ]
            )
        )
        self.local_ttx = ptw.CastParamsToTensor(
            transform=ptx.Compose(
                [
                    ptx.RandomResizedCrop(
                        local_size, scale=local_crops_scale, interpolation=Image.BICUBIC
                    ),
                    normalize,
                ]
            )
        )
        self.max_scale_ratio = self.compute_max_scale_ratio_aug()

    def __call__(self, image: Image.Image):
        Nl = self.n_local_crops
        Ng = self.n_global_crops

        image, canon_params = self.canonicalize(image.convert("RGB"))

        global_crops, global_params = default_collate(
            [self.global_ttx(image) for _ in range(self.n_global_crops)]
        )
        local_crops, local_params = default_collate(
            [self.local_ttx(image) for _ in range(self.n_local_crops)]
        )

        canon_params: Tensor = canon_params.unsqueeze(0)
        local_params = torch.cat([canon_params.expand(Nl, -1), local_params], 1)
        global_params = torch.cat([canon_params.expand(Ng, -1), global_params], 1)
        return global_crops, global_params, local_crops, local_params

    def recreate_canonical(
        self, image: Float[Tensor, "C H W"], canonical_params: Float[Tensor, "4"]
    ) -> Float[Tensor, "C H W"]:
        """
        Recreate the canonical image from the image and canonical parameters.
        """
        return self.canonicalize.transform.consume_transform(image, canonical_params)[0]

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
    t = ParametrizedMultiCropV2()
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
    batch = next(iter(loader))
    print(batch[0].shape, batch[1].shape, batch[2].shape, batch[3].shape)
