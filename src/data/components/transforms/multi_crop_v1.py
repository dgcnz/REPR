from PIL import Image
import parameterized_transforms.transforms as ptx
import parameterized_transforms.wrappers as ptw
from torch.utils.data import default_collate
import torch
from jaxtyping import Float
from torch import Tensor


class ParametrizedMultiCropV1(object):
    def __init__(
        self,
        canonical_size: int = 512,
        canonical_crop_scale=(0.9, 1.0),
        global_crops_scale: tuple[float, float] = (0.14, 1.0),
        n_views: int = 2,
    ):
        self.canonical_size = canonical_size
        self.canonical_crop_scale = canonical_crop_scale
        self.global_crops_scale = global_crops_scale
        self.n_views = n_views

        self.canonicalize = ptw.CastParamsToTensor(
            transform=ptx.RandomResizedCrop(
                canonical_size, scale=canonical_crop_scale, interpolation=Image.BICUBIC
            )
        )
        self.augmentations = ptw.CastParamsToTensor(
            transform=ptx.Compose(
                [
                    ptx.RandomResizedCrop(
                        224, scale=global_crops_scale, interpolation=Image.BICUBIC
                    ),
                    ptx.ToTensor(),
                    ptx.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
        )
        self.n_views = n_views
        self.max_scale_ratio = self.compute_max_scale_ratio_aug()

    def __call__(self, image):
        image, canonical_params = self.canonicalize(image)
        V = self.n_views
        # canonical_params = [yc, xc, hc, hw]
        # [V, 4]
        canonical_params = canonical_params.unsqueeze(0).expand(V, -1)
        # params[i] = [y, x, h, w]
        imgs, params = default_collate([self.augmentations(image) for _ in range(V)])
        # concatenate canonical params with params
        # params[i] = [yc, xc, hc, hw, y, x, h, w]
        params = torch.cat([canonical_params, params], dim=1)  # [nV, 8]
        return imgs, params

    def recreate_canonical(
        self, image: Float[Tensor, "C H W"], canonical_params: Float[Tensor, "4"]
    ) -> Float[Tensor, "C H W"]:
        """
        Recreate the canonical image from the image and canonical parameters.
        """
        return self.canonicalize.transform.consume_transform(image, canonical_params)[0]

    def compute_max_scale_ratio_aug(self):
        rrc = self.augmentations.transform.transforms[0]
        scale = rrc.scale
        ratio = rrc.ratio
        scale_min, scale_max = min(scale), max(scale)
        ratio_min, ratio_max = min(ratio), max(ratio)
        max_ratio = ((scale_max * ratio_max) / (scale_min * ratio_min)) ** 0.5
        return max_ratio


if __name__ == "__main__":
    t = ParametrizedMultiCropV1()
    print(t.compute_max_scale_ratio_aug())

    img = Image.open("artifacts/labrador.jpg")
    out = t(img)
    print(out[0].shape, out[1].shape)
