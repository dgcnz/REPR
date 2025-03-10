from PIL import Image
import parameterized_transforms.transforms as ptx
import parameterized_transforms.wrappers as ptw


class ParametrizedMultiCropBare(object):
    def __init__(self, global_crops_scale: tuple[float, float] = (0.14, 1.0), n_views: int = 2):
        self.transform = ptw.CastParamsToTensor(
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

    def __call__(self, image):
        return [self.transform(image) for _ in range(self.n_views)]
