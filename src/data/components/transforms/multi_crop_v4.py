from PIL import Image
import torch
from src.data.components.transforms.multi_crop_v2 import ParametrizedMultiCropV2


class ParametrizedMultiCropV4(ParametrizedMultiCropV2):
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
        
        crops = [torch.stack(crop) for crop in [global_crops, local_crops] if crop]
        params = [torch.stack(param) for param in [global_params, local_params] if param]
        return crops, params



if __name__ == "__main__":
    gV, lV = 2, 0
    V = gV + lV
    t = ParametrizedMultiCropV4(n_global_crops=gV, n_local_crops=lV, distort_color=True)
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

    for i in range(gV):
        print(x[0].shape)

    for i in range(gV):
        print(params[0].shape)

    # print without scientific notation
    torch.set_printoptions(sci_mode=False)

    print(params[0].shape)
    print(params[0])
