import torch
import pytest
from PIL import Image
from src.data.components.transforms.multi_crop_v3 import ParametrizedMultiCropV3
from src.data.components.transforms.multi_crop_v2 import ParametrizedMultiCropV2
from src.models.components.partmae_v5 import PART_mae_vit_base_patch16 as PART_v5
from src.models.components.partmae_v5_2 import PART_mae_vit_base_patch16 as PART_v5_2


class MockedDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.img = Image.open("artifacts/labrador.jpg")
        self.transform = transform

    def __getitem__(self, idx):
        return self.transform(self.img)

    def __len__(self):
        return 100000


@pytest.mark.parametrize("version", ["v5", "v5_2"])
@pytest.mark.parametrize("compile", [True, False])
def test_v5_vs_v5_2(version, compile, benchmark_v2):
    # Settings
    B, gV, lV = 128, 2, 4
    gH = 224
    lH = 96
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_fn = PART_v5 if version == "v5" else PART_v5_2
    t_kwargs = {
        "global_size": gH,
        "local_size": lH,
        "n_global_crops": gV,
        "n_local_crops": lV,
        "distort_color": True,
    }
    if version == "v5":
        t = ParametrizedMultiCropV2(**t_kwargs)
    else:
        t = ParametrizedMultiCropV3(**t_kwargs)

    ds = MockedDataset(t)
    module = model_fn(pos_mask_ratio=0.75, mask_ratio=0.75).to(device).eval()
    if compile:
        module = torch.compile(module)
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=B)))
    batch = torch.utils._pytree.tree_map_only(
        torch.Tensor, lambda x: x.to(device), batch
    )

    def ffn(*batch):
        with torch.no_grad():
            return module(*batch)

    benchmark_v2.benchmark(ffn, args=batch, n_warmup=5, n_runs=10)
    benchmark_v2.group_by(f"compile: {compile}")
    benchmark_v2.drop_columns(exclude=["time/min (ms)", "time/max (ms)"])
