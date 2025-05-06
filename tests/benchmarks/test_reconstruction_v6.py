import torch
import pytest
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import default_collate
from src.data.components.transforms.multi_crop_v3 import ParametrizedMultiCropV3
from src.models.components.partmae_v5_2 import PARTMaskedAutoEncoderViT
import torch.utils._pytree as pytree
from torch import Tensor
from src.callbacks.reconstruction_logger import (
    plot_reconstructions,
    compute_reconstructions,
    preprocess_reconstructions,
)


class MockedDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.img = Image.open("artifacts/labrador.jpg")
        self.transform = transform

    def __getitem__(self, idx):
        return self.transform(self.img)

    def __len__(self):
        return 1000


@pytest.mark.parametrize("target_device", ["cpu", "cuda"])
@pytest.mark.parametrize("compile", [False, True])
def test_reconstruction_performance(target_device, compile, benchmark_v2):
    """
    Benchmark the reconstruction performance on different devices.
    The model runs on CUDA, but reconstruction can be performed on either CPU or CUDA.
    """
    # Settings for the model and data
    gV, lV = 2, 4  # Number of global and local views
    B = 32
    S = 4
    # Set up transform and model
    transform = ParametrizedMultiCropV3(
        n_global_crops=gV, n_local_crops=lV, canonical_crop_scale=(1, 1)
    )

    # Create a mock dataset with multiple images
    ds = MockedDataset(transform)

    batch_samples = [next(iter(ds)) for _ in range(B)]
    batch = default_collate(batch_samples)
    batch = pytree.tree_map_only(Tensor, lambda x: x.cuda(), batch)

    # Create and configure the model
    model = PARTMaskedAutoEncoderViT(
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        mask_ratio=0.7,
        pos_mask_ratio=0.7,
        sampler="ongrid_canonical",
    ).cuda()

    out = model(*batch)
    _compute_reconstructions = (
        torch.compile(compute_reconstructions) if compile else compute_reconstructions
    )

    # Define the function to benchmark - includes device transfer time
    def reconstruction_fn(device):
        # Process model output and generate plots
        io = preprocess_reconstructions(batch, out, num_samples=S, device=device)
        fig = plot_reconstructions(
            _compute_reconstructions(
                patch_size=model.patch_size,
                canonical_img_size=model.canonical_img_size,
                max_scale_ratio=model.max_scale_ratio,
                io=io,
            )
        )
        # Close the figure to free memory
        plt.close(fig)
        return fig

    # Run the benchmark
    benchmark_v2.benchmark(
        reconstruction_fn, args=(target_device,), n_warmup=5, n_runs=20
    )

    # Group results by device for comparison
    benchmark_v2.group_by(f"device: {target_device}")
    benchmark_v2.drop_columns(
        exclude=["time/min (ms)", "time/max (ms)", "time/mean (ms)"]
    )
