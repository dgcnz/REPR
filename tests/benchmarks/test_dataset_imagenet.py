import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
    BatchSampler,
    RandomSampler,
)
from data.components.sampling_utils import sample_and_stitch
import torchvision.transforms as TT
from PIL import Image
from typing import Any, Literal
import pytest
from datasets import load_dataset

class PARTDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        patch_size: int,
        slice: bool = True,
        device: torch.device = "cpu",
    ):
        self.dataset = dataset
        self.patch_size = patch_size
        self.slice = slice
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int | slice | list):
        if isinstance(idx, int):
            img = self.dataset[idx]["image"]
            img = img.unsqueeze(0)
            return sample_and_stitch(img, self.patch_size)
        else:
            if self.slice:
                imgs = self.dataset[idx]["image"]
                # imgs is a list of tensors of shape (C, H, W)]
                # need to convert it to a single tensor of shape (N, C, H, W)
                imgs = torch.stack(imgs, dim=0)
                return sample_and_stitch(imgs, self.patch_size)
            else:
                if isinstance(idx, list):
                    idx = idx
                else:
                    start = idx.start or 0
                    stop = idx.stop or len(self.dataset)
                    step = idx.step or 1
                    idx = range(start, stop, step)
                imgs = []
                for i in idx:
                    img = self.dataset[i]["image"]
                    imgs.append(img.unsqueeze(0))
                try:
                    imgs = torch.cat(imgs, dim=0).to(self.device)
                except Exception as e:
                    raise Exception(
                        f"Error with shapes {[x.shape for x in imgs]}"
                    ) from e
                return sample_and_stitch(imgs, self.patch_size)


class PARTCollator(object):
    def __init__(
        self,
        patch_size: int,
        mode: Literal["offgrid", "ongrid", "canonical"] = "offgrid",
        device: torch.device = "cpu",
        collated: bool = False,
    ):
        self.patch_size = patch_size
        self.mode = mode
        self.device = device
        self.collated = collated

    def __call__(self, batch: list[tuple[Image.Image, Any]]):
        if self.collated:
            raise NotImplementedError("Collated not implemented")
        else:
            X = torch.stack([x["image"] for x in batch], dim=0)
        X = X.to(self.device)
        return sample_and_stitch(X, self.patch_size, self.mode)


class PARTBatchedDataLoader(object):
    def __init__(
        self,
        loader: DataLoader,
        patch_size: int,
        mode: Literal["offgrid", "ongrid", "canonical"] = "offgrid",
        device: torch.device = "cpu",
    ):
        self.loader = loader
        self.patch_size = patch_size
        self.mode = mode
        self.device = device

    def __iter__(self):
        return (
            sample_and_stitch(data["image"].to(self.device), self.patch_size, self.mode)
            for data in iter(self.loader)
        )


def nop(*args):
    return args[0][0]


rgb_transform = TT.Lambda(lambda x: x.convert("RGB"))
IMAGENET_TRANSFORMS = TT.Compose([rgb_transform, TT.Resize((224, 224)), TT.ToTensor()])


def apply_transforms(data):
    return {
        "image": [IMAGENET_TRANSFORMS(x) for x in data["image"]],
        "label": data["label"],
    }




def load_k_batches(dataloader: DataLoader, k: int):
    # loader = iter(dataloader)
    loader = dataloader
    for _ in range(k):
        _ = next(loader)


@pytest.mark.parametrize(
    "batch_size, num_workers",
    [(32, 0), (32, 4), (32, 8)],
    # [(32, 0)],
    # [(32, 0), (64, 0)],
)
@pytest.mark.parametrize("slice", [False])
def test_benchmark_dataset(batch_size: int, num_workers: int, slice: bool, benchmark_v2):
    torch.multiprocessing.set_start_method("spawn", force=True)
    patch_size = 16
    k = 1
    ds = load_dataset("ILSVRC/imagenet-1k", split="validation").with_format("torch")
    ds.set_transform(apply_transforms)
    dataset = PARTDataset(ds, patch_size, slice=slice, device="cuda")
    sampler = BatchSampler(RandomSampler(dataset), batch_size, False)

    loader = DataLoader(
        dataset,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=nop,
        pin_memory=False,
    )
    loader = iter(loader)
    benchmark_v2.benchmark(load_k_batches, args=(loader, k), n_runs=10, n_warmup=2)


@pytest.mark.parametrize(
    "batch_size, num_workers",
    [(32, 0), (32, 4), (32, 8)],
)
def test_benchmark_collate(batch_size: int, num_workers: int, benchmark_v2):
    patch_size = 16
    k = 1
    torch.multiprocessing.set_start_method("spawn", force=True)
    ds = load_dataset("ILSVRC/imagenet-1k", split="validation").with_format("torch")
    ds.set_transform(apply_transforms)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=PARTCollator(patch_size, device="cuda"),
        shuffle=True,
    )
    loader = iter(loader)
    benchmark_v2.benchmark(load_k_batches, args=(loader, k), n_runs=10, n_warmup=2)


@pytest.mark.parametrize(
    "batch_size, num_workers",
    [(32, 0), (32, 4), (32, 8)],
)
def test_benchmark_dataloader(batch_size: int, num_workers: int, benchmark_v2):
    patch_size = 16
    k = 1
    torch.multiprocessing.set_start_method("fork", force=True)
    ds = load_dataset("ILSVRC/imagenet-1k", split="validation").with_format("torch")
    ds.set_transform(apply_transforms)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        shuffle=True,
    )
    part_loader = PARTBatchedDataLoader(loader, patch_size, device="cuda")
    part_loader = iter(part_loader)
    benchmark_v2.benchmark(load_k_batches, args=(part_loader, k), n_runs=10, n_warmup=2)


if __name__ == "__main__":

    patch_size = 16
    torch.multiprocessing.set_start_method("spawn", force=True)
    ds = load_dataset("ILSVRC/imagenet-1k", split="validation").with_format("torch")
    ds.set_transform(apply_transforms)
    loader = DataLoader(
        ds,
        batch_size=4,
        pin_memory=False,
        collate_fn=PARTCollator(patch_size, device="cuda"),
    )
    X = next(iter(loader))