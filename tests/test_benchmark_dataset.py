import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler, SequentialSampler
from data.components.sampling_utils import sample_and_stitch
from torchvision.transforms.functional import to_tensor
from PIL import Image
from typing import Any, Literal
import pytest
from utils import InfiniteDataset, ensure_sync


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
            img, _ = self.dataset[idx]
            img = img.unsqueeze(0)
            return sample_and_stitch(img, self.patch_size)
        else:
            if self.slice:
                raise NotImplementedError("Slicing not implemented")
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
                    img, _ = self.dataset[i]
                    imgs.append(img.unsqueeze(0))

                imgs = torch.cat(imgs, dim=0).to(self.device)
                return sample_and_stitch(imgs, self.patch_size)


class PARTCollator(object):
    def __init__(
        self,
        patch_size: int,
        mode: Literal["offgrid", "ongrid", "canonical"] = "offgrid",
        device: torch.device = "cpu",
    ):
        self.patch_size = patch_size
        self.mode = mode
        self.device = device

    def __call__(self, batch: list[tuple[Image.Image, Any]]):
        X = torch.cat([to_tensor(x).unsqueeze(0) for x, _ in batch], dim=0)
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
            sample_and_stitch(X.to(self.device), self.patch_size, self.mode)
            for X, _ in iter(self.loader)
        )

def nop(*args):
    return args[0][0]

@pytest.mark.benchmark(group="Implementation", min_rounds=100)
@pytest.mark.parametrize(
    "batch_size, num_workers, img_size, patch_size, in_memory",
    # [(32, 4, 32, 4, True), (32, 4, 128, 16, True), (32, 4, 128, 16, False), (32, 4, 512, 16, False)],
    [(32, 4, 32, 4, False),  (32, 4, 256, 16, False)],
    # [(2, 8, 256, 16, True)],
)
def test_benchmark_dataset(
    batch_size: int, num_workers: int, img_size: int, patch_size: int, in_memory: bool, benchmark
):
    torch.multiprocessing.set_start_method("spawn", force=True)
    benchmark.group += f"[B: {batch_size}, I: {img_size}, M: {in_memory}]"
    data = InfiniteDataset(img_size, to_tensor, in_memory, 20000)
    dataset = PARTDataset(data, patch_size, slice=False, device="cuda")
    sampler = BatchSampler(SequentialSampler(dataset), batch_size, False)
    
    loader = DataLoader(
        dataset, sampler=sampler, num_workers=num_workers, collate_fn=nop, pin_memory=False
    )

    benchmark(ensure_sync(next), iter(loader))


@pytest.mark.benchmark(group="Implementation", min_rounds=100)
@pytest.mark.parametrize(
    "batch_size, num_workers, img_size, patch_size, in_memory",
    # [(32, 4, 32, 4, True), (32, 4, 128, 16, True), (32, 4, 128, 16, False), (32, 4, 512, 16, False)],
    [(32, 4, 32, 4, False),  (32, 4, 256, 16, False)],
    # [(32, 8, 256, 16, False)],
)
def test_benchmark_collator(
    batch_size: int, num_workers: int, img_size: int, patch_size: int, in_memory: bool, benchmark
):
    torch.multiprocessing.set_start_method("fork", force=True)
    benchmark.group += f"[B: {batch_size}, I: {img_size}, M: {in_memory}]"
    data = InfiniteDataset(img_size, to_tensor, in_memory, 20000)
    loader = DataLoader(
        data, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )
    part_loader = PARTBatchedDataLoader(loader, patch_size, device="cuda")
    benchmark(ensure_sync(next), iter(part_loader))


if __name__ == "__main__":
    batch_size, img_size, patch_size, in_memory = 1, 32, 4, True
    num_workers = 0
    data = InfiniteDataset(img_size, to_tensor, in_memory) 
    dataset = PARTDataset(data, patch_size, slice=False, device="cpu")
    sampler = BatchSampler(RandomSampler(dataset), batch_size, False)

    def nop(*args):
        return args[0][0]



    loader = DataLoader(
        dataset, sampler=sampler, collate_fn=nop
    )
    X, Y = next(iter(loader))
    print(X.shape, Y.shape)
    exit(0)

    collate_fn = PARTCollator(4)
    loader = DataLoader(data, batch_size=2, collate_fn=collate_fn)
    X, Y = next(iter(loader))
    print(X.shape, Y.shape)
