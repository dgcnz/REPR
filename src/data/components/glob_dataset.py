from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

class GlobImageDataset(Dataset):
    """
    A dataset of images defined by a glob pattern,
    optionally scoped to a root directory, without labels or class folders.
    Returns only image tensors.

    Args:
        root (str or Path):
            Base directory to search (used with glob or regex).
        pattern (str, Path):
            - Glob pattern relative to `root` (e.g. '**/*.png' or 'images/**/*.jpg'),
        transform (callable, optional): Transform to apply to each image.
    """
    def __init__(self, root, pattern: str, transform=None):
        self.root = Path(root)
        self.transform = transform

        # Treat as glob relative to root
        self.paths = sorted(self.root.glob(str(pattern)))
        if not self.paths:
            raise RuntimeError(f"No images found in {self.root!r} for pattern: {pattern}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

if __name__ == "__main__":
    from src.data.components.transforms.multi_crop_v3 import ParametrizedMultiCropV3
    from torch.utils._pytree import tree_map_only
    import torch

    transform = ParametrizedMultiCropV3(
        n_global_crops=2,
        n_local_crops=0,
        canonical_size=240,
        canonical_crop_scale=(1.0, 1.0),
        global_size=224,
        local_size=96,
        distort_color=False,
    )

    # Using glob under './data':
    # Matches files like './data/.../000123_0003.png'
    print("Init dataset")
    dataset = GlobImageDataset(
        root='/home/dgcnz/development/datasets/clevrtex/clevrtexv2_outd',
        pattern='**/*_[0-9][0-9][0-9][0-9][0-9][0-9]_0003.png',
        transform=transform
    )
    print(len(dataset))
    # import pytree
    print(tree_map_only(
        torch.Tensor,
        lambda x: x.shape,
        dataset[0]
    ))
    # get all unique shapes
    # clevrtex [240, 320] always


    batch = dataset[0]
    # print(batch)
