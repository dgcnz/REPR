from torch.utils.data import Dataset
from datasets import load_dataset
import os
import warnings

class HFDataset(Dataset):
    def __init__(
        self,
        dataset: str,
        name: str = None,
        split="train",
        img_key: str = "image",
        transform=None,
    ):
        warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

        if "HF_HOME" in os.environ:
            cache_dir = os.environ["HF_HOME"]
            print(f"Using cache dir: {cache_dir}")
        else:
            cache_dir = None
        self.ds = load_dataset(dataset, name, split=split, cache_dir=cache_dir)
        self.transform = transform
        self.img_key = img_key

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.ds[idx][self.img_key])
        return self.ds[idx]
