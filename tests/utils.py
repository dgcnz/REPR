import torch
from PIL import Image
from torch.utils.data import Dataset
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def ensure_sync(f):
    def wrapper(*args, **kwargs):
        torch.cuda.synchronize()
        out = f(*args, **kwargs)
        torch.cuda.synchronize()
        return out

    return wrapper


class InfiniteDataset(Dataset):
    def __init__(self, img_size: int, transform=None, in_memory=False, max_size: int = 1000):
        assert img_size in [32, 128, 256, 512]
        self.img_path = "tests/resources/{}_{}.jpg".format(img_size, img_size)
        self.transform = transform
        self.img = Image.open(self.img_path) if in_memory else None
        self.in_memory = in_memory
        self.max_size = max_size

    def __len__(self):
        return self.max_size

    def __getitem__(self, _):
        if self.in_memory:
            im = self.img
        else:
            im = Image.open(self.img_path)
        if self.transform:
            im = self.transform(im)
        return im, 0
