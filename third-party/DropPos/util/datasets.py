# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import os
import PIL
import torchvision.transforms
from PIL import Image

from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class ImageListFolder(datasets.ImageFolder):
    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        ann_file=None,
        loader=default_loader,
    ):
        self.root = root
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.nb_classes = 1000

        assert ann_file is not None
        print("load info from", ann_file)

        self.samples = []
        ann = open(ann_file)
        for elem in ann.readlines():
            cut = elem.split(" ")
            path_current = os.path.join(root, cut[0])
            target_current = int(cut[1])
            self.samples.append((path_current, target_current))
        ann.close()

        print("load finish")


class HFWrapper(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        elem = self.dataset[index]
        img, target = elem["image"], elem["label"]
        if self.transform is not None:
            img = self.transform(img.convert("RGB"))
        return img, target

    def __len__(self):
        return len(self.dataset)


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_path.startswith("hf+"):
        items = args.data_path.split("+")
        path = items[1]
        name = items[2] or None if len(items) >= 3 else None
        cache_dir = items[3] or None if len(items) >= 4 else None

        from datasets import load_dataset

        dataset = load_dataset(
            path, name=name, split="train" if is_train else "validation", cache_dir=cache_dir
        )
        dataset = HFWrapper(dataset, transform)
    if args.data_path.startswith("snellius+"):
        # remove snellius+ prefix
        path = args.data_path[len("snellius+") :]
        root = os.path.join(path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
    else:
        # TODO modify your own dataset here
        folder = os.path.join(args.data_path, "train" if is_train else "val")
        ann_file = os.path.join(args.data_path, "train.txt" if is_train else "val.txt")
        dataset = ImageListFolder(folder, transform=transform, ann_file=ann_file)

        print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation="bicubic",
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(
            size, interpolation=torchvision.transforms.InterpolationMode.BICUBIC
        ),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
