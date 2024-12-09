from torchvision.datasets import CIFAR10
import torch
from typing import Callable, Optional, Tuple, Any
from PIL import Image
import math
import numpy as np
from torch import Tensor
import skimage
from jaxtyping import Float, Int
from torchvision import transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset

def patchify_and_resize_np(start_x, start_y, end_x, end_y, image, resize):
    # image.shape: [32, 32, 3]
    image_patched = image[start_y:end_y, start_x:end_x, :]
    if resize is not None:
        image_patched = skimage.transform.resize(image_patched, resize)

    return image_patched


def patchify_and_resize(start_x, start_y, end_x, end_y, image, resize):
    # print(image[:, start_y:end_y, start_x:end_x].shape)
    image_patched = image[:, start_y:end_y, start_x:end_x]
    if resize is not None:
        image_patched = skimage.transform.resize(image_patched, resize)

    return image_patched


def patchify_and_resize_batch_np(start_x, start_y, end_x, end_y, image, resize):
    # image.shape: [batch, 32, 32, 3]
    image_patched = image[:, start_y:end_y, start_x:end_x, :]
    if resize is not None:
        image_patched = skimage.transform.resize(image_patched, resize)

    return image_patched


class VectorizedCIFAR(CIFAR10):
    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }

    def __init__(
            self,
            sampling: str,
            shuffle_patches: bool,
            img_size: int,
            patch_size: int,
            min_wh: int,
            max_wh: int,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            visualize: bool = False,
    ) -> None:

        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        # # load targets
        self.img_size = img_size
        self.patch_size = patch_size
        self.sampling = sampling
        self.shuffle_patches = shuffle_patches
        self.visualize = visualize

        if sampling == 'v1':
            # v1 unshuffled
            self.targets = torch.zeros(2, self.num_patches, dtype=int)  # [2, 64]
            x = torch.arange(0, img_size, patch_size, dtype=int)
            y = torch.arange(0, img_size, patch_size, dtype=int)
            target_x, target_y = torch.meshgrid(x, y, indexing='xy')
            self.targets[0], self.targets[1] = target_x.flatten(), target_y.flatten()

        elif sampling == 'v2':
            x_start = torch.arange(img_size - patch_size + 1)
            y_start = torch.arange(img_size - patch_size + 1)
            # save x_start and y_start in a file for future loading
            t = torch.cartesian_prod(y_start, x_start).T  # [y_start, x_start] torch.Size([2, 841])
            self.targets = torch.cat((t[None, 1, :], t[None, 0, :]), dim=0)  # [2, 841] switch y and x to x and y

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # self.targets: list 50000, self.data (50000, 32, 32, 3)
        # randomly choose the number of patches and then reshape
        if self.sampling == 'v1':
            # shuffled v1
            if self.shuffle_patches:
                r = torch.randperm(self.num_patches)
                self.targets = self.targets[:, r]
            target = self.targets

        elif self.sampling == 'v2':
            rand_ind = torch.randint(0, self.targets.shape[1], (self.num_patches,))  # torch.Size([64])
            # rand_ind = torch.arange(128*5, 128*5+64)
            target = self.targets[:, rand_ind]  # [num_channels, 64]: chosen x1, y1, x2, y2

        img = self.data[index]

        visualize = False
        if visualize:
            # img[32,32,3]
            patch_func = np.vectorize(patchify_and_resize_np,
                                      excluded=['image', 'resize'],
                                      # otypes=['uint8'],
                                      signature='(),(),(),()->(l,m,k)'
                                      )
            if self.sampling == 'v1' or self.sampling == 'v2':
                # patches: [64, 4, 4, 3]
                patches = patch_func(target[0], target[1], target[0] + self.patch_size, target[1] + self.patch_size,
                                     image=img,
                                     resize=None
                                     # resize=(3, self.patch_size, self.patch_size)
                                     )
                bbs = torch.cat((target, torch.cat(
                    (target[0].unsqueeze(dim=0) + self.patch_size, target[1].unsqueeze(dim=0) + self.patch_size),
                    dim=0)), dim=0)

            for i in [8, 16, 32, 64]:  # i is the number of patches visualized
                # img_patches: [64, 4, 4, 3]: ndarray
                # viz_img_bbs(img, bbs[:, :i])
                # reconstruct_image_unfold_v4(patches, target)
                # reconstruct_image_unfold_v4(patches[:i, :, :, :], bbs[:, :i])  # img_patches: [64, 4, 4, 3], [4, 64]
                # reconstruct_image_unfold(self.data[index], self.targets)
                pass

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)  # [3, 32, 32]

        patchify = np.vectorize(patchify_and_resize,
                                excluded=['image', 'resize'],
                                # otypes=['uint8'],
                                signature='(),(),(),()->(l,m,k)'
                                )

        if self.sampling == 'v1' or self.sampling == 'v2':
            # img_patches: [64, 3, 4, 4]
            img_patches = patchify(target[0], target[1], target[0]+self.patch_size, target[1]+self.patch_size,
                                   image=np.asarray(img),
                                   resize=None
                                   # resize=(3, self.patch_size, self.patch_size)
                                   )
        patches_per_width = int(math.sqrt(self.num_patches))
        img_patches = torch.from_numpy(img_patches)
        # [3, 64, 4, 4] -> [3, 8, 8, 4, 4] -> [3, 8, 4, 8, 4]
        img_patches = img_patches.permute(1, 0, 2, 3)
        img_patches = img_patches.reshape(3, patches_per_width, patches_per_width, self.patch_size, self.patch_size)
        img_patches = img_patches.permute(0, 1, 3, 2, 4).reshape(3, self.img_size, self.img_size)

        if self.target_transform is not None:
            target = self.target_transform(target)
        # print('reshaping and targets time {}'.format(str(datetime.timedelta(seconds=int(time.time() - start_time)))))
        if self.sampling == 'v1' or self.sampling == 'v2':
            # todo: I should do the normalization somewhere else
            target = (target / self.patch_size)

        return img_patches, target  # [4, 64]

def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            # color_jitter=args.color_jitter,
            # auto_augment=args.aa,
            # interpolation=args.train_interpolation,
            # re_prob=args.reprob,
            # re_mode=args.remode,
            # re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    # t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


class PatchedDataset(Dataset):
    def __init__(self, dataset: Dataset, patch_size: int):
        self.dataset = dataset
        self.patch_size = patch_size

    def _get_random_coordinates(self, image_size: tuple[int, int]):
        x = torch.randint(0, image_size[0] - self.patch_size, (1,))
        y = torch.randint(0, image_size[1] - self.patch_size, (1,))
        return x, y

    def __getitem__(self, index: int):
        img, target = self.dataset[index]
        img_patches = patchify_and_resize(0, 0, self.patch_size, self.patch_size, img, None)
        return img_patches, target

# testing
from collections import namedtuple
ARGS = namedtuple("args", ["input_size"])
args = ARGS(32)
transform = build_transform(False, args)


dataset = VectorizedCIFAR(
    sampling="v2",
    min_wh=4,
    max_wh=4,
    root="data",
    shuffle_patches=False,
    img_size=32,
    patch_size=16,
    transform=transform,
    target_transform=None,
    download=False,
    train=True,
)

imgix = 1
img_patches, target = dataset[imgix]
print(img_patches.shape)
print(target.shape)
print(dataset.data[imgix].shape)

# plot image
import matplotlib.pyplot as plt
# plot original image and sampled image
fig, axes = plt.subplots(1, 2)
axes[0].imshow(dataset.data[imgix])
axes[1].imshow(img_patches.permute(1, 2, 0))
fig.show()
plt.show()


