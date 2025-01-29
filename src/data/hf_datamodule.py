from typing import Any, Dict, Optional, Callable

import functools
from lightning import LightningDataModule
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as TTv2
from datasets import load_dataset
import torch
import logging
import timm.data
from datasets import Dataset
from timm.data import Mixup, FastCollateMixup

logging.basicConfig(level=logging.INFO)

IMAGENET_TRANSFORMS = TTv2.Compose(
    [
        TTv2.ToImage(),
        TTv2.ToDtype(torch.float32, scale=True),
        TTv2.RGB(),
        TTv2.Resize(size=(224, 224), interpolation=TTv2.InterpolationMode.BICUBIC),
        TTv2.CenterCrop(size=(224, 224)),
        TTv2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

DEFAULT_TEST_TRANSFORM = TTv2.Compose(
    [
        TTv2.ToImage(),
        TTv2.RGB(),
        TTv2.ToDtype(torch.float32, scale=True),
        TTv2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

CIFAR10_TRAIN_TRANSFORM = timm.data.create_transform(
    input_size=32,
    is_training=True,
    color_jitter=0.4,
    auto_augment="rand-m9-mstd0.5-inc1",
    interpolation="bicubic",
    re_prob=0,  # 0.25 when finetuning for classification
    re_mode="pixel",
    re_count=1,
)

IMG_KEY = "image"
LABEL_KEY = "label"


def to_hf_transform(transform: Callable) -> Callable:
    @functools.wraps(transform)
    def _transform(batch):
        batch[IMG_KEY] = [transform(x.convert("RGB")) for x in batch[IMG_KEY]]
        return batch

    return _transform

def hf_mixup_fn(mixup_fn: Callable) -> Callable:
    @functools.wraps(mixup_fn)
    def _mixup_fn(batch: list[dict]):
        batch = torch.utils.data._utils.collate.default_collate(batch)
        # default collate
        batch = mixup_fn(batch[IMG_KEY], batch[LABEL_KEY])
        # reverts to dict
        return {IMG_KEY: batch[0], LABEL_KEY: batch[1]}
    return _mixup_fn


class HFDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_name: str = "uoft-cs/cifar10",
        train_transform: Callable = DEFAULT_TEST_TRANSFORM,
        test_transform: Callable = DEFAULT_TEST_TRANSFORM,
        img_key: str = "image",
        label_key: str | None = None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        val_fraction: float = None,  # split train into train/val
        test_fraction: float = None,  # split val into val/test
        mixup: FastCollateMixup | None = None,
        cache_dir: str | None = None,
    ) -> None:
        """Initialize a `HFDataModule`.

        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.dataset_name = dataset_name
        self.img_key = img_key
        self.label_key = label_key
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.prev_img_key = img_key
        self.prev_label_key = label_key

        self.train_transform = train_transform
        self.test_transform = test_transform

        self.batch_size_per_device = batch_size
        self.cli_logger = logging.getLogger(self.__class__.__name__)

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        logging.info(f"Preparing {self.hparams.dataset_name} dataset.")
        if self.hparams.dataset_name == self.dataset_name:
            logging.warning(
                "Before running this, make sure you use the HF CLI to download the data:\n"
                "huggingface-cli download ILSVRC/imagenet-1k --repo-type dataset"
            )

        _ = load_dataset(self.hparams.dataset_name, cache_dir=self.hparams.cache_dir)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            ds = load_dataset(
                self.dataset_name, cache_dir=self.hparams.cache_dir
            ).with_format("torch")

            # rename dataset keys to the standard "image" and "label"
            column_mapping = dict()
            if self.prev_img_key != IMG_KEY:
                column_mapping[self.prev_img_key] = IMG_KEY
            if self.prev_label_key is not None and self.prev_label_key != LABEL_KEY:
                column_mapping[self.prev_label_key] = LABEL_KEY
            if column_mapping:
                ds = ds.rename_columns(column_mapping)

            # split datasets
            if "validation" in ds:
                self.data_train = ds["train"]
                self.data_val = ds["validation"]
            elif "val" in ds:
                self.data_train = ds["train"]
                self.data_val = ds["val"]
            else:
                # split train into train/val
                if self.hparams.val_fraction is None:
                    raise ValueError(
                        "Validation fraction must be provided if no validation set is found."
                    )
                splits = ds["train"].train_test_split(
                    test_size=self.hparams.val_fraction
                )
                self.data_train = splits["train"]
                self.data_val = splits["test"]

            # split validation set into validation and test sets
            if self.hparams.test_fraction is None:
                self.data_test = ds["test"]
            else:
                splits = self.data_val.train_test_split(
                    test_size=self.hparams.test_fraction
                )
                self.data_val = splits["train"]
                self.data_test = splits["test"]

            self.data_train.set_transform(to_hf_transform(self.train_transform))
            self.data_val.set_transform(to_hf_transform(self.train_transform))
            self.data_test.set_transform(to_hf_transform(self.test_transform))

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        collate_fn = None
        if self.hparams.mixup is not None:
            collate_fn = hf_mixup_fn(self.hparams.mixup)
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


if __name__ == "__main__":

    _ = HFDataModule()
