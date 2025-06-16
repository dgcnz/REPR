import os
import random

import hydra
import torch
import torch.nn as nn
import torchvision.transforms as T
from hydra.utils import instantiate
from omegaconf import DictConfig
from torchvision.transforms.functional import InterpolationMode

from src.data.VOCdevkit.vocdata import VOCDataModule
from src.data.coco.coco_data_module import CocoDataModule
from src.data.cityscapes.cityscapes_data import CityscapesDataModule
from src.data.ade20k.ade20kdata import Ade20kDataModule
from src.experiments.utils.neco_utils import PredsmIoU
from src.experiments.utils.linear_finetuning_transforms import SepTransforms
from src.experiments.utils.timm_seg_utils import extract_feature_map


@hydra.main(version_base="1.3", config_path="../../../fabric_configs/experiment/linear_segmentation", config_name="config")
def eval_bulk(cfg: DictConfig) -> None:
    ckpt_path_backbone = cfg.ckpt_path_backbone
    ckpt_path_head = cfg.ckpt_path_head
    batch_size = cfg.batch_size
    input_size = cfg.input_size
    mask_eval_size = cfg.mask_eval_size
    dataset_name = cfg.data.dataset_name
    data_dir = cfg.data.data_dir
    num_classes = cfg.num_classes
    device = "cuda" if torch.cuda.is_available() else "cpu"
    miou_metric = PredsmIoU(num_classes, num_classes)

    model: torch.nn.Module = instantiate(cfg.model.net)
    model.eval().to(device)

    finetune_head = nn.Conv2d(model.embed_dim, num_classes, 1)

    if ckpt_path_backbone:
        state = torch.load(ckpt_path_backbone, map_location="cpu")
        msg = model.load_state_dict(state, strict=False)
        print(msg)

    # load linear head
    state_dict = torch.load(ckpt_path_head)
    weights = {k.replace("finetune_head.", ""): v for k, v in state_dict.items()}
    msg = finetune_head.load_state_dict(weights, strict=False)
    print(msg)
    assert len(msg[0]) == 0
    finetune_head.eval()
    finetune_head.to(device)

    # Init transforms and data
    val_image_transforms = T.Compose(
        [
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_target_transforms = T.Compose(
        [
            T.Resize((input_size, input_size), interpolation=InterpolationMode.NEAREST),
            T.ToTensor(),
        ]
    )

    if dataset_name == "voc":
        num_classes = 21
        ignore_index = 255
        data_module = VOCDataModule(
            batch_size=batch_size,
            num_workers=5,
            train_split="trainaug",
            val_split="val",
            data_dir=data_dir,
            train_image_transform=val_image_transforms,
            drop_last=False,
            val_image_transform=val_image_transforms,
            val_target_transform=val_target_transforms,
        )
    elif "coco" in dataset_name:
        assert len(dataset_name.split("-")) == 2
        mask_type = dataset_name.split("-")[-1]
        assert mask_type in ["thing", "stuff"]
        if mask_type == "thing":
            num_classes = 12
        else:
            num_classes = 15
        ignore_index = 255
        file_list = os.listdir(os.path.join(data_dir, "images", "train2017"))
        file_list_val = os.listdir(os.path.join(data_dir, "images", "val2017"))
        random.shuffle(file_list_val)
        # sample 10% of train images
        random.shuffle(file_list)
        file_list = file_list[: int(len(file_list) * 0.1)]
        print(f"sampled {len(file_list)} COCO images for training")

        data_module = CocoDataModule(
            batch_size=batch_size,
            num_workers=5,
            file_list=file_list,
            data_dir=data_dir,
            file_list_val=file_list_val,
            mask_type=mask_type,
            train_transforms=val_image_transforms,
            val_transforms=val_image_transforms,
            val_target_transforms=val_target_transforms,
        )

    elif dataset_name == "ade20k":
        num_classes = 151
        ignore_index = 0
        val_transforms = SepTransforms(val_image_transforms, val_target_transforms)
        data_module = Ade20kDataModule(
            data_dir,
            train_transforms=val_transforms,
            val_transforms=val_transforms,
            shuffle=False,
            num_workers=5,
            batch_size=batch_size,
        )
    elif dataset_name == "cityscapes":
        num_classes = 19
        ignore_index = 255
        val_transforms = SepTransforms(val_image_transforms, val_target_transforms)
        data_module = CityscapesDataModule(
            root=data_dir,
            train_transforms=val_transforms,
            val_transforms=val_transforms,
            shuffle=True,
            num_workers=5,
            batch_size=batch_size,
        )
    else:
        raise ValueError(f"{dataset_name} not supported as dataset")

    data_module.setup()

    # Get head predictions
    with torch.no_grad():
        for i, (imgs, masks) in enumerate(data_module.val_dataloader()):
            bs = imgs.size(0)
            tokens = extract_feature_map(model, imgs.to(device))
            tokens = nn.functional.interpolate(
                tokens, size=(mask_eval_size, mask_eval_size), mode="bilinear"
            )
            mask_preds = finetune_head(tokens)
            mask_preds = torch.argmax(mask_preds, dim=1).unsqueeze(1)

            # downsample masks and preds
            gt = masks * 255
            gt = nn.functional.interpolate(
                gt, size=(mask_eval_size, mask_eval_size), mode="nearest"
            )
            valid = gt != ignore_index  # mask to remove object boundary class

            # update metric
            miou_metric.update(gt[valid].cpu(), mask_preds[valid].cpu())

            if (i + 1) % 50 == 0:
                print(f"{(i + 1) * bs} done")

    # Calculate mIoU
    miou = miou_metric.compute(True, linear_probe=True)[0]
    miou_metric.reset()
    print(miou)


if __name__ == "__main__":
    eval_bulk()
