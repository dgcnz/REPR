from detectron2.config import LazyConfig
from src.models.components.timm_backbone import TimmBackbone
from detectron2.config import LazyCall as L
from detectron2.modeling import SimpleFeaturePyramid
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from functools import partial
from torch import nn


cfg = LazyConfig.load(
    "third-party/detectron2/projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_100ep.py"
)
train = cfg.train
dataloader = cfg.dataloader
optimizer = cfg.optimizer
model = cfg.model
# Original
model.backbone = L(SimpleFeaturePyramid)(
    net=L(TimmBackbone)(
        model_name="vit_base_patch16_224",
        features_only=True,
        pretrained=False,
        in_channels=3,
        out_indices=(-1,),
        dynamic_img_size=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        pretrained_cfg_overlay=dict(
            file="artifacts/model-th0jy6wx:v0/backbone_0099.ckpt"
        ),
        pretrained_strict=False,
    ),
    in_feature="p-1",  # changed
    out_channels=256,
    scale_factors=(4.0, 2.0, 1.0, 0.5),
    top_block=L(LastLevelMaxPool)(),
    norm="LN",
    square_pad=1024,
)

#   class_token=False,
#   global_pool="avg",
#   pretrained_cfg_overlay=dict(file=PARTViTModule.save_backbone(module_ckpt_path)),
#   pretrained_strict=False,
lr_multiplier = cfg.lr_multiplier

train.init_checkpoint = None

## droppos :
# dataloader.train.total_batch_size = 16
# train.max_iter = 88750  # 12ep
# lr_multiplier.scheduler.milestones = [78889, 85463]
# lr_multiplier.scheduler.num_updates = train.max_iter
# optimizer.lr = 3e-4

# part:

train.float32_precision = "high"
train.amp.precision = "16-mixed"
train.accumulate_grad_batches = 2
train.max_iter = train.max_iter * 12 // 100  # 100ep -> 12ep
lr_multiplier.scheduler.milestones = [
    milestone * 12 // 100 for milestone in lr_multiplier.scheduler.milestones
]
lr_multiplier.scheduler.num_updates = train.max_iter
dataloader.train.total_batch_size = 32

## default:
# dataloader.train.total_batch_size = 64
# optimizer.lr = 2e-4
