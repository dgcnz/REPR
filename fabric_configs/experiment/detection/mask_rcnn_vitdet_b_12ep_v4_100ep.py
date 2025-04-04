from detectron2.config import LazyConfig


cfg = LazyConfig.load(
    "third-party/detectron2/projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_100ep.py"
)
train = cfg.train
dataloader = cfg.dataloader
optimizer = cfg.optimizer
model = cfg.model
lr_multiplier = cfg.lr_multiplier

train.init_checkpoint = (
    "artifacts/model-th0jy6wx:v0/epoch_0099.ckpt"
    # "artifacts/model-th0jy6wx:v0/epoch_0124.ckpt"
)
## droppos : 
# dataloader.train.total_batch_size = 16
# train.max_iter = 88750  # 12ep
# lr_multiplier.scheduler.milestones = [78889, 85463]
# lr_multiplier.scheduler.num_updates = train.max_iter
# optimizer.lr = 3e-4

# part:

train.max_iter = train.max_iter * 12 // 100  # 100ep -> 12ep
lr_multiplier.scheduler.milestones = [
    milestone * 12 // 100 for milestone in lr_multiplier.scheduler.milestones
]
lr_multiplier.scheduler.num_updates = train.max_iter

## default:
# dataloader.train.total_batch_size = 64
# optimizer.lr = 2e-4