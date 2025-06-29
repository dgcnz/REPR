_base_ = [
    "../_base_/datasets/pascal_voc12_aug.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_40k.py",
]

crop_size = (512, 512)

norm_cfg = dict(type="SyncBN", requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='TIMMViT',
        model_name='vit_small_patch16_224.dino',
        pretrained=True,
        in_channels=3,
        img_size=crop_size,
        out_indices=(-4, -3, -2, -1),  # last 4 layers
        freeze=True,
    ),
    decode_head=dict(
        type='LinearHead',
        in_channels=[384, 384, 384, 384],
        in_index=[0, 1, 2, 3],
        input_transform='resize_concat',  # upsample & concat
        channels=384 * 4,  # 1536
        num_classes=150,
        dropout_ratio=0.0,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    init_cfg=dict(type="Pretrained", checkpoint=""),
    test_cfg=dict(mode='whole'),
)

# Override optimizer and LR schedule following VicRegL conventions
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.05
)

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
)

# Data settings: 2 images per GPU
data = dict(samples_per_gpu=2)
