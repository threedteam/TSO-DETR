IMAGE_SIZE=640

model = dict(
    type="DeformableDETR",
    backbone=dict(
        type='PyramidVisionTransformerV2',
        init_cfg=dict(checkpoint='https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b1.pth')
    ),
    neck=dict(
        type="ChannelMapper",
        in_channels=[64, 128, 320, 512],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type="GN", num_groups=32),
        num_outs=4,
    ),
    bbox_head=dict(
        type="DeformableDETRHead",
        num_query=300,
        num_classes=5,
        in_channels=2048,
        sync_cls_avg_factor=True,
        as_two_stage=False,
        transformer=dict(
            type="DeformableDetrTransformer",
            encoder=dict(
                type="DetrTransformerEncoder",
                num_layers=6,
                transformerlayers=dict(
                    type="BaseTransformerLayer",
                    attn_cfgs=dict(
                        type="MultiScaleDeformableAttention", embed_dims=256
                    ),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=("self_attn", "norm", "ffn", "norm"),
                ),
            ),
            decoder=dict(
                type="DeformableDetrTransformerDecoder",
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type="DetrTransformerDecoderLayer",
                    attn_cfgs=[
                        dict(
                            type="MultiheadAttention",
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1,
                        ),
                        dict(type="MultiScaleDeformableAttention", embed_dims=256),
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=(
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                ),
            ),
        ),
        positional_encoding=dict(
            type="SinePositionalEncoding", num_feats=128, normalize=True, offset=-0.5
        ),
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=5.0),
        loss_iou=dict(type="GIoULoss", loss_weight=2.0),
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type="HungarianAssigner",
            cls_cost=dict(type="FocalLossCost", weight=2.0),
            reg_cost=dict(type="BBoxL1Cost", weight=5.0, box_format="xywh"),
            iou_cost=dict(type="IoUCost", iou_mode="giou", weight=2.0),
        ),
    ),
    test_cfg=dict(max_per_img=100),
)

workers = 2
batch_size = 2
dataset_type = "CocoDataset"
data_root = '/data/mmdSTTL/projects/data/CQUCH_reDA' # mod this
anno_root = f'{data_root}/annotations'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(IMAGE_SIZE, IMAGE_SIZE), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(IMAGE_SIZE, IMAGE_SIZE),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ]) 
]

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=workers,
    train=dict(
        type="CocoDataset",
        ann_file=f"{anno_root}/instances_train2017.json",
        img_prefix=f"{data_root}/train2017",
        pipeline=train_pipeline,
    ),
    val=dict(
        type="CocoDataset",
        ann_file=f"{anno_root}/instances_test2017.json",
        img_prefix=f"{data_root}/test2017",
        pipeline=test_pipeline,
    ),
    test=dict(
        type="CocoDataset",
        ann_file=f"{anno_root}/instances_test2017.json",
        img_prefix=f"{data_root}/test2017",
        pipeline=test_pipeline,
    ),
)

lr = 1e-4
evaluation = dict(
    interval=1, metric="bbox", save_best="auto"
)  # bbox, save best bbox map.
optimizer = dict(
    type="AdamW",
    lr=lr,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(lr_mult=0.1),
            "sampling_offsets": dict(lr_mult=0.1),
            "reference_points": dict(lr_mult=0.1),
        }
    ),
)

# 梯度裁剪
optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy="step", warmup="linear", warmup_iters=500, warmup_ratio=0.001, step=[40,80]
)

total_epochs = 100
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)

log_config = dict(
    interval=50, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)
checkpoint_config = dict(interval=1, max_keep_ckpts=3)

dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]
gpu_ids = range(0, 1)
