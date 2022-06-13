_base_ = [
    '../_base_/datasets/coco_detection_augment_hyunzin.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    type='YOLOF',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron/resnet50_caffe')),
    neck=dict(
        type='DilatedEncoder',
        in_channels=2048,
        out_channels=512,
        block_mid_channels=128,
        num_residual_blocks=4),
    bbox_head=dict(
        type='YOLOFHead',
        num_classes=10,
        in_channels=512,
        reg_decoded_bbox=True,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[0.5, 1.0, 2.0],
            scales=[1, 2, 4, 8, 16],
            strides=[32]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1., 1., 1., 1.],
            add_ctr_clamp=True,
            ctr_clamp=32),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='UniformAssigner', pos_ignore_thr=0.15, neg_ignore_thr=0.7),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))
# optimizer
optimizer = dict(
    type='SGD',
    lr=0.0012,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        norm_decay_mult=0., custom_keys={'backbone': dict(lr_mult=1. / 3)}))
# optimizer = dict(
#     _delete_=True,
#     type='AdamW',
#     lr=2e-4,
#     weight_decay=0.0001,
#     paramwise_cfg=dict(
#         custom_keys={
#             'backbone': dict(lr_mult=0.1),
#             'sampling_offsets': dict(lr_mult=0.1),
#             'reference_points': dict(lr_mult=0.1)
#         }))
# lr_config = dict(warmup_iters=1500, warmup_ratio=0.00066667)


## scheduler
lr_config = dict(
    _delete_=True,
    policy="cyclic",
    by_epoch=False,
    target_ratio=(0.5, 1e-4),  # 0.1 0.00001
    cyclic_times=5,
    step_ratio_up=0.5,
    gamma=0.7
)