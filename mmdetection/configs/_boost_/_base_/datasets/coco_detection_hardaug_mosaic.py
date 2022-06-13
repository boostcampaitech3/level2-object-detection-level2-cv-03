# dataset settings
dataset_type = 'CocoDataset'
data_root = '../../dataset/'
data_json = '../../dataset/MultilabelKFold/'

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

img_scale = (1024,1024)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

albu_train_transforms=[
    dict(
        type="OneOf",
        transforms=[
            dict(type="Blur"),
            dict(type="MedianBlur"),
            dict(type="GaussianBlur"),
        ],
        p=0.3,
    ),
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.1,
        rotate_limit=20,
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.3),
    dict(
        type='HueSaturationValue',
        p=0.3,
    ),
    dict(
        type='OneOf',
        transforms=[
            dict(type="IAAAdditiveGaussianNoise"),
            dict(type="GaussNoise"),
        ],
        p=0.3,
    ),
    dict(
        type='Sharpen',
        p=0.3,
    ),
    dict(
        type='RandomSizedBBoxSafeCrop',
        width=768,
        height=768,
        p=0.1
    )
]

train_pipeline = [
    dict(type="RandomFlip", flip_ratio=0.5, direction="vertical"),
    dict(type="RandomFlip", flip_ratio=0.5, direction="horizontal"),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format="pascal_voc",
            label_fields=['gt_labels'],
            filter_lost_elements=True),
        keymap={
            'img':'image',
            'gt_bboxes':'bboxes'
        }),
    dict(
        type='Mosaic', 
        img_scale=img_scale, 
        center_ratio_range=(0.8, 1.2),
        pad_val=20.0), ## 이전에는 114
    dict(
        #### 이 부분이 multiscale 부분입니다! ####
        type="Resize",
        img_scale=[(1024, 1024)],
        multiscale_mode="range",
        ratio_range=(1.0, 2.0),
        keep_ratio=True,
    ),
    #######################################
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        classes=classes, ## 해당 부분 추가!
        ann_file=data_json + 'cv_train_1.json',
        img_prefix=data_root,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False,
    ),
    pipeline=train_pipeline,)

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(1024,1024)],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(1536, 1536), (1536, 1024), (1024, 1536), (1024, 1024),
                   (1536, 768), (768, 1536), (768, 768),
                   (1024, 768), (768, 1024),
                   ],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    train=train_dataset,
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_json + 'cv_val_1.json',
        img_prefix=data_root,
        pipeline=test_val_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')

