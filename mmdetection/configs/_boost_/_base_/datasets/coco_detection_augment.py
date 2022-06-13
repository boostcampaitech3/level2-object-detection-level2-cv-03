# dataset settings
dataset_type = 'CocoDataset'
data_root = '../../dataset/'
data_json = '../../dataset/MultilabelKFold/'

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

img_scale = (1024,1024)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)


albu_transforms=[
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=0,
        interpolation=1,
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],
        p=0.1),
    dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=0.5),
            dict(type='MedianBlur', blur_limit=3, p=0.5)
        ],
        p=0.1),
]


train_pipeline = [
    # dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    # dict(
    #     type='MixUp',
    #     img_scale=img_scale,
    #     ratio_range=(0.8, 1.6),
    #     pad_val=114.0),
    # dict(type='CutOut', 
    #     n_holes=3,
    #     cutout_shape=[(4, 4), (4, 8), (8, 4),]),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Albu',
        transforms=albu_transforms,
        bbox_params=dict(
            type='BboxParams',
            format="pascal_voc",
            label_fields=['gt_labels'],
            filter_lost_elements=True),
        keymap={
            'img':'image',
            'gt_bboxes':'bboxes'
        }
        ),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
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
'''
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(
    #     type='Albu',
    #     transforms=[
    #         dict(
    #             type='ShiftScaleRotate',
    #             shift_limit=0.0625,
    #             scale_limit=0.0,
    #             rotate_limit=0,
    #             interpolation=1,
    #             p=0.5),
    #         dict(
    #             type='RandomBrightnessContrast',
    #             brightness_limit=[0.1, 0.3],
    #             contrast_limit=[0.1, 0.3],
    #             p=0.2),
    #         dict(type='ChannelShuffle', p=0.1),
    #         dict(
    #             type='OneOf',
    #             transforms=[
    #                 dict(type='Blur', blur_limit=3, p=0.5),
    #                 dict(type='MedianBlur', blur_limit=3, p=0.5)
    #             ],
    #             p=0.1),]),
    dict(
        type="MultiScaleFlipAug",
        img_scale=[(1024, 1024), (1024,768), (768,1024), (768,768)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ]),
]
'''
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

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(
    #     type='Albu',
    #     transforms=[
    #         dict(
    #             type='ShiftScaleRotate',
    #             shift_limit=0.0625,
    #             scale_limit=0.0,
    #             rotate_limit=0,
    #             interpolation=1,
    #             p=0.5),
    #         dict(
    #             type='RandomBrightnessContrast',
    #             brightness_limit=[0.1, 0.3],
    #             contrast_limit=[0.1, 0.3],
    #             p=0.2),
    #         dict(type='ChannelShuffle', p=0.1),
    #         dict(
    #             type='OneOf',
    #             transforms=[
    #                 dict(type='Blur', blur_limit=3, p=0.5),
    #                 dict(type='MedianBlur', blur_limit=3, p=0.5)
    #             ],
    #             p=0.1),]),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(1024, 1024), (1024,768), (768,1024), (768,768)],
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
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_json + 'test.json',
        img_prefix=data_root,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')

