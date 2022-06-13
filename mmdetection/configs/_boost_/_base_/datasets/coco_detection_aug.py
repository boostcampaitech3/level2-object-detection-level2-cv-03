# dataset settings
dataset_type = "CocoDataset"
data_root = "../../dataset/"

# classes = (
#     "General trash",
#     "Paper",
#     "Paper pack",
#     "Metal",
#     "Glass",
#     "Plastic",
#     "Styrofoam",
#     "Plastic bag",
#     "Battery",
#     "Clothing",
# )
classes = (
    "Paper",
    "Paper pack",
    "Metal",
    "Glass",
    "Styrofoam",
    "Plastic bag",
    "Battery",
    "Clothing",
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
albu_train_transforms = [
    dict(type="Blur"),
    # dict(
    #     type="ShiftScaleRotate",
    #     shift_limit=0.0625,
    #     scale_limit=0.1,
    #     rotate_limit=20,
    #     p=0.5,
    # ),
    dict(
        type="RandomBrightnessContrast",
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.4,
    ),
    # dict(
    #     type="HueSaturationValue",
    #     p=0.3,
    # ),
    # dict(
    #     type="OneOf",
    #     transforms=[
    #         dict(type="IAAAdditiveGaussianNoise"),
    #         dict(type="GaussNoise"),
    #     ],
    #     p=0.3,
    # ),
    # dict(
    #     type="Sharpen",
    #     p=0.3,
    # ),
    # dict(type="RandomSizedBBoxSafeCrop", width=768, height=768, p=0.1),
]

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="Albu",
        transforms=albu_train_transforms,
        bbox_params=dict(
            type="BboxParams",
            format="pascal_voc",
            label_fields=["gt_labels"],
            min_visibility=0.0,
            filter_lost_elements=True,
        ),
        keymap={"img": "image", "gt_bboxes": "bboxes"},
        update_pad_shape=False,
    ),
    dict(
        #### 이 부분이 multiscale 부분입니다! ####
        type="Resize",
        img_scale=[(1024, 1024)],
        multiscale_mode="range",
        ratio_range=(0.5, 1.5),
        keep_ratio=True,
    ),
    #######################################
    dict(type="RandomFlip", flip_ratio=0.5, direction="vertical"),
    dict(type="RandomFlip", flip_ratio=0.5, direction="horizontal"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]


# albu_train_transforms = [
#     dict(
#         type="OneOf",
#         transforms=[
#             dict(type="Blur"),
#             dict(type="MotionBlur"),
#             dict(type="GaussNoise"),
#             dict(type="ImageCompression", quality_lower=75),
#         ],
#         p=0.4,
#     ),
# ]
# train_pipeline = [
#     dict(type="LoadImageFromFile"),
#     dict(type="LoadAnnotations", with_bbox=True),
#     dict(
#         type="Resize",
#         img_scale=[(1024, 1024)],
#         multiscale_mode="range",
#         ratio_range=(0.5, 2.0),
#         keep_ratio=True,
#     ),
#     dict(type="RandomFlip", flip_ratio=0.5, direction="vertical"),
#     dict(type="RandomFlip", flip_ratio=0.5, direction="horizontal"),
#     # dict(type="GaussNoise", var_limit=(10.0, 50.0), p=0.3),
#     dict(
#         type="Albu",
#         transforms=albu_train_transforms,
#         bbox_params=dict(
#             type="BboxParams",
#             format="pascal_voc",
#             label_fields=["gt_labels"],
#             min_visibility=0.0,
#             filter_lost_elements=True,
#         ),
#         keymap={"img": "image", "gt_bboxes": "bboxes"},
#         update_pad_shape=False,
#     ),
#     dict(type="Normalize", **img_norm_cfg),
#     dict(type="Pad", size_divisor=32),
#     dict(type="DefaultFormatBundle"),
#     dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
# ]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root  # annotation file 명을 지정
        + "v5/cv_train_1_minor_v5.json",  # validation을 안나눠서 설정, validation set 설정 시 여기다 해주면 될 듯
        img_prefix=data_root,  # 이미지 파일이 있는 디렉토리의 주소
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + "v5/cv_val_1_minor_v5.json",
        img_prefix=data_root,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + "test.json",
        img_prefix=data_root,
        pipeline=test_pipeline,
    ),
)
# evaluation = dict(interval=1, classwise=True, metric="bbox", save_best="auto")
