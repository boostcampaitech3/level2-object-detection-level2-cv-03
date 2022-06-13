_base_ = [
    "_base_/models/faster_rcnn_r50_fpn.py",  # mask_rcnn은 segmentation 정보가 없어서 사용하지 못함
    "_base_/datasets/coco_detection_aug.py",
    "_base_/schedules/schedule_1x.py",
    "../_base_/default_runtime.py",
]
pretrained = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth"
# pretrained = "_base_/models/swin_large_patch4_window7_224_22k.pth"
model = dict(
    backbone=dict(
        _delete_=True,  # restnet para와 SwinTransformer backbone의 para가 맞지 않아서 미리 지우기 위함 / 기존에 작성된 config를 삭제하는 역할
        type="SwinTransformer",
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type="Pretrained", checkpoint=pretrained),
    ),
    neck=dict(in_channels=[192, 384, 768, 1536]),
)

optimizer = dict(
    _delete_=True,
    type="AdamW",
    lr=0.00002,  # 16 batch 0.0001
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
)
# stepLR update
lr_config = dict(warmup_iters=1000)
# cyclicLR
# lr_config = dict(
#     target_ratio=(0.5, 1e-4),  # 0.1 0.00001
# )

runner = dict(max_epochs=40)
