_base_ = [
    '../_base_/models/faster_rcnn_r50_pafpn.py',
    '../_base_/datasets/coco_detection_hardaug_mosaic.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa
model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[96, 192, 384, 768]))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
# lr_config = dict(warmup_iters=1000, step=[8, 11])
runner = dict(max_epochs=100)
# lr_config = dict(
#     _delete_=True,
#     policy="Cyclic",
#     by_epoch=False,
#     target_ratio=(1, 1e-4),  # 0.1 0.00001
#     cyclic_times=1,
#     step_ratio_up=0.5,
#     gamma=0.5
# )

lr_config = dict(
    _delete_=True,
    policy='CosineRestart',
    warmup='linear',
    warmup_iters=1,
    warmup_ratio=1.0 / 10,
    warmup_by_epoch=True,
    # CosineRestarteLrUpdaterHook
    periods=[40],
    restart_weights=[1],
    min_lr_ratio=1e-5)

# lr_config = dict(
#     _delete_=True,
#     policy='cyclic',
#     target_ratio=(10, 1e-4),
#     cyclic_times=1,
#     step_ratio_up=0.4,
# )
# momentum_config = dict(
#     policy='cyclic',
#     target_ratio=(0.85 / 0.95, 1),
#     cyclic_times=1,
#     step_ratio_up=0.4,
# )