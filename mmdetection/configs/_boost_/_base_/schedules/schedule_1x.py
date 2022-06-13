# optimizer
optimizer = dict(type="SGD", lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy="step", warmup="linear", warmup_iters=500, warmup_ratio=0.001, step=[26, 38]
)
# lr_config = dict(
#     policy="cyclic",
#     by_epoch=False,
#     target_ratio=(0.1, 1e-4),
#     cyclic_times=5,
#     step_ratio_up=0.5,
# )
runner = dict(type="EpochBasedRunner", max_epochs=12)
