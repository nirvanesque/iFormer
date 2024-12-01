_base_ = [
    '_base_/models/mask-rcnn_r50_fpn.py',
    '_base_/datasets/coco_instance.py',
    '_base_/schedules/schedule_1x.py',
    '_base_/default_runtime.py'
]
# optimizer
model = dict(
    backbone=dict(
        type='iFormer_l',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='iFormer_l_distill.pth',
        ),
        out_indices=(0, 1, 2, 3),
    ),
    neck=dict(
        type='FPN',
        in_channels=[48, 96, 256, 384],
        out_channels=256,
        num_outs=5))
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0002, weight_decay=0.05)  # 0.0001
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(_delete_=True, type='AdamW', lr=0.0002, weight_decay=0.05))
optimizer_config = dict(grad_clip=None)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6, #0.001
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]