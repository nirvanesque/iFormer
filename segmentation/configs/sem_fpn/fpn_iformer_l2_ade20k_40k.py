_base_ = [
    '../_base_/models/fpn_r50.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py',
    # '../_base_/schedules/schedule_40k.py',
]
crop_size = (512, 512)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    size=crop_size,
    seg_pad_val=255)
# model settings
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='iFormer_l2',
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='iFormer_l2_distill.pth',
        ),
        out_indices=(0, 1, 2, 3),
    ),
    neck=dict(in_channels=[64, 128, 256, 512]),
    decode_head=dict(num_classes=150))

gpu_multiples = 2  # we use 8 gpu instead of 4 in mmsegmentation, so lr*2 and max_iters/2
# optimizer
optimizer = dict(type='AdamW', lr=0.0001 * gpu_multiples, weight_decay=0.0001)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
optimizer_config = dict()
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-6,
        power=0.9,
        begin=0,
        end=40000,
        by_epoch=False)
]
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=80000 // gpu_multiples)
checkpoint_config = dict(by_epoch=False, interval=8000 // gpu_multiples)
evaluation = dict(interval=8000 // gpu_multiples, metric='mIoU')

train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))