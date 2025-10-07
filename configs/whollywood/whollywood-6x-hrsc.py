_base_ = [
    '../_base_/datasets/hrsc.py', '../_base_/schedules/schedule_6x.py',
    '../_base_/default_runtime.py'
]
angle_version = 'le90'

# model settings
model = dict(
    type='WhollyWoodP2R',
    crop_size=[800, 800],
    basic_pattern='basic_patterns/hrsc',
    dense_cls=[],
    square_cls=[],
    use_setrc=False,
    use_setsk=True,
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        boxtype2tensor=False),
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='WhollyWoodP2RHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8],
        regress_ranges=[(-1, 1e8)],
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=True,
        centerness_on_reg=True,
        use_hbbox_loss=False,
        use_hbox_output=False,
        use_transform_ss=True,
        scale_angle=True,
        pseudo_generator=False,
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version=angle_version),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='RotatedIoULoss', loss_weight=1.0),
        loss_ss_bbox=dict(type='mmdet.SmoothL1Loss', loss_weight=0.005, beta=1),
        angle_coder=dict(
            type='PSCCoder',
            angle_version=angle_version,
            dual_freq=False,
            num_step=3),
        loss_angle=dict(type='mmdet.L1Loss', loss_weight=0.05),
        loss_ss_symmetry=dict(type='mmdet.SmoothL1Loss', loss_weight=1.0, beta=0.1),
        loss_centerness=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=0.0)),
    # training and testing settings
    train_cfg=None,
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=2000))

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='mmdet.FixShapeResize', width=800, height=800, keep_ratio=True),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='ConvertWeakSupervision',
         point_proportion=1.0,
         hbox_proportion=0.0,
         modify_labels=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='RandomRotate', prob=1, angle_range=180),
    dict(type='mmdet.RandomShift', prob=0.5, max_shift_px=16),
    dict(type='mmdet.PackDetInputs')
]

train_dataloader = dict(
    batch_size=2,
    dataset=dict(pipeline=train_pipeline))

val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='mmdet.FixShapeResize', width=800, height=800, keep_ratio=True),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict(
    batch_size=2,
    dataset=dict(pipeline=val_pipeline))

test_dataloader = val_dataloader

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.00005,
        betas=(0.9, 0.999),
        weight_decay=0.05))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=72, val_interval=12)
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=12))
