_base_ = [
    '../_base_/datasets/sku110k.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
angle_version = 'le90'

# model settings
model = dict(
    type='Point2RBoxV2',
    ss_prob=[0.68, 0.07, 0.25],
    copy_paste_start_epoch=6,
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
        out_channels=128,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=3,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='Point2RBoxV2Head',
        num_classes=1,
        in_channels=128,
        feat_channels=128,
        strides=[8],
        edge_loss_start_epoch=6,
        joint_angle_start_epoch=1,
        voronoi_type='gaussian-orientation',
        voronoi_thres=dict(default=[0.995, 0.01]),
        square_cls=[],
        edge_loss_cls=[0],
        post_process={},
        angle_coder=dict(
            type='PSCCoder',
            angle_version='le90',
            dual_freq=False,
            num_step=3,
            thr_mod=0),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GDLoss', loss_type='gwd', loss_weight=5.0),
        loss_overlap=dict(
            type='GaussianOverlapLoss', loss_weight=10.0, lamb=0),
        loss_voronoi=dict(
            type='VoronoiWatershedLoss', loss_weight=5.0),
        loss_bbox_edg=dict(
            type='EdgeLoss', loss_weight=0.3),
        loss_ss=dict(
            type='Point2RBoxV2ConsistencyLoss', loss_weight=1.0)),
    # training and testing settings
    train_cfg=None,
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=2000))

# load point annotations
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='rbox'),
    dict(type='mmdet.FixShapeResize', width=800, height=800, keep_ratio=True),
    dict(type='ConvertWeakSupervision', point_proportion=1., hbox_proportion=0),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='RandomRotate', prob=1, angle_range=180),
    dict(type='mmdet.PackDetInputs')
]

train_dataloader = dict(batch_size=4,
                        dataset=dict(pipeline=train_pipeline))

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.00005,
        betas=(0.9, 0.999),
        weight_decay=0.05))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=12)
custom_hooks = [dict(type='mmdet.SetEpochInfoHook')]
