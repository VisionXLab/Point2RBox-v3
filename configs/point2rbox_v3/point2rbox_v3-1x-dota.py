_base_ = [
    '../_base_/datasets/dota.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
angle_version = 'le90'

# !↓在这里控制有没有用SAM!
use_sam = True
# !↑在这里控制有没有用SAM!

# !↓在这里控制有没有用label assign!
label_assign = True
# !↑在这里控制有没有用label assign!

use_class_specific_watershed = False

#控制是否使用“可靠的sigma”来生成gaussian-full和提示框
use_reliable_sigma= False

mask_filter_config=dict(
         {
         'default': {
             'required_metrics': ['color_consistency', 'center_alignment'],
             'weights': {'color_consistency': 2, 'center_alignment': 10}
         },
         # Tennis Court
         7: {
             'required_metrics': ['rectangularity', 'color_consistency','aspect_ratio_reasonableness', 'center_alignment'],
             'weights': {'rectangularity': 6,'aspect_ratio_reasonableness': 5, 'color_consistency': 2, 'center_alignment': 10},
             'aspect_ratio_range': (1, 5)
         },
         # Bridge
         2: {
             'required_metrics': ['rectangularity', 'color_consistency', 'center_alignment'],
             'weights': {'rectangularity': 6, 'color_consistency': 2, 'center_alignment': 10}
         },
         # Ground Track Field, Basketball Court, Soccer Ball Field
         3: {
             'required_metrics': ['rectangularity', 'circularity', 'color_consistency', 
                                  'aspect_ratio_reasonableness', 'center_alignment'],
             'weights': {'rectangularity': 6, 'circularity': -3, 
                         'aspect_ratio_reasonableness': 5, 'color_consistency': 2,'center_alignment': 10},
             'aspect_ratio_range': (1, 5),
             'penalty_circularity': 100
         },
         8: {
             'required_metrics': ['rectangularity', 'circularity', 'color_consistency', 
                                  'aspect_ratio_reasonableness', 'center_alignment'],
             'weights': {'rectangularity': 6, 'circularity': -3, 
                         'aspect_ratio_reasonableness': 5, 'color_consistency': 2, 'center_alignment': 10},
             'aspect_ratio_range': (1, 5),
             'penalty_circularity': 100
         },
         10: {
             'required_metrics': ['rectangularity', 'circularity', 'color_consistency', 
                                  'aspect_ratio_reasonableness', 'center_alignment'],
             'weights': {'rectangularity': 6, 'circularity': -3, 
                         'aspect_ratio_reasonableness': 5, 'color_consistency': 2, 'center_alignment': 10},
             'aspect_ratio_range': (1, 5),
             'penalty_circularity': 100
         },
         # Baseball Diamond
         1: {
             'required_metrics': ['aspect_ratio_reasonableness', 'center_alignment'],
             'weights': {'aspect_ratio_reasonableness': 5, 'color_consistency': 2,'center_alignment': 10},
             'aspect_ratio_range': (1, 5)
         },
         # Roundabout
         11: {
             'required_metrics': ['circularity',  'center_alignment','color_consistency'],
             'weights': {'circularity': 5, 'center_alignment': 10, 'color_consistency': 2},
         },
         # Storage Tank 
         9: {
             'required_metrics': ['circularity',  'center_alignment','color_consistency'],
             'weights': {'circularity': 5, 'center_alignment': 10, 'color_consistency': 2},
         },
         # Plane, Helicopter
         0: {
             'required_metrics': ['aspect_ratio_reasonableness', 'center_alignment','color_consistency'],
             'weights': {'aspect_ratio_reasonableness': 5, 'center_alignment': 10, 'color_consistency': 2},
             'aspect_ratio_range': (1, 5)
         },
         14: {
             'required_metrics': ['aspect_ratio_reasonableness', 'center_alignment', 'color_consistency'],
             'weights': {'aspect_ratio_reasonableness': 5, 'center_alignment': 10, 'color_consistency': 2},
             'aspect_ratio_range': (1, 5)
         }
         }
         )

sam_instance_thr = 4 # 认为这个图可以进入SAM的instance数量的阈值
prompt_sigma_scale = 1 # 用于生成提示框的sigma缩放比例
# ↑注意，这个地方是控制面积的放大倍数，也就是说如果它是4的话，那么长和宽都会放大2倍
bad_watershed_classes=[-2]  # 坏分水岭类在此处配置

if use_sam == False:
    mask_filter_config = None
    sam_instance_thr = -1
    bad_watershed_classes = [-2] # 定义为-2就肯定没有类别可以选中了
# model settings



model = dict(
    type='Point2RBoxV2Fpn' if label_assign else 'Point2RBoxV2',
    ss_prob=[0.68, 0.07, 0.25],
    copy_paste_start_epoch=6,
    **({'label_assign_pseudo_label_switch_eopch': 6} if label_assign else {}),
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
        out_indices=(0, 1, 2, 3) if label_assign else (1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048] if label_assign else [512, 1024, 2048],
        out_channels=256 if label_assign else 128,
        start_level=1 if label_assign else 0,
        add_extra_convs='on_output',
        num_outs=5 if label_assign else 3,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='Point2RBoxV2FpnHead' if label_assign else 'Point2RBoxV2Head',
        num_classes=15,
        in_channels=256 if label_assign else 128,
        feat_channels=256 if label_assign else 128,
        strides=[8, 16, 32, 64, 128] if label_assign else [8],
        **({'use_adaptive_scale': False} if label_assign else {}),
        edge_loss_start_epoch=6,
        joint_angle_start_epoch=1,
        voronoi_type='standard',
        voronoi_thres=dict(
            default=[0.994, 0.005],
            override=(([2, 11], [0.999, 0.6]),
                    ([7, 8, 10, 14], [0.95, 0.005]))),
        square_cls=[1, 9, 11],
        edge_loss_cls=[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13],
        post_process={11: 1.2},
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
            type='GaussianOverlapLoss', loss_weight=10.0, lamb=0,
            # exclusion_rules=[
            #     dict(type='pair_distance', labels=[3, 10], distance_thr=200),
            # ]
            ),
        loss_pgdm=dict(
            type='PgdmLoss', loss_weight_watershed=5.0, loss_weight_sam=5.0,
            sam_checkpoint='./mobile_sam.pt',
            sam_type='vit_t',
            sam_device='cuda',
            mask_filter_config=mask_filter_config,
            sam_instance_thr=sam_instance_thr,
            use_class_specific_watershed=use_class_specific_watershed,
            ),
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
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='ConvertWeakSupervision', point_proportion=1., hbox_proportion=0),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.PackDetInputs')
]

train_dataloader = dict(batch_size=2,
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