_base_ = ['h2rbox_v2p-le90_r50_fpn-1x_dota.py']

# load hbox annotations
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='ConvertWeakSupervision',
         point_proportion=0.0,
         hbox_proportion=1.0,
         modify_labels=True),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='RandomRotate', prob=1, angle_range=180),
    dict(type='mmdet.PackDetInputs')
]

train_dataloader = dict(dataset=dict())

data_root = 'data/split_ms_dota/'

train_dataloader = dict(
    dataset=dict(data_root=data_root, pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(data_root=data_root))
test_dataloader = dict(dataset=dict(data_root=data_root))
