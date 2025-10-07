_base_ = 'whollywood-1x-dota-using-pseudo.py'

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='rbox'),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='RandomRotate', prob=1, angle_range=180),
    dict(type='mmdet.PackDetInputs')
]

data_root = 'data/split_ms_dota/'

train_dataloader = dict(
    dataset=dict(data_root=data_root, pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(data_root=data_root))
test_dataloader = dict(dataset=dict(data_root=data_root))
