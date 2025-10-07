_base_ = 'whollywood-1x-dota-p2r-pseudo.py'

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    # This modify labels to (cls, ws) format. ws: 0=RBox, 1=HBox, 2=Point
    dict(type='ConvertWeakSupervision',
         point_proportion=1.0,
         hbox_proportion=0.0,
         modify_labels=True),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.RandomShift', prob=0.5, max_shift_px=16),
    dict(type='RandomRotate', prob=1, angle_range=180),
    dict(type='mmdet.PackDetInputs')
]

test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='ConvertWeakSupervision',
         point_proportion=1.0,
         hbox_proportion=0.0,
         modify_labels=True),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='mmdet.PackDetInputs')
]

data_root = 'data/split_ms_dota/'

train_dataloader = dict(
    dataset=dict(data_root=data_root, pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(data_root=data_root))
test_dataloader = dict(
    dataset=dict(data_root=data_root, pipeline=test_pipeline))

test_evaluator = dict(
    outfile_prefix='data/split_ms_dota/whollywood_pseudo_labels')
