sam_instance_thr = 4 
mask_filter_config = dict(
    {
        # 默认配置，适用于大多数没有强几何先验的类别
        'default': {
            'required_metrics': ['color_consistency', 'center_alignment'],
            'weights': {'color_consistency': 2, 'center_alignment': 10}
        },
        # 网球场 (16), 篮球场 (3)
        # 特征: 标准的矩形，固定的长宽比。
        16: {
            'required_metrics': ['rectangularity', 'color_consistency', 'aspect_ratio_reasonableness', 'center_alignment'],
            'weights': {'rectangularity': 5, 'aspect_ratio_reasonableness': 10, 'color_consistency': 2, 'center_alignment': 10},
            'aspect_ratio_range': (1.5, 2.5)
        },
        3: {
            'required_metrics': ['rectangularity', 'color_consistency', 'aspect_ratio_reasonableness', 'center_alignment'],
            'weights': {'rectangularity': 5, 'aspect_ratio_reasonableness': 10, 'color_consistency': 2, 'center_alignment': 10},
            'aspect_ratio_range': (1.5, 2.5)
        },
        # 桥梁 (4), 立交桥 (12)
        # 特征: 通常是细长的矩形，长宽比变化较大。
        4: {
            'required_metrics': ['rectangularity', 'color_consistency', 'center_alignment'],
            'weights': {'rectangularity': 5, 'color_consistency': 2, 'center_alignment': 10}
        },
        12: {
            'required_metrics': ['rectangularity', 'color_consistency', 'center_alignment'],
            'weights': {'rectangularity': 5, 'color_consistency': 2, 'center_alignment': 10}
        },
        # 田径场 (10), 体育场 (14)
        10: {
            'required_metrics': ['rectangularity', 'circularity', 'color_consistency', 'aspect_ratio_reasonableness', 'center_alignment'],
            'weights': {'rectangularity': 6, 'circularity': -3, 'aspect_ratio_reasonableness': 5, 'color_consistency': 2, 'center_alignment': 10},
            'aspect_ratio_range': (1.3, 2.5),
            'penalty_circularity': 100
        },
        14: {
            'required_metrics': ['rectangularity', 'circularity', 'color_consistency', 'aspect_ratio_reasonableness', 'center_alignment'],
            'weights': {'rectangularity': 6, 'circularity': -3, 'aspect_ratio_reasonableness': 5, 'color_consistency': 2, 'center_alignment': 10},
            'aspect_ratio_range': (1.0, 3.0),
            'penalty_circularity': 100
        },
        # 储油罐 (15), 风车 (19)
        # 特征: 俯瞰视角下呈圆形。
        15: {
            'required_metrics': ['circularity', 'center_alignment', 'color_consistency'],
            'weights': {'circularity': 8, 'center_alignment': 10, 'color_consistency': 2},
        },
    }
)
sam_sample_rules=None
