sam_instance_thr = 4 # 认为这个图可以进入SAM的instance数量的阈值
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
sam_sample_rules = dict({
    "filter_pairs": [(3, 10, 200)]})