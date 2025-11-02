sam_instance_thr = 4 # 认为这个图可以进入SAM的instance数量的阈值
mask_filter_config=dict(
         {
         'default': {
             'required_metrics': ['color_consistency', 'center_alignment'],
             'weights': {'color_consistency': 2, 'center_alignment': 10}
         }
         }
         )
sam_sample_rules = dict({
    "filter_pairs": [(34, 39, 200)]})