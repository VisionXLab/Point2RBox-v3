from mmdet.models.losses.utils import weighted_loss
from mmrotate.models.losses.gaussian_dist_loss import postprocess

def rbbox2roi(bbox_list):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.

    Returns:
        Tensor: shape (n, 6), [batch_ind, cx, cy, w, h, a]
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :5]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 6))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois


def filter_masks(masks, scores, class_id, img_np, point, filter_config=None, debug=False):
        """基于特定配置筛选最佳掩码
        
        Args:
            masks: SAM生成的掩码列表
            scores: SAM生成的掩码置信度
            class_id: 类别ID
            img_np: 原始图像
            point: 提示点坐标
            filter_config: 筛选配置字典
            debug: 是否输出调试信息
            
        Returns:
            best_mask_idx: 最佳掩码索引
            metrics_values: 所有掩码的评分
            shape_metrics: 所有掩码的特征指标
        """
        if filter_config is None:
            # 使用默认配置
            filter_config = {
                'default': {
                    'required_metrics': ['color_consistency', 'center_alignment'],
                    'weights': {'color_consistency': 6, 'center_alignment': 10}
                },
                # Tennis Court
                7: {
                    'required_metrics': ['rectangularity', 'color_consistency', 
                                        'aspect_ratio_reasonableness', 'center_alignment'],
                    'weights': {'rectangularity': 3, 'color_consistency': 6, 
                            'aspect_ratio_reasonableness': 10, 'center_alignment': 10},
                    'aspect_ratio_range': (1.3, 3)
                },
                # Bridge
                2: {
                    'required_metrics': ['rectangularity', 'color_consistency', 
                                        'aspect_ratio_reasonableness', 'center_alignment'],
                    'weights': {'rectangularity': 3, 'color_consistency': 6, 'center_alignment': 10}
                },
                # Ground Track Field, Basketball Court, Soccer Ball Field
                3: {
                    'required_metrics': ['rectangularity', 'circularity', 'color_consistency', 
                                        'aspect_ratio_reasonableness', 'center_alignment'],
                    'weights': {'rectangularity': 6, 'circularity': -3, 'color_consistency': 6, 
                            'aspect_ratio_reasonableness': 5, 'center_alignment': 10},
                    'aspect_ratio_range': (1, 3),
                    'penalty_circularity': 100
                },
                8: {
                    'required_metrics': ['rectangularity', 'circularity', 'color_consistency', 
                                        'aspect_ratio_reasonableness', 'center_alignment'],
                    'weights': {'rectangularity': 6, 'circularity': -3, 'color_consistency': 6, 
                            'aspect_ratio_reasonableness': 5, 'center_alignment': 10},
                    'aspect_ratio_range': (1, 3),
                    'penalty_circularity': 100
                },
                10: {
                    'required_metrics': ['rectangularity', 'circularity', 'color_consistency', 
                                        'aspect_ratio_reasonableness', 'center_alignment'],
                    'weights': {'rectangularity': 6, 'circularity': -3, 'color_consistency': 6, 
                            'aspect_ratio_reasonableness': 5, 'center_alignment': 10},
                    'aspect_ratio_range': (1, 3),
                    'penalty_circularity': 100
                },
                # Baseball Diamond
                1: {
                    'required_metrics': ['aspect_ratio_reasonableness', 'center_alignment'],
                    'weights': {'aspect_ratio_reasonableness': 5, 'center_alignment': 10},
                    'aspect_ratio_range': (1, 1.3)
                },
                # Roundabout
                11: {
                    'required_metrics': ['circularity', 'rectangularity', 'center_alignment'],
                    'weights': {'circularity': 5, 'rectangularity': -2, 'center_alignment': 10}
                },
                # Plane, Helicopter
                0: {
                    'required_metrics': ['aspect_ratio_reasonableness', 'center_alignment', 'rectangularity'],
                    'weights': {'aspect_ratio_reasonableness': 8, 'center_alignment': 10, 'rectangularity': -1},
                    'aspect_ratio_range': (1.0, 2.0)
                },
                14: {
                    'required_metrics': ['aspect_ratio_reasonableness', 'center_alignment', 'rectangularity'],
                    'weights': {'aspect_ratio_reasonableness': 8, 'center_alignment': 10, 'rectangularity': -1},
                    'aspect_ratio_range': (1.0, 2.0)
                }
            }
        
        # 获取类别配置，如果没有特定配置则使用默认配置
        class_config = filter_config.get(class_id, filter_config.get('default'))
        
        # 获取所需指标和权重
        required_metrics = class_config.get('required_metrics', [])
        weights = class_config.get('weights', {})
        aspect_ratio_range = class_config.get('aspect_ratio_range', None)
        
        # 计算每个掩码的形状特征
        shape_metrics = [calculate_shape_metrics(
            mask, 
            required_metrics,
            original_image=img_np,
            aspect_ratio_range=aspect_ratio_range,
            prompt_point=point
        ) for mask in masks]
        
        # 计算每个掩码的评分
        metrics_values = []
        for i in range(len(masks)):
            score = 0
            for metric_name, weight in weights.items():
                metric_value = shape_metrics[i].get(metric_name, 0)
                
                # 应用特殊逻辑，例如对圆度的惩罚
                if metric_name == 'circularity' and metric_value > 0.85:
                    if 'penalty_circularity' in class_config:
                        metric_value = class_config['penalty_circularity']
                
                score += metric_value * weight
            
            metrics_values.append(score)
        
        # 选择评分最高的掩码
        best_mask_idx = np.argmax(metrics_values)
        
        if debug:
            print(f"Class ID: {class_id}, Best mask: {best_mask_idx}")
            print(f"Metrics: {shape_metrics[best_mask_idx]}")
            print(f"Score: {metrics_values[best_mask_idx]}")
        
        return best_mask_idx, metrics_values, shape_metrics

@weighted_loss
def gwd_sigma_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, normalize=True):
    """Gaussian Wasserstein distance loss.
    Modified from gwd_loss. 
    gwd_sigma_loss only involves sigma in Gaussian, with mu ignored.

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Corresponding gt bboxes.
        fun (str): The function applied to distance. Defaults to 'log1p'.
        tau (float): Defaults to 1.0.
        alpha (float): Defaults to 1.0.
        normalize (bool): Whether to normalize the distance. Defaults to True.

    Returns:
        loss (torch.Tensor)

    """
    Sigma_p = pred
    Sigma_t = target

    whr_distance = Sigma_p.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    whr_distance = whr_distance + Sigma_t.diagonal(
        dim1=-2, dim2=-1).sum(dim=-1)

    _t_tr = (Sigma_p.bmm(Sigma_t)).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    _t_det_sqrt = (Sigma_p.det() * Sigma_t.det()).clamp(1e-7).sqrt()
    whr_distance = whr_distance + (-2) * (
        (_t_tr + 2 * _t_det_sqrt).clamp(1e-7).sqrt())

    distance = (alpha * alpha * whr_distance).clamp(1e-7).sqrt()

    if normalize:
        scale = 2 * (
            _t_det_sqrt.clamp(1e-7).sqrt().clamp(1e-7).sqrt()).clamp(1e-7)
        distance = distance / scale

    return postprocess(distance, fun=fun, tau=tau)