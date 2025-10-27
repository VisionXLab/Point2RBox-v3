import cv2
import torch
import numpy as np
def calculate_shape_metrics(image, mask, required_metrics, original_image=None, aspect_ratio_range=None, prompt_point=None, debug=False):
    """Calculate shape metrics on demand
    
    Args:
        mask (np.ndarray): Binary mask
        required_metrics (list): List of metrics to calculate, e.g. ['circularity', 'rectangularity']
        original_image (np.ndarray or torch.Tensor, optional): Original image for color consistency calculation
        aspect_ratio_range (tuple, optional): (min_ratio, max_ratio) for aspect ratio reasonableness
        prompt_point (np.ndarray, optional): Original prompt point coordinates [x, y]
            
    Returns:
        dict: Dictionary with requested metrics
    """
    results = {}
    
    # Only calculate required metrics
    if 'circularity' in required_metrics:
        # Calculate mask contours
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            results['circularity'] = 0.0
        else:
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            
            if contour_area == 0:
                results['circularity'] = 0.0
            else:
                (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                radius= int(radius)
                
                height, width = image.shape[-2:]
                
                # print(f"Image size: {height}x{width}, Circle center: ({x:.2f}, {y:.2f}), Radius: {radius}")
                
                
                tmp_mask = np.zeros((height, width), dtype=np.uint8)
                center = (int(x), int(y))
                cv2.circle(tmp_mask, center, radius, 255, -1)
                
                min_circle_area = cv2.countNonZero(tmp_mask)
        
                    
                if min_circle_area == 0:
                    results['circularity'] = 0.0
                else:
                    results['circularity'] = contour_area / min_circle_area

    if 'rectangularity' in required_metrics:
        # Calculate mask contours
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            results['rectangularity'] = 0.0
        else:
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            
            if contour_area == 0:
                results['rectangularity'] = 0.0
            else:
                # Calculate minimum rotated bounding rectangle
                rect = cv2.minAreaRect(largest_contour)
                box = cv2.boxPoints(rect)
                box = np.int64(box)

                height, width = image.shape[-2:]
                # print(f"Image size: {height}x{width}, Box points: {box}")
                tmp_mask = np.zeros((height, width), dtype=np.uint8)

                cv2.fillPoly(tmp_mask, [box], 255)

                box_area = cv2.countNonZero(tmp_mask)
                
                if box_area == 0:
                    results['rectangularity'] = 0.0
                else:
                    # Rectangularity = contour area / min bounding rect area
                    results['rectangularity'] = contour_area / box_area

    if 'color_consistency' in required_metrics and original_image is not None:
        if np.sum(mask) == 0:
            results['color_consistency'] = 0.0
        else:
            if isinstance(original_image, torch.Tensor):
                img_np = original_image.detach().cpu().numpy()
                if len(img_np.shape) == 3 and img_np.shape[0] <= 3:
                    img_np = np.transpose(img_np, (1, 2, 0))
            else:
                img_np = original_image
            
            if len(img_np.shape) == 3:
                y_coords, x_coords = np.where(mask > 0)
                
                center_y, center_x = prompt_point[1], prompt_point[0]
                distances = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
                
                sigma_w = np.sqrt(img_np.shape[0]**2 + img_np.shape[1]**2) * 0.1
                weights = np.exp(-(distances**2) / (2 * sigma_w**2))
                
                weighted_stds = []
                for i in range(img_np.shape[2]):
                    channel_values = img_np[y_coords, x_coords, i]
                    
                    if len(channel_values) > 1:
                        weighted_mean = np.sum(weights * channel_values) / np.sum(weights)
                        weighted_variance = np.sum(weights * (channel_values - weighted_mean)**2) / np.sum(weights)
                        weighted_std = np.sqrt(weighted_variance)
                        weighted_stds.append(weighted_std)
                    else:
                        weighted_stds.append(0)
                
                mean_weighted_std = np.mean(weighted_stds)
                
                if mean_weighted_std == 0:
                    results['color_consistency'] = 1.0
                else:
                    consistency = np.exp(-mean_weighted_std / 30.0)
                    results['color_consistency'] = consistency
            else:
                y_coords, x_coords = np.where(mask > 0)
                
                center_y, center_x = prompt_point[1], prompt_point[0]
                distances = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
                
                sigma_w = np.sqrt(img_np.shape[0]**2 + img_np.shape[1]**2) * 0.1
                weights = np.exp(-(distances**2) / (2 * sigma_w**2))
                
                pixels = img_np[y_coords, x_coords]
                
                if len(pixels) > 1:
                    weighted_mean = np.sum(weights * pixels) / np.sum(weights)
                    weighted_variance = np.sum(weights * (pixels - weighted_mean)**2) / np.sum(weights)
                    weighted_std = np.sqrt(weighted_variance)
                    
                    consistency = np.exp(-weighted_std / 30.0)
                    results['color_consistency'] = consistency
                else:
                    results['color_consistency'] = 0.0
    
    if 'aspect_ratio_reasonableness' in required_metrics and aspect_ratio_range is not None:
        min_ratio, max_ratio = aspect_ratio_range
        
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            results['aspect_ratio_reasonableness'] = 0.0
        else:
            largest_contour = max(contours, key=cv2.contourArea)
            
            rect = cv2.minAreaRect(largest_contour)
            width, height = rect[1]
            
            if width <= 0 or height <= 0:
                results['aspect_ratio_reasonableness'] = 0.0
            else:
                aspect_ratio = max(width, height) / min(width, height)
                
                if min_ratio <= aspect_ratio <= max_ratio:
                    results['aspect_ratio_reasonableness'] = 1.0
                else:
                    if aspect_ratio < min_ratio:
                        deviation = min_ratio / aspect_ratio - 1
                    else:  # aspect_ratio > max_ratio
                        deviation = aspect_ratio / max_ratio - 1
                    
                    reasonableness = np.exp(-deviation * 2)
                    results['aspect_ratio_reasonableness'] = reasonableness
    
    if 'center_alignment' in required_metrics and prompt_point is not None:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            results['center_alignment'] = 0.0
        else:
            largest_contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest_contour)
            
            box = cv2.boxPoints(rect)
            
            is_inside = cv2.pointPolygonTest(box, (prompt_point[0], prompt_point[1]), False)
            
            if is_inside < 0:
                results['center_alignment'] = -100.0
            else:
                center_x, center_y = rect[0]
                distance = np.sqrt((center_x - prompt_point[0])**2 + (center_y - prompt_point[1])**2)
                
                H, W = mask.shape
                max_dis = np.sqrt((H**2 + W**2))
                sigma_for_center_alignment = max_dis * 0.05
                
                alignment_score = np.exp(-(distance**2) / (2 * (sigma_for_center_alignment**2)))
                results['center_alignment'] = alignment_score
            
    if debug:
        print(f"Calculated shape metrics: {results}")
    return results


def filter_masks(image, masks, scores, class_id, img_np, point, filter_config=None, debug=False):
    if filter_config is None:
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
    
    class_config = filter_config.get(class_id, filter_config.get('default'))
    
    required_metrics = class_config.get('required_metrics', [])
    weights = class_config.get('weights', {})
    aspect_ratio_range = class_config.get('aspect_ratio_range', None)
    
    shape_metrics = [calculate_shape_metrics(
        image,
        mask, 
        required_metrics,
        original_image=img_np,
        aspect_ratio_range=aspect_ratio_range,
        prompt_point=point,
        debug=debug
    ) for mask in masks]
    
    metrics_values = []
    for i in range(len(masks)):
        score = 0
        for metric_name, weight in weights.items():
            metric_value = shape_metrics[i].get(metric_name, 0)
            
            if metric_name == 'circularity' and metric_value > 0.8:
                if 'penalty_circularity' in class_config:
                    metric_value = class_config['penalty_circularity']
            
            score += metric_value * weight
        
        metrics_values.append(score)
    
    best_mask_idx = np.argmax(metrics_values)
    
    if debug:
        print(f"Class ID: {class_id}, Best mask: {best_mask_idx}")
        print(f"Metrics: {shape_metrics[best_mask_idx]}")
        print(f"Score: {metrics_values[best_mask_idx]}")
    
    return best_mask_idx, metrics_values, shape_metrics