import cv2
import torch
from torch import nn
import numpy as np
from mmrotate.registry import MODELS
from .utils import gwd_sigma_loss
try:
    from mobile_sam import sam_model_registry, SamPredictor
except ImportError:
    raise ImportError("Please install MobileSAM: pip install git+https://github.com/ChaoningZhang/MobileSAM.git")

@MODELS.register_module()
class SamLoss(nn.Module):
    def __init__(self, model_checkpoint='./mobile_sam.pt', model_type='vit_t', model_device="cuda",
                 mask_filter_config=None, loss_weight=1, debug=False):
        super(SamLoss, self).__init__()

        sam = sam_model_registry[model_type](checkpoint=model_checkpoint)
        sam.to(model_device)
        self.predictor = SamPredictor(sam)
        
        self.mask_filter_config = mask_filter_config
        self.loss_weight = loss_weight
        self.debug = debug

    def forward(self, pred, label, image):
        loss, self.vis = self.sam_loss_helper(*pred, 
                                label,
                                image)
        return self.loss_weight * loss
    
    def sam_loss_helper(self, mu, sigma, label, image):
        if self.debug:
            print("Entering SAM branch:")
        
        device = mu.device
        img_np = (image - image.min()) / (image.max() - image.min()) * 255.0
        img_np = img_np.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)

        H, W = img_np.shape[:2]
        J = len(mu)

        self.predictor.set_image(img_np)
        points = mu.detach().cpu().numpy()

        markers = torch.full((H, W), J+1, dtype=torch.int32, device=mu.device)

        total_loss = 0.0
        valid_instances = 0
        L, V = torch.linalg.eigh(sigma)

        for j, point in enumerate(points):
            if self.debug:
                print(f"Processing point {j+1}/{J} at {point}")
            
            point_coords, point_labels = self.set_prompt_points(points, j, H, W)

            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=None, # if None，SAM ignore automatically
                multimask_output=True
            )

            masks = self.postprocess_masks(masks)
            
            class_id = label[j].item()
            best_mask_idx, metrics_values, shape_metrics = self.filter_masks(
                masks, scores, class_id, img_np, point, self.mask_filter_config, self.debug
            )

            mask = masks[best_mask_idx]
            mask_tensor = torch.from_numpy(mask).to(mu.device)
            markers[mask_tensor] = j + 1


            xy = mask_tensor.nonzero()[:, (1, 0)].float()  # Obtain the pixel coordinates within the mask
            
            if len(xy) > 0:
                xy_centered = xy - mu[j]
                xy_rotated = V[j].T.matmul(xy_centered[:, :, None])[:, :, 0]

                max_x = torch.max(torch.abs(xy_rotated[:, 0]))
                max_y = torch.max(torch.abs(xy_rotated[:, 1]))
    
                L_target = torch.stack((max_x, max_y)) ** 2
                
                L_diag = torch.diag_embed(L[j])
                L_target_diag = torch.diag_embed(L_target)
                
                instance_loss = gwd_sigma_loss(L_diag.unsqueeze(0), L_target_diag.unsqueeze(0).detach(), reduction='mean')
    
                total_loss += instance_loss
                valid_instances += 1
        
        final_loss = total_loss / max(1, valid_instances)  # 避免除以0
        
        return final_loss, markers

    def set_prompt_points(self, points, j, H, W):

        all_points = []
        all_labels = []
        
        all_points.append([points[j][0], points[j][1]])
        all_labels.append(1)  # 1 for Foreground

        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]

        for dx, dy in offsets:
            px = int(points[j][0] + dx*10)
            py = int(points[j][1] + dy*10)

            if 0 <= px < W and 0 <= py < H:
                all_points.append([px, py])
                all_labels.append(1)  # 1 for Foreground
        
        # other points for Background
        for k in range(J):
            if k != j:     
                all_points.append(points[k])
                all_labels.append(0)
        
        # convert to numpy.array
        point_coords = np.array(all_points)
        point_labels = np.array(all_labels)

        return point_coords, point_labels

    def postprocess_masks(self, masks):
        # 对所有掩码进行开运算处理以去除噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 使用5x5的椭圆形结构元素
        masks_processed = []

        for mask in masks:
            # 1. 先进行开运算去除小噪点
            mask_opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            
            # 2. 使用连通域分析找到最大的区域
            num_labels, labels_conn, stats, centroids = cv2.connectedComponentsWithStats(mask_opened)
            
            # 3. 如果存在连通区域（除了背景）
            if num_labels > 1:
                # 找到最大的非背景连通区域（背景的标签是0）
                # stats的第5列是面积（前4列是x, y, width, height）
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                
                # 创建一个只包含最大连通区域的掩码
                largest_mask = (labels_conn == largest_label)
                masks_processed.append(largest_mask)
            else:
                # 如果没有连通区域，保留原始开运算结果
                masks_processed.append(mask_opened > 0)

        return masks

    def filter_masks(self, masks, scores, class_id, img_np, point, filter_config=None, debug=False):
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
        shape_metrics = [self.calculate_shape_metrics(
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


        # 计算掩码的形状特征函数 - 按需调用
    
    def calculate_shape_metrics(self, mask, required_metrics, original_image=None, aspect_ratio_range=None, prompt_point=None):
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
                    # 使用最小外接圆计算圆度
                    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                    radius= int(radius)
                    
                    height, width = original_image.shape[-2:]
                    
                    # print(f"Image size: {height}x{width}, Circle center: ({x:.2f}, {y:.2f}), Radius: {radius}")
                    
                    
                    tmp_mask = np.zeros((height, width), dtype=np.uint8)
                    center = (int(x), int(y))
                    cv2.circle(tmp_mask, center, radius, 255, -1)
                    
                    min_circle_area = cv2.countNonZero(tmp_mask)
            
                        
                    if min_circle_area == 0:
                        results['circularity'] = 0.0
                    else:
                        # 圆度: 轮廓面积 / 最小外接圆面积, 完美圆形 = 1
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

                    # 获取图像尺寸
                    height, width = original_image.shape[-2:]
                    # print(f"Image size: {height}x{width}, Box points: {box}")
                    # 创建掩码
                    tmp_mask = np.zeros((height, width), dtype=np.uint8)

                    # 在掩码上绘制矩形（会自动被图像边界裁切）
                    cv2.fillPoly(tmp_mask, [box], 255)

                    # 计算矩形区域的面积（被裁切后的）
                    box_area = cv2.countNonZero(tmp_mask)
                    
                    if box_area == 0:
                        results['rectangularity'] = 0.0
                    else:
                        # Rectangularity = contour area / min bounding rect area
                        results['rectangularity'] = contour_area / box_area

        # 新增: 颜色一致度计算 - 添加对输入图像类型的处理
        if 'color_consistency' in required_metrics and original_image is not None:
            # 确保掩码至少有一个像素
            if np.sum(mask) == 0:
                results['color_consistency'] = 0.0
            else:
                # 处理图像类型，确保是numpy数组，并且通道在最后一个维度
                if isinstance(original_image, torch.Tensor):
                    # 转换PyTorch张量为NumPy数组，并确保通道在最后
                    img_np = original_image.detach().cpu().numpy()
                    if len(img_np.shape) == 3 and img_np.shape[0] <= 3:  # 通道在第一个维度 (C,H,W)
                        img_np = np.transpose(img_np, (1, 2, 0))  # 转换为 (H,W,C)
                else:
                    img_np = original_image
                
                # 提取掩码内的像素并考虑到中心点的距离权重
                if len(img_np.shape) == 3:  # 彩色图像
                    # 获取掩码内所有点的坐标
                    y_coords, x_coords = np.where(mask > 0)
                    
                    # 计算每个点到中心点的距离
                    center_y, center_x = prompt_point[1], prompt_point[0]  # 注意坐标轴顺序(x,y)到(行,列)
                    distances = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
                    
                    # 创建距离的权重 - 使用高斯衰减，距离越近权重越高
                    # 使用图像对角线长度的一小部分作为sigma
                    sigma_w = np.sqrt(img_np.shape[0]**2 + img_np.shape[1]**2) * 0.1
                    weights = np.exp(-(distances**2) / (2 * sigma_w**2))
                    
                    # 应用掩码和权重到原始图像的每个通道
                    weighted_stds = []
                    for i in range(img_np.shape[2]):
                        # 提取当前通道掩码内的所有像素值
                        channel_values = img_np[y_coords, x_coords, i]
                        
                        # 如果有足够的像素，计算加权标准差
                        if len(channel_values) > 1:
                            # 计算加权平均值
                            weighted_mean = np.sum(weights * channel_values) / np.sum(weights)
                            # 计算加权标准差
                            weighted_variance = np.sum(weights * (channel_values - weighted_mean)**2) / np.sum(weights)
                            weighted_std = np.sqrt(weighted_variance)
                            weighted_stds.append(weighted_std)
                        else:
                            weighted_stds.append(0)
                    
                    # 计算加权标准差的平均值
                    mean_weighted_std = np.mean(weighted_stds)
                    
                    if mean_weighted_std == 0:  # 避免除以0
                        results['color_consistency'] = 1.0
                    else:
                        # 使用指数函数将标准差映射到[0,1]，标准差越小，一致度越高
                        consistency = np.exp(-mean_weighted_std / 30.0)  # 30是一个经验值，可以根据图像特性调整
                        results['color_consistency'] = consistency
                else:  # 灰度图像
                    # 获取掩码内所有点的坐标
                    y_coords, x_coords = np.where(mask > 0)
                    
                    # 计算每个点到中心点的距离
                    center_y, center_x = prompt_point[1], prompt_point[0]  # 注意坐标轴顺序(x,y)到(行,列)
                    distances = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
                    
                    # 创建距离的权重 - 使用高斯衰减，距离越近权重越高
                    sigma_w = np.sqrt(img_np.shape[0]**2 + img_np.shape[1]**2) * 0.1
                    weights = np.exp(-(distances**2) / (2 * sigma_w**2))
                    
                    # 提取掩码内的像素
                    pixels = img_np[y_coords, x_coords]
                    
                    # 计算加权标准差
                    if len(pixels) > 1:
                        # 计算加权平均值
                        weighted_mean = np.sum(weights * pixels) / np.sum(weights)
                        # 计算加权标准差
                        weighted_variance = np.sum(weights * (pixels - weighted_mean)**2) / np.sum(weights)
                        weighted_std = np.sqrt(weighted_variance)
                        
                        # 使用指数函数将标准差映射到[0,1]
                        consistency = np.exp(-weighted_std / 30.0)
                        results['color_consistency'] = consistency
                    else:
                        results['color_consistency'] = 0.0
        
        # 新增: 长宽比合理度计算
        if 'aspect_ratio_reasonableness' in required_metrics and aspect_ratio_range is not None:
            min_ratio, max_ratio = aspect_ratio_range
            
            # 计算掩码的外接矩形
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                results['aspect_ratio_reasonableness'] = 0.0
            else:
                largest_contour = max(contours, key=cv2.contourArea)
                
                # 获取最小外接矩形
                rect = cv2.minAreaRect(largest_contour)
                width, height = rect[1]
                
                # 确保宽高都大于0
                if width <= 0 or height <= 0:
                    results['aspect_ratio_reasonableness'] = 0.0
                else:
                    # 计算长宽比（始终使用较大值除以较小值，这样比例总是>=1）
                    aspect_ratio = max(width, height) / min(width, height)
                    
                    # 如果长宽比在指定范围内，则合理度为1
                    if min_ratio <= aspect_ratio <= max_ratio:
                        results['aspect_ratio_reasonableness'] = 1.0
                    else:
                        # 计算长宽比偏离程度
                        if aspect_ratio < min_ratio:
                            deviation = min_ratio / aspect_ratio - 1
                        else:  # aspect_ratio > max_ratio
                            deviation = aspect_ratio / max_ratio - 1
                        
                        # 将偏离度转换为合理度分数（使用指数衰减）
                        reasonableness = np.exp(-deviation * 2)  # 2是衰减系数，可以调整
                        results['aspect_ratio_reasonableness'] = reasonableness
        
        # 新增: 中心点重合度计算
        if 'center_alignment' in required_metrics and prompt_point is not None:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                results['center_alignment'] = 0.0
            else:
                largest_contour = max(contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(largest_contour)
                
                # --- 新增逻辑开始 ---
                # 获取矩形的四个顶点
                box = cv2.boxPoints(rect)
                
                # 检查 prompt_point (元组形式) 是否在矩形轮廓内部
                # cv2.pointPolygonTest 返回 >0 (内), 0 (边上), <0 (外)
                is_inside = cv2.pointPolygonTest(box, (prompt_point[0], prompt_point[1]), False)
                
                if is_inside < 0:
                    # 如果点在矩形外部，直接设置一个巨大的惩罚值
                    results['center_alignment'] = -100.0
                else:
                    # 如果点在内部或边上，执行原来的基于距离的打分逻辑
                    center_x, center_y = rect[0]
                    distance = np.sqrt((center_x - prompt_point[0])**2 + (center_y - prompt_point[1])**2)
                    
                    # 附带修复：从mask获取H, W，而不是依赖未定义的全局变量
                    H, W = mask.shape
                    max_dis = np.sqrt((H**2 + W**2))
                    sigma_for_center_alignment = max_dis * 0.05
                    
                    # 使用高斯函数计算得分
                    alignment_score = np.exp(-(distance**2) / (2 * (sigma_for_center_alignment**2)))
                    results['center_alignment'] = alignment_score
                # --- 新增逻辑结束 ---
                
        if self.debug:
            print(f"Calculated shape metrics: {results}")
        return results







    













