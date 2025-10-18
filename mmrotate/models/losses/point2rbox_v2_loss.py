# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from mmdet.models.losses.utils import weighted_loss

from mmrotate.registry import MODELS
from mmrotate.models.losses.gaussian_dist_loss import postprocess


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


def bhattacharyya_coefficient(pred, target):
    """Calculate bhattacharyya coefficient between 2-D Gaussian distributions.

    Args:
        pred (Tuple): tuple of (xy, sigma).
            xy (torch.Tensor): center point of 2-D Gaussian distribution
                with shape (N, 2).
            sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
                with shape (N, 2, 2).
        target (Tuple): tuple of (xy, sigma).

    Returns:
        coef (Tensor): bhattacharyya coefficient with shape (N,).
    """
    xy_p, Sigma_p = pred
    xy_t, Sigma_t = target

    _shape = xy_p.shape

    xy_p = xy_p.reshape(-1, 2)
    xy_t = xy_t.reshape(-1, 2)
    Sigma_p = Sigma_p.reshape(-1, 2, 2)
    Sigma_t = Sigma_t.reshape(-1, 2, 2)

    Sigma_M = (Sigma_p + Sigma_t) / 2
    dxy = (xy_p - xy_t).unsqueeze(-1)
    t0 = torch.exp(-0.125 * dxy.permute(0, 2, 1).bmm(torch.linalg.solve(Sigma_M, dxy)))
    t1 = (Sigma_p.det() * Sigma_t.det()).clamp(1e-7).sqrt()
    t2 = Sigma_M.det()

    coef = t0 * (t1 / t2).clamp(1e-7).sqrt()[..., None, None]
    coef = coef.reshape(_shape[:-1])
    return coef


@weighted_loss
def gaussian_overlap_loss(pred, target, alpha=0.01, beta=0.6065, overlap_scale=None):
    """Calculate Gaussian overlap loss based on bhattacharyya coefficient.

    Args:
        pred (Tuple): tuple of (xy, sigma).
            xy (torch.Tensor): center point of 2-D Gaussian distribution
                with shape (N, 2).
            sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
                with shape (N, 2, 2).

    Returns:
        loss (Tensor): overlap loss with shape (N, N).
    """
    mu, sigma = pred
    B = mu.shape[0]
    mu0 = mu[None].expand(B, B, 2)
    sigma0 = sigma[None].expand(B, B, 2, 2)
    mu1 = mu[:, None].expand(B, B, 2)
    sigma1 = sigma[:, None].expand(B, B, 2, 2)
    loss = bhattacharyya_coefficient((mu0, sigma0), (mu1, sigma1))
    if overlap_scale is not None:
        loss = torch.mul(loss, overlap_scale) * overlap_scale.numel() / torch.relu(overlap_scale).sum()
    loss[torch.eye(B, dtype=bool)] = 0
    loss = F.leaky_relu(loss - beta, negative_slope=alpha) + beta * alpha
    loss = loss.sum(-1)
    return loss


@MODELS.register_module()
class GaussianOverlapLoss(nn.Module):
    """Gaussian Overlap Loss.

    Args:
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 lamb=1e-4):
        super(GaussianOverlapLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.lamb = lamb

    def forward(self,
                pred,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                overlap_scale=None):
        """Forward function.

        Args:
            pred (Tuple): tuple of (xy, sigma).
                xy (torch.Tensor): center point of 2-D Gaussian distribution
                    with shape (N, 2).
                sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
                    with shape (N, 2, 2).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        assert len(pred[0]) == len(pred[1])

        sigma = pred[1]
        L = torch.linalg.eigh(sigma)[0].clamp(1e-7).sqrt()
        loss_lamb = F.l1_loss(L, torch.zeros_like(L), reduction='none')
        loss_lamb = self.lamb * loss_lamb.log1p().mean()
        
        return self.loss_weight * (loss_lamb + gaussian_overlap_loss(
            pred,
            None,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            overlap_scale=overlap_scale
            ))


def plot_gaussian_voronoi_watershed(*images):
    """Plot figures for debug."""
    import matplotlib.pyplot as plt
    plt.figure(dpi=300, figsize=(len(images) * 4, 4))
    plt.tight_layout()
    fileid = np.random.randint(0, 20)
    for i in range(len(images)):
        img = images[i]
        img = (img - img.min()) / (img.max() - img.min())
        if img.dim() == 3:
            img = img.permute(1, 2, 0)
        img = img.detach().cpu().numpy()
        plt.subplot(1, len(images), i + 1)
        if i == 3:
            plt.imshow(img)
            x = np.linspace(0, 1024, 1024)
            y = np.linspace(0, 1024, 1024)
            X, Y = np.meshgrid(x, y)
            plt.contourf(X, Y, img, levels=8, cmap=plt.get_cmap('magma'))
        else:
            plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    plt.savefig(f'debug/Gaussian-Voronoi-{fileid}.png')
    plt.close()


def gaussian_2d(xy, mu, sigma, normalize=False):
    dxy = (xy - mu).unsqueeze(-1)
    t0 = torch.exp(-0.5 * dxy.permute(0, 2, 1).bmm(torch.linalg.solve(sigma, dxy)))
    if normalize:
        t0 = t0 / (2 * np.pi * sigma.det().clamp(1e-7).sqrt())
    return t0


def gaussian_voronoi_watershed_loss(mu, sigma,
                                    label, image, 
                                    pos_thres, neg_thres, 
                                    down_sample=2, topk=0.95, 
                                    default_sigma=4096,
                                    voronoi='gaussian-orientation',
                                    alpha=0.1,
                                    debug=False):
    J = len(sigma)
    if J == 0:
        return sigma.sum()
    
    D = down_sample
    H, W = image.shape[-2:]
    h, w = H // D, W // D
    x = torch.linspace(0, h, h, device=mu.device)
    y = torch.linspace(0, w, w, device=mu.device)
    xy = torch.stack(torch.meshgrid(x, y, indexing='xy'), -1)
    vor = mu.new_zeros(J, h, w)
    # Get distribution for each instance
    mm = (mu.detach() / D).round()
    if voronoi == 'standard':
        sg = sigma.new_tensor((default_sigma, 0, 0, default_sigma)).reshape(2, 2)
        sg = sg / D ** 2
        for j, m in enumerate(mm):
            vor[j] = gaussian_2d(xy.view(-1, 2), m[None], sg[None]).view(h, w)
    elif voronoi == 'gaussian-orientation':
        L, V = torch.linalg.eigh(sigma)
        L = L.detach().clone()
        L = L / (L[:, 0:1] * L[:, 1:2]).sqrt() * default_sigma
        sg = V.matmul(torch.diag_embed(L)).matmul(V.permute(0, 2, 1)).detach()
        sg = sg / D ** 2
        for j, (m, s) in enumerate(zip(mm, sg)):
            vor[j] = gaussian_2d(xy.view(-1, 2), m[None], s[None]).view(h, w)
    elif voronoi == 'gaussian-full':
        sg = sigma.detach() / D ** 2
        for j, (m, s) in enumerate(zip(mm, sg)):
            vor[j] = gaussian_2d(xy.view(-1, 2), m[None], s[None]).view(h, w)
    # val: max prob, vor: belong to which instance, cls: belong to which class
    val, vor = torch.max(vor, 0)
    if D > 1:
        vor = vor[:, None, :, None].expand(-1, D, -1, D).reshape(H, W)
        val = F.interpolate(
            val[None, None], (H, W), mode='bilinear', align_corners=True)[0, 0]
    cls = label[vor]
    kernel = val.new_ones((1, 1, 3, 3))
    kernel[0, 0, 1, 1] = -8
    ridges = torch.conv2d(vor[None].float(), kernel, padding=1)[0] != 0
    vor += 1
    pos_thres = val.new_tensor(pos_thres)
    neg_thres = val.new_tensor(neg_thres)
    vor[val < pos_thres[cls]] = 0
    vor[val < neg_thres[cls]] = J + 1
    vor[ridges] = J + 1

    cls_bg = torch.where(vor == J + 1, 15, cls)
    cls_bg = torch.where(vor == 0, -1, cls_bg)

    # PyTorch does not support watershed, use cv2
    img_uint8 = (image - image.min()) / (image.max() - image.min()) * 255
    img_uint8 = img_uint8.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    img_uint8 = cv2.medianBlur(img_uint8, 3)
    markers = vor.detach().cpu().numpy().astype(np.int32)
    markers = vor.new_tensor(cv2.watershed(img_uint8, markers))

    if debug:
        plot_gaussian_voronoi_watershed(image, cls_bg, markers)

    L, V = torch.linalg.eigh(sigma)
    L_target = []
    for j in range(J):
        xy = (markers == j + 1).nonzero()[:, (1, 0)].float()
        if len(xy) == 0:
            L_target.append(L[j].detach())
            continue
        xy = xy - mu[j]
        xy = V[j].T.matmul(xy[:, :, None])[:, :, 0]
        max_x = torch.max(torch.abs(xy[:, 0]))
        max_y = torch.max(torch.abs(xy[:, 1]))
        L_target.append(torch.stack((max_x, max_y)) ** 2)
    L_target = torch.stack(L_target)
    L = torch.diag_embed(L)
    L_target = torch.diag_embed(L_target)
    loss = gwd_sigma_loss(L, L_target.detach(), reduction='none')
    loss = torch.topk(loss, int(np.ceil(len(loss) * topk)), largest=False)[0].mean()
    return loss, (vor, markers)


@MODELS.register_module()
class VoronoiWatershedLoss(nn.Module):
    """Gaussian Overlap Loss.

    Args:
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """

    def __init__(self,
                 down_sample=2,
                 reduction='mean',
                 loss_weight=1.0,
                 topk=0.95,
                 alpha=0.1,
                 debug=False,
                 mask_filter_config=None,
                 sam_instance_thr=-1,
                 use_class_specific_watershed=False):
        super(VoronoiWatershedLoss, self).__init__()
        self.down_sample = down_sample
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.topk = topk
        self.alpha = alpha
        self.debug = debug
        self.mask_filter_config = mask_filter_config
        self.sam_instance_thr = sam_instance_thr
        self.use_class_specific_watershed = use_class_specific_watershed

    def forward(self, pred, label, image, pos_thres, neg_thres, voronoi='orientation'):
        """Forward function.

        Args:
            pred (Tuple): Tuple of (xy, sigma).
                xy (torch.Tensor): Center point of 2-D Gaussian distribution
                    with shape (N, 2).
                sigma (torch.Tensor): Covariance matrix of 2-D Gaussian distribution
                    with shape (N, 2, 2).
            image (torch.Tensor): The image for watershed with shape (3, H, W).
            standard_voronoi (bool, optional): Use standard or Gaussian voronoi.

        Returns:
            torch.Tensor: The calculated loss
        """
        loss, self.vis = get_loss_from_mask(*pred, 
                                            label,
                                            image, 
                                            pos_thres, 
                                            neg_thres, 
                                            self.down_sample, 
                                            topk=self.topk,
                                            voronoi=voronoi,
                                            alpha=self.alpha,
                                            debug=self.debug,
                                            mask_filter_config=self.mask_filter_config,
                                            sam_instance_thr=self.sam_instance_thr)
        return self.loss_weight * loss

def get_loss_from_mask(mu, sigma, label, image, pos_thres, neg_thres,
                       down_sample=2, topk=0.95, voronoi='gaussian-orientation',
                       alpha=0.1, debug=False, mask_filter_config=None, sam_instance_thr=-1,
                       sam_checkpoint='./mobile_sam.pt',model_type='vit_t', device=None): 
    J = len(sigma)
    if J == 0:
        return sigma.sum()
    if J <= sam_instance_thr:
        loss, markers = segment_anything(
            image, mu, sigma,
            device=mu.device, 
            sam_checkpoint=sam_checkpoint,
            model_type=model_type,
            label=label,
            debug=debug,
            mask_filter_config=mask_filter_config)
        vor = markers.clone()
        return loss, (vor, markers)
    else:
        loss, (vor, markers) = gaussian_voronoi_watershed_loss(
            mu, sigma, label, image, 
            pos_thres, neg_thres, down_sample, topk, 
            voronoi, alpha,
            debug=debug,
        )
        return loss, (vor, markers)


def segment_anything(image, mu, sigma, device=None, sam_checkpoint=None, model_type=None, label=None, debug=False, mask_filter_config=None):
    try:
        from mobile_sam import sam_model_registry, SamPredictor
        import numpy as np
        import os
        import time
        from PIL import Image
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Please install MobileSAM: pip install git+https://github.com/ChaoningZhang/MobileSAM.git")

    def filter_masks(masks, scores, class_id, img_np, point, filter_config=None, debug=False):
        if filter_config is None:
            # default configuration
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
            mask, 
            required_metrics,
            original_image=img_np,
            aspect_ratio_range=aspect_ratio_range,
            prompt_point=point
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

    if device is None:
        device = "cuda"

    img_np = (image - image.min()) / (image.max() - image.min()) * 255.0
    img_np = img_np.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)

    H, W = img_np.shape[:2]
    J = len(mu)

    if sam_checkpoint is None:
        import os
        import time
        from PIL import Image
        from scipy import ndimage
        common_paths = [
            "./mobile_sam.pt"
        ]
        for path in common_paths:
            if os.path.exists(path):
                sam_checkpoint = path
                break
        if sam_checkpoint is None:
            raise ValueError("MobileSAM checkpoint not found. Please specify the sam_checkpoint parameter")

    if not hasattr(segment_anything, 'sam_model') or not hasattr(segment_anything, 'model_type') or segment_anything.model_type != model_type:
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device)
        segment_anything.sam_model = sam
        segment_anything.model_type = model_type
    else:
        sam = segment_anything.sam_model

    predictor = SamPredictor(sam)

    predictor.set_image(img_np)

    points = mu.detach().cpu().numpy()
    markers = torch.full((H, W), J+1, dtype=torch.int32, device=mu.device)

    total_loss = 0.0
    valid_instances = 0
    L, V = torch.linalg.eigh(sigma)


    def calculate_shape_metrics(mask, required_metrics, original_image=None, aspect_ratio_range=None, prompt_point=None):
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

                # cv2.pointPolygonTest 返回 >0 (内), 0 (边上), <0 (外)
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

    for j, point in enumerate(points):
        if debug:
            print(f"Processing point {j+1}/{J} at {point}")

        all_points = []
        all_labels = []

        all_points.append(point)
        all_labels.append(1)

        for k in range(J):
            if k != j:
                # Check if one is soccer field and one is track field
                j_label = label[j].item()
                k_label = label[k].item()
                soccer_track_pair = ((j_label == 10 and k_label == 3) or (j_label == 3 and k_label == 10))

                if soccer_track_pair:
                    # Calculate distance between points
                    dist = np.sqrt(((points[j] - points[k]) ** 2).sum())
                    # If they're close, skip adding this point as background
                    if dist < 200:
                        continue

                all_points.append(points[k])
                all_labels.append(0)

        point_coords = np.array(all_points)
        point_labels = np.array(all_labels)

        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=None,
            multimask_output=True
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        masks_processed = []

        for mask in masks:
            mask_opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

            num_labels, labels_conn, stats, centroids = cv2.connectedComponentsWithStats(mask_opened)

            if num_labels > 1:
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

                largest_mask = (labels_conn == largest_label)
                masks_processed.append(largest_mask)
            else:
                masks_processed.append(mask_opened > 0)

        masks = masks_processed

        best_mask_idx = np.argmax(scores)

        class_id = label[j].item()

        best_mask_idx, metrics_values, shape_metrics = filter_masks(
            masks, scores, class_id, img_np, point, mask_filter_config, debug
        )

        mask = masks[best_mask_idx]

        mask_tensor = torch.from_numpy(mask).to(mu.device)
        markers[mask_tensor] = j + 1

        xy = mask_tensor.nonzero()[:, (1, 0)].float()

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

    final_loss = total_loss / max(1, valid_instances)

    return final_loss, markers

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


def plot_edge_map(feat, edgex, edgey):
    """Plot figures for debug."""
    import matplotlib.pyplot as plt
    plt.figure(dpi=300, figsize=(4, 4))
    plt.tight_layout()
    fileid = np.random.randint(0, 20)
    for i in range(len(feat)):
        img0 = feat[i, :3]
        img0 = (img0 - img0.min()) / (img0.max() - img0.min())
        img1 = edgex[i, :3]
        img1 = (img1 - img1.min()) / (img1.max() - img1.min())
        img2 = edgey[i, :3]
        img2 = (img2 - img2.min()) / (img2.max() - img2.min())
        img3 = img1 + img2
        img3 = (img3 - img3.min()) / (img3.max() - img3.min())
        img = torch.cat((torch.cat((img0, img2), -1), 
                         torch.cat((img1, img3), -1)), -2
                         ).permute(1, 2, 0).detach().cpu().numpy()
        N = int(np.ceil(np.sqrt(len(feat))))
        plt.subplot(N, N, i + 1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    plt.savefig(f'debug/Edge-Map-{fileid}.png')
    plt.close()


@MODELS.register_module()
class EdgeLoss(nn.Module):
    """Edge Loss.

    Args:
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """

    def __init__(self,
                 resolution=24,
                 max_scale=1.6,
                 sigma=6,
                 reduction='mean',
                 loss_weight=1.0,
                 debug=False):
        super(EdgeLoss, self).__init__()
        self.resolution = resolution
        self.max_scale = max_scale
        self.sigma = sigma
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.center_idx = self.resolution / self.max_scale
        self.debug = debug

        self.roi_extractor = MODELS.build(dict(
            type='RotatedSingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlignRotated',
                    out_size=(2 * self.resolution + 1),
                    sample_num=2,
                    clockwise=True),
            out_channels=1,
            featmap_strides=[1],
            finest_scale=1024))

        edge_idx = torch.arange(0, self.resolution + 1)
        edge_distribution = torch.exp(-((edge_idx - self.center_idx) ** 2) / (2 * self.sigma ** 2))
        edge_distribution[0] = edge_distribution[-1] = 0
        self.register_buffer('edge_idx', edge_idx)
        self.register_buffer('edge_distribution', edge_distribution)

    def forward(self, pred, edge):
        """Forward function.

        Args:
            pred (Tuple): Batched predicted rboxes
            edge (torch.Tensor): The edge map with shape (B, 1, H, W).

        Returns:
            torch.Tensor: The calculated loss
        """
        G = self.resolution
        C = self.center_idx
        roi = rbbox2roi(pred)
        roi[:, 3:5] *= self.max_scale
        feat = self.roi_extractor([edge], roi)
        if len(feat) == 0:
            return pred[0].new_tensor(0)
        featx = feat.sum(1).abs().sum(1)
        featy = feat.sum(1).abs().sum(2)
        featx2 = torch.flip(featx[:, :G + 1], (-1,)) + featx[:, G:]
        featy2 = torch.flip(featy[:, :G + 1], (-1,)) + featy[:, G:]  # (N, 25)
        ex = ((featx2 * self.edge_distribution).softmax(1) * self.edge_idx).sum(1) / C
        ey = ((featy2 * self.edge_distribution).softmax(1) * self.edge_idx).sum(1) / C
        exy = torch.stack((ex, ey), -1)
        rbbox_concat = torch.cat(pred, 0)
        
        if self.debug:
            edgex = featx[:, None, None, :].expand(-1, 1, 2 * self.resolution + 1, -1)
            edgey = featy[:, None, :, None].expand(-1, 1, -1, 2 * self.resolution + 1)
            plot_edge_map(feat, edgex, edgey)

        return self.loss_weight * F.smooth_l1_loss(rbbox_concat[:, 2:4], 
                                      (rbbox_concat[:, 2:4] * exy).detach(),
                                      beta=8)


@MODELS.register_module()
class Point2RBoxV2ConsistencyLoss(nn.Module):
    """Consistency Loss.

    Args:
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0):
        super(Point2RBoxV2ConsistencyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, ori_pred, trs_pred, square_mask, aug_type, aug_val):
        """Forward function.

        Args:
            ori_pred (Tuple): (Sigma, theta)
            trs_pred (Tuple): (Sigma, theta)
            square_mask: When True, the angle is ignored
            aug_type: 'rot', 'flp', 'sca'
            aug_val: Rotation or scale value

        Returns:
            torch.Tensor: The calculated loss
        """
        ori_gaus, ori_angle = ori_pred
        trs_gaus, trs_angle = trs_pred

        if aug_type == 'rot':
            rot = ori_gaus.new_tensor(aug_val)
            cos_r = torch.cos(rot)
            sin_r = torch.sin(rot)
            R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
            ori_gaus = R.matmul(ori_gaus).matmul(R.permute(0, 2, 1))
            d_ang = trs_angle - ori_angle - aug_val
        elif aug_type == 'flp':
            ori_gaus = ori_gaus * ori_gaus.new_tensor((1, -1, -1, 1)).reshape(2, 2)
            d_ang = trs_angle + ori_angle
        else:
            sca = ori_gaus.new_tensor(aug_val)
            ori_gaus = ori_gaus * sca
            d_ang = trs_angle - ori_angle
        
        loss_ssg = gwd_sigma_loss(ori_gaus.bmm(ori_gaus), trs_gaus.bmm(trs_gaus))
        d_ang = (d_ang + math.pi / 2) % math.pi - math.pi / 2
        loss_ssa = F.smooth_l1_loss(d_ang, torch.zeros_like(d_ang), reduction='none', beta=0.1)
        loss_ssa = loss_ssa[~square_mask].sum() / max(1, (~square_mask).sum())

        return self.loss_weight * (loss_ssg + loss_ssa)