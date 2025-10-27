# Copyright (c) OpenMMLab. All rights reserved.
import math
from click import prompt
from pandas import Timestamp
from sympy import im
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torch import Tensor
from mmdet.models.losses.utils import weighted_loss

from mmrotate.registry import MODELS
from mmrotate.models.losses.gaussian_dist_loss import postprocess
from mmrotate.models.losses.utils import filter_masks
from mmrotate.models.losses.vis import plot_gaussian_voronoi_watershed, visualize_loss_calculation, save_debug_visualization


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
    ...
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
                 lamb=1e-4,
                 ):
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
            overlap_scale (torch.Tensor, optional): scale matrix for overlap loss with shape(N, N).

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        assert len(pred[0]) == len(pred[1])
        
        mu, sigma = pred
        J = mu.shape[0]

        L = torch.linalg.eigh(sigma)[0].clamp(1e-7).sqrt()
        loss_lamb = F.l1_loss(L, torch.zeros_like(L), reduction='none')
        loss_lamb = self.lamb * loss_lamb.log1p().mean()
        
        overlap_loss = gaussian_overlap_loss(
            pred,
            None,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            overlap_scale=overlap_scale,
        )

        return self.loss_weight * (loss_lamb + overlap_loss)


def gaussian_2d(xy, mu, sigma, normalize=False):
    dxy = (xy - mu).unsqueeze(-1)
    t0 = torch.exp(-0.5 * dxy.permute(0, 2, 1).bmm(torch.linalg.solve(sigma, dxy)))
    if normalize:
        t0 = t0 / (2 * np.pi * sigma.det().clamp(1e-7).sqrt())
    return t0


def sigma_to_rbox_params(sigma: torch.Tensor):
    if not (sigma.shape == (2, 2)):
        raise ValueError("输入必须是一个 (2, 2) 的张量")
    L, V = torch.linalg.eigh(sigma)
    W_rotated = 2 * torch.sqrt(L[1])
    H_rotated = 2 * torch.sqrt(L[0])
    major_axis_vector = V[:, 1]
    angle_rad = torch.atan2(major_axis_vector[1], major_axis_vector[0])
    return W_rotated, H_rotated, angle_rad

def _get_box_prompt_from_gaussian(mu_j, sigma_j, sigma_scale=1, ellipse_scale_factor=1):
    W_base, H_base, angle_rad = sigma_to_rbox_params(sigma_j)
    
    scale_factor_from_sigma = math.sqrt(sigma_scale)
    
    final_scale_factor = scale_factor_from_sigma * ellipse_scale_factor
    
    semi_axis_a = (W_base / 2) * final_scale_factor
    semi_axis_b = (H_base / 2) * final_scale_factor

    cos_theta = torch.cos(angle_rad)
    sin_theta = torch.sin(angle_rad)
    
    half_width_bbox = torch.sqrt((semi_axis_a * cos_theta)**2 + (semi_axis_b * sin_theta)**2)
    half_height_bbox = torch.sqrt((semi_axis_a * sin_theta)**2 + (semi_axis_b * cos_theta)**2)

    mu_x, mu_y = mu_j[0], mu_j[1]
    x_min = mu_x - half_width_bbox
    y_min = mu_y - half_height_bbox
    x_max = mu_x + half_width_bbox
    y_max = mu_y + half_height_bbox
    
    bbox_prompt = torch.stack([x_min, y_min, x_max, y_max], dim=-1).detach().cpu().numpy()
    
    return bbox_prompt.reshape(1, 4)


def segment_anything(image, mu, sigma, device=None, sam_checkpoint=None, model_type=None, label=None, debug=False, mask_filter_config=None, sam_sample_rules=None):
    if debug:
        print("Entering SAM branch:")
    try:
        from mobile_sam import sam_model_registry, SamPredictor
        import numpy as np
        import os
        import time
        from PIL import Image
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Please install MobileSAM: pip install git+https://github.com/ChaoningZhang/MobileSAM.git")

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
            raise ValueError("未找到MobileSAM检查点，请指定sam_checkpoint参数")

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
    for j, point in enumerate(points):
        if debug:
            print(f"Processing point {j+1}/{J} at {point}")

        box_prompt = None

        all_points = []
        all_labels = []
        
        all_points.append(point)
        all_labels.append(1)
        
        for k in range(J):
            if k != j:
                if sam_sample_rules is not None:
                    skip = False
                    j_label = label[j].item()
                    k_label = label[k].item()
                    dist = np.sqrt(((points[j] - points[k]) ** 2).sum())
                    for filter_pair in sam_sample_rules["filter_pairs"]:
                        class_id1, class_id2, dist_thr = filter_pair
                        if ((j_label == class_id1 and k_label == class_id2) or (j_label == class_id2 and k_label == class_id1)) \
                            and dist < dist_thr:
                            skip = True
                            break
                    if skip:
                        continue
                    
                all_points.append(points[k])
                all_labels.append(0)
                
        point_coords = np.array(all_points)
        point_labels = np.array(all_labels)

        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_prompt,
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
            image, masks, scores, class_id, img_np, point, mask_filter_config, debug
        )
        if debug:
            save_debug_visualization(image, masks, scores, shape_metrics, metrics_values, 
                                    best_mask_idx, class_id, "Optimized Mask Selection")
        
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
            
            if debug:
                visualize_loss_calculation(
                    image, mask_tensor, mu[j], V[j], 
                    xy_centered, xy_rotated, max_x, max_y, 
                    L[j], L_target, instance_loss, 
                    j, class_id
                )
        
            total_loss += instance_loss
            valid_instances += 1
            
    final_loss = total_loss / max(1, valid_instances)
    
    return final_loss, markers




def voronoi_watershed_loss(mu, sigma, label, image, pos_thres=0.994, neg_thres=0.005, down_sample=2, topk=0.95, default_sigma=4096, voronoi='gaussian-orientation', alpha=0.1, debug=False):
    
    J = len(sigma)
    if J == 0:
        return sigma.sum()
    D = down_sample
    H, W = image.shape[-2:]
    if debug:
        print(f'Gaussian Voronoi Watershed Loss: {H}x{W}, downsample={D}, J={J}')
        print(f'default_sigma={default_sigma}, voronoi={voronoi}, alpha={alpha}')
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

    cls_bg = torch.where(vor == J + 1, 16, cls)
    cls_bg = torch.where(vor == 0, -1, cls_bg)

    # PyTorch does not support watershed, use cv2
    img_uint8 = (image - image.min()) / (image.max() - image.min()) * 255
    img_uint8 = img_uint8.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    img_uint8 = cv2.medianBlur(img_uint8, 3)
    markers = vor.detach().cpu().numpy().astype(np.int32)
    markers = vor.new_tensor(cv2.watershed(img_uint8, markers))
    if debug:
        plot_gaussian_voronoi_watershed(image, cls_bg, markers, labels=label)

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

def get_loss_from_mask(mu, sigma, label, image, pos_thres, neg_thres, down_sample=2, topk=0.95, default_sigma=4096, voronoi='gaussian-orientation', alpha=0.1, debug=False, mask_filter_config=None,
# sam_checkpoint='./sam_vit_h_4b8939.pth',model_type='vit_h',
sam_checkpoint='./mobile_sam.pt',model_type='vit_t',
sam_instance_thr=-1,
device=None,
sam_sample_rules=None): 
    if debug:
        print(f"SAM config: checkpoint={sam_checkpoint}, model_type={model_type}, sam_instance_thr={sam_instance_thr}")
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
            mask_filter_config=mask_filter_config,
            sam_sample_rules=sam_sample_rules,
        )
        vor = markers.clone()
        return loss, (vor, markers)
    else:
        loss, (vor, markers) = voronoi_watershed_loss(
            mu, sigma, label, image, 
            pos_thres, neg_thres, down_sample, topk, 
            default_sigma, voronoi, alpha,
            debug=debug,
        )
        return loss, (vor, markers)
  


@MODELS.register_module()
class VoronoiWatershedLoss(nn.Module):
    """VoronoiWatershedLoss.
    """

    def __init__(self,
                 loss_weight=1.0,
                 down_sample=2,
                 topk=0.95,
                 alpha=0.1,
                 default_sigma=4096,
                 debug=False,
                 mask_filter_config=None,
                 sam_instance_thr=-1,
                 sam_sample_rules=None,
                 use_class_specific_watershed=False
                 ):
        super(VoronoiWatershedLoss, self).__init__()
        self.loss_weight = loss_weight
        self.down_sample = down_sample
        self.topk = topk
        self.alpha = alpha
        self.default_sigma = default_sigma
        self.debug = debug
        self.mask_filter_config = mask_filter_config
        self.sam_instance_thr = sam_instance_thr
        self.sam_sample_rules = sam_sample_rules
        self.use_class_specific_watershed = use_class_specific_watershed
        self.vis = None
        

    def forward(self, pred, label, image, pos_thres, neg_thres, voronoi='orientation'):
        loss, self.vis = get_loss_from_mask(
        *pred, 
        label,
        image, 
        pos_thres, 
        neg_thres, 
        self.down_sample, 
        default_sigma=self.default_sigma,
        topk=self.topk,
        voronoi=voronoi,
        alpha=self.alpha,
        debug=self.debug,
        mask_filter_config=self.mask_filter_config,
        sam_instance_thr=self.sam_instance_thr,
        sam_sample_rules=self.sam_sample_rules)
        return self.loss_weight * loss

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
        if bboxes.size(0) > 0 :
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
