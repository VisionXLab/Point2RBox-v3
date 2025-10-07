# Copyright (c) OpenMMLab. All rights reserved.
import os, copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import Scale, ConvModule
from mmdet.models.dense_heads import FCOSHead
from mmdet.models.utils import (filter_scores_and_topk, multi_apply,
                                select_single_mlvl, unpack_gt_instances)
from mmdet.structures import SampleList
from mmdet.structures.bbox import cat_boxes
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, reduce_mean)
from mmengine import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmrotate.registry import MODELS, TASK_UTILS
from mmrotate.structures import RotatedBoxes, rbox2qbox, hbox2rbox, rbox2hbox

INF = 1e8


@MODELS.register_module()
class WhollyWoodP2HHead(FCOSHead):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.

    Compared with FCOS head, Rotated FCOS head add a angle branch to
    support rotated object detection.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        angle_version (str): Angle representations. Defaults to 'le90'.
        use_hbbox_loss (bool): If true, use horizontal bbox loss and
            loss_angle should not be None. Default to False.
        scale_angle (bool): If true, add scale to angle pred branch.
            Default to True.
        angle_coder (:obj:`ConfigDict` or dict): Config of angle coder.
        h_bbox_coder (dict): Config of horzional bbox coder,
            only used when use_hbbox_loss is True.
        bbox_coder (:obj:`ConfigDict` or dict): Config of bbox coder. Defaults
            to 'DistanceAnglePointCoder'.
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_bbox (:obj:`ConfigDict` or dict): Config of localization loss.
        loss_centerness (:obj:`ConfigDict`, or dict): Config of centerness loss.
        loss_angle (:obj:`ConfigDict` or dict, Optional): Config of angle loss.

    Example:
        >>> self = RotatedFCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, angle_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 angle_version: str = 'le90',
                 use_hbbox_loss: bool = False,
                 scale_angle: bool = True,
                 use_hbox_output: bool = False,
                 square_cls: list = [1, 9, 11],
                 pseudo_generator: bool = False,
                 post_process: dict = {},
                 angle_coder: ConfigType = dict(type='PseudoAngleCoder'),
                 h_bbox_coder: ConfigType = dict(
                     type='mmdet.DistancePointBBoxCoder'),
                 bbox_coder: ConfigType = dict(type='DistanceAnglePointCoder'),
                 loss_cls: ConfigType = dict(
                     type='mmdet.FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox: ConfigType = dict(
                     type='RotatedIoULoss', loss_weight=1.0),
                 loss_centerness: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_angle: OptConfigType = None,
                 **kwargs):
        self.angle_version = angle_version
        self.use_hbbox_loss = use_hbbox_loss
        self.is_scale_angle = scale_angle
        self.use_hbox_output = use_hbox_output
        self.square_cls = square_cls
        self.pseudo_generator = pseudo_generator
        self.post_process = post_process
        self.angle_coder = TASK_UTILS.build(angle_coder)
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            bbox_coder=bbox_coder,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness,
            **kwargs)
        if loss_angle is not None:
            self.loss_angle = MODELS.build(loss_angle)
        else:
            self.loss_angle = None
        if self.use_hbbox_loss:
            assert self.loss_angle is not None
            self.h_bbox_coder = TASK_UTILS.build(h_bbox_coder)
            
    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self.conv_angle = nn.Conv2d(
            self.feat_channels, self.angle_coder.encode_size, 3, padding=1)
        if self.is_scale_angle:
            self.scale_angle = Scale(1.0)
        self.conv_gate = nn.Conv2d(
            self.feat_channels, 1, 3, padding=1)
        
    def forward(
            self, x: Tuple[Tensor]
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of each level outputs.

            - cls_scores (list[Tensor]): Box scores for each scale level, \
            each is a 4D-tensor, the channel number is \
            num_points * num_classes.
            - bbox_preds (list[Tensor]): Box energies / deltas for each \
            scale level, each is a 4D-tensor, the channel number is \
            num_points * 4.
            - centernesses (list[Tensor]): centerness for each scale level, \
            each is a 4D-tensor, the channel number is num_points * 1.
        """        
        x0 = x[0]
        x1 = nn.functional.interpolate(x[1], x0.shape[-2:], mode='nearest')
        x2 = nn.functional.interpolate(x[2], x0.shape[-2:], mode='nearest')
        x3 = nn.functional.interpolate(x[3], x0.shape[-2:], mode='nearest')
        x4 = nn.functional.interpolate(x[4], x0.shape[-2:], mode='nearest')
        V = torch.stack((x0, x1, x2, x3, x4), -1)

        k0 = self.conv_gate(x0)
        k1 = self.conv_gate(x1)
        k2 = self.conv_gate(x2)
        k3 = self.conv_gate(x3)
        k4 = self.conv_gate(x4)
        K = (torch.stack((k0, k1, k2, k3, k4), -1) / 8).softmax(-1)
        KV = torch.sum(V * K, -1)

        # Use this to test the bin2dec equation
        # K = K.new_tensor([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
        coef_sin = torch.sin(2 * torch.pi * K.new_tensor((0, 1, 2, 3, 4)) / 5)
        coef_cos = torch.cos(2 * torch.pi * K.new_tensor((0, 1, 2, 3, 4)) / 5)
        phase_sin = torch.sum(K * coef_sin, dim=-1)
        phase_cos = torch.sum(K * coef_cos, dim=-1)
        Y = 5 / (2 * torch.pi) * (torch.pi - torch.atan2(phase_sin, -phase_cos))  # In range [0, 5)
        scale_preds = torch.exp2(Y) / 4
        # print(K, Y)

        cls_feat = KV
        reg_feat = KV

        # Use this to turn off fusion & scaling
        # cls_feat = x[0]
        # reg_feat = x[0]
        # scale_preds = 1

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.conv_cls(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = self.conv_reg(reg_feat)
        angle_pred = self.conv_angle(reg_feat)

        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)

        bbox_pred = self.scales[0](bbox_pred).float()
        bbox_pred = bbox_pred.clamp(min=0) * scale_preds
        if self.is_scale_angle:
            angle_pred = self.scale_angle(angle_pred).float()

        return (cls_score,), (bbox_pred,), (angle_pred,), (centerness,)

    def loss_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        angle_preds: List[Tensor],
        centernesses: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            angle_preds (list[Tensor]): Box angle for each scale level, each \
                is a 4D-tensor, the channel number is num_points * encode_size.
            centernesses (list[Tensor]): centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) \
               == len(angle_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        # bbox_targets here is in format t,b,l,r
        # angle_targets is not coded here
        labels, bbox_targets, angle_targets = self.get_targets(
            all_level_points, batch_gt_instances)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds, angle_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        angle_dim = self.angle_coder.encode_size
        flatten_angle_preds = [
            angle_pred.permute(0, 2, 3, 1).reshape(-1, angle_dim)
            for angle_pred in angle_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_angle_preds = torch.cat(flatten_angle_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)[:, 0]
        flatten_ws = torch.cat(labels)[:, 1]
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_angle_targets = torch.cat(angle_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)
        
        # Recalculate num_pos to exclude point annotations
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)
                    & (flatten_ws != 2)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_angle_preds = flatten_angle_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_angle_targets = flatten_angle_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            if self.use_hbox_output:
                bbox_coder = self.bbox_coder
            else:
                if self.use_hbbox_loss:
                    bbox_coder = self.h_bbox_coder
                else:
                    bbox_coder = self.bbox_coder
                    pos_decoded_angle_preds = self.angle_coder.decode(
                        pos_angle_preds, keepdim=True)
                    pos_bbox_preds = torch.cat(
                        [pos_bbox_preds, pos_decoded_angle_preds], dim=-1)
                    pos_bbox_targets = torch.cat(
                        [pos_bbox_targets, pos_angle_targets], dim=-1)
            pos_decoded_bbox_preds = bbox_coder.decode(pos_points,
                                                       pos_bbox_preds)
            pos_decoded_target_preds = bbox_coder.decode(
                pos_points, pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            # To avoid unused params in distributed training
            dummy_loss = pos_angle_preds.sum() + pos_centerness[0].sum()
            loss_bbox = loss_bbox + 0 * dummy_loss
            if self.loss_angle is not None:
                pos_angle_targets = self.angle_coder.encode(pos_angle_targets)
                loss_angle = self.loss_angle(
                    pos_angle_preds, pos_angle_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum() + pos_angle_preds.sum()
            if self.loss_angle is not None:
                loss_angle = pos_angle_preds.sum()

        loss_dict = dict(
                loss_cls=loss_cls,
                loss_bbox=loss_bbox)
        if self.loss_angle:
            loss_dict['loss_angle'] = loss_angle
        return loss_dict

    def get_targets(
        self, points: List[Tensor], batch_gt_instances: InstanceList
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """Compute regression, classification and centerness targets for points
        in multiple images.
        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
        Returns:
            tuple: Targets of each level.
            - concat_lvl_labels (list[Tensor]): Labels of each level.
            - concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                level.
            - concat_lvl_angle_targets (list[Tensor]): Angle targets of \
                each level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list, angle_targets_list = multi_apply(
            self._get_targets_single,
            batch_gt_instances,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        angle_targets_list = [
            angle_targets.split(num_points, 0)
            for angle_targets in angle_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_angle_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            angle_targets = torch.cat(
                [angle_targets[i] for angle_targets in angle_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
            concat_lvl_angle_targets.append(angle_targets)
        return (concat_lvl_labels, concat_lvl_bbox_targets,
                concat_lvl_angle_targets)

    def _get_targets_single(
            self, gt_instances: InstanceData, points: Tensor,
            regress_ranges: Tensor,
            num_points_per_lvl: List[int]) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = len(gt_instances)
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels

        if num_gts == 0:
            return gt_labels.new_full((num_points, 2), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4)), \
                   gt_bboxes.new_zeros((num_points, 1)), \

        areas = gt_bboxes.areas
        if self.use_hbox_output:
            gt_bboxes = hbox2rbox(rbox2hbox(gt_bboxes.tensor))
        else:
            gt_bboxes = gt_bboxes.regularize_boxes(self.angle_version)

        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        points = points[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 5)
        gt_ctr, gt_wh, gt_angle = torch.split(gt_bboxes, [2, 2, 1], dim=2)

        gt_point = (gt_labels[:, 1] == 2)[None].expand(num_points, num_gts)
        if self.use_hbox_output or self.bbox_coder.use_hbox:
            offset = points - gt_ctr
        else:
            cos_angle, sin_angle = torch.cos(gt_angle), torch.sin(gt_angle)
            rot_matrix = torch.cat([cos_angle, sin_angle, -sin_angle, cos_angle],
                                dim=-1).reshape(num_points, num_gts, 2, 2)
            offset = points - gt_ctr
            offset = torch.matmul(rot_matrix, offset[..., None])
            offset = offset.squeeze(-1)

        w, h = gt_wh[..., 0].clone(), gt_wh[..., 1].clone()
        w[gt_point] = 48
        h[gt_point] = 48
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        left = w / 2 + offset_x
        right = w / 2 - offset_x
        top = h / 2 + offset_y
        bottom = h / 2 - offset_y
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            stride = offset.new_zeros(offset.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            inside_center_bbox_mask = (abs(offset) < stride).all(dim=-1)
            inside_gt_bbox_mask = torch.logical_and(inside_center_bbox_mask,
                                                    inside_gt_bbox_mask)

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        angle_targets = gt_angle[range(num_points), min_area_inds]

        return labels, bbox_targets, angle_targets

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        angle_preds: List[Tensor],
                        score_factors: Optional[List[Tensor]] = None,
                        batch_data_samples: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        with_nms: bool = True) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.
        Note: When score_factors is not None, the cls_scores are
        usually multiplied by it then obtain the real score used in NMS,
        such as CenterNess in FCOS, IoU branch in ATSS.
        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            angle_preds (list[Tensor]): Box angle for each scale level
                with shape (N, num_points * encode_size, H, W)
            score_factors (list[Tensor], optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, num_priors * 1, H, W). Defaults to None.
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.
        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.
                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 5),
                  the last dimension 5 arrange as (x, y, w, h, t).
        """
        assert len(cls_scores) == len(bbox_preds)

        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)

        result_list = []

        for img_id in range(len(batch_data_samples[2])):
            data_sample = (batch_data_samples[0][img_id], batch_data_samples[1][img_id], batch_data_samples[2][img_id])
            cls_score_list = select_single_mlvl(
                cls_scores, img_id, detach=True)
            bbox_pred_list = select_single_mlvl(
                bbox_preds, img_id, detach=True)
            angle_pred_list = select_single_mlvl(
                angle_preds, img_id, detach=True)
            if with_score_factors:
                score_factor_list = select_single_mlvl(
                    score_factors, img_id, detach=True)
            else:
                score_factor_list = [None for _ in range(num_levels)]
            
            if self.training or self.pseudo_generator:
                predict_by_feat = self._predict_by_feat_single_pseudo
            else:
                predict_by_feat = self._predict_by_feat_single

            results = predict_by_feat(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                angle_pred_list=angle_pred_list,
                score_factor_list=score_factor_list,
                mlvl_priors=mlvl_priors,
                data_sample=data_sample,
                cfg=cfg,
                rescale=rescale,
                with_nms=with_nms)
            result_list.append(results)
        return result_list
    
    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        outs = self(x)
        predictions = self.predict_by_feat(
            *outs, batch_data_samples=unpack_gt_instances(batch_data_samples), rescale=rescale)
        return predictions
    
    def _predict_by_feat_single_pseudo(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                angle_pred_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                data_sample: dict,
                                cfg: ConfigDict,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.
        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            angle_pred_list (list[Tensor]): Box angle for a single scale
                level with shape (N, num_points * encode_size, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image, each item has shape
                (num_priors * 1, H, W).
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (mmengine.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.
        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.
                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 5),
                  the last dimension 5 arrange as (x, y, w, h, t).
        """        
        gt_instances = data_sample[0]
        img_meta = data_sample[2]
        img_shape = img_meta['img_shape']
        if self.training:
            scale_factor = [1, 1]
        else:
            scale_factor = img_meta['scale_factor']
        gt_bboxes = gt_instances.bboxes.tensor
        gt_cls = gt_labels = gt_instances.labels
        gt_pos = (gt_bboxes[:, 0:2] / self.strides[0] * scale_factor[1]).long()

        cls_score, bbox_pred, angle_pred = cls_score_list[0], bbox_pred_list[0], angle_pred_list[0]
        H, W = cls_score.shape[1:3]

        gt_valid_mask = (0 <= gt_pos[:, 0]) & (gt_pos[:, 0] < W) & (0 <= gt_pos[:, 1]) & (gt_pos[:, 1] < H)
        gt_idx = gt_pos[:, 1] * W + gt_pos[:, 0]
        gt_idx = gt_idx.clamp(0, cls_score[0].numel() - 1)
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)[gt_idx] * self.strides[0]
        cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)[gt_idx]
        priors = mlvl_priors[0][gt_idx]
        if self.use_hbox_output:
            bboxes = hbox2rbox(self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape))
        else:
            angle_pred = angle_pred.permute(1, 2, 0).reshape(
                    -1, self.angle_coder.encode_size)[gt_idx]
            decoded_angle = self.angle_coder.decode(angle_pred, keepdim=True)
            bbox_pred = torch.cat([bbox_pred, decoded_angle], dim=-1)
            bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)

        bboxes[:, 0:2] = gt_bboxes[:, 0:2]
        bboxes[~gt_valid_mask, 2:] = 0
        bboxes[:, 2:4] = bboxes[:, 2:4] / scale_factor[1]

        # When having ws label, we replace point labels and keep RBox/HBox as is
        if gt_labels.dim() == 2:
            gt_point = gt_labels[:, 1] == 2
            bboxes[~gt_point, 2:] = gt_bboxes[~gt_point, 2:]
            gt_labels[gt_point, 1] = 1
            gt_cls = gt_labels[:, 0]

        for id in self.post_process.keys():
            bboxes[gt_cls == id, 2:4] *= self.post_process[id]
        for id in self.square_cls:
            bboxes[gt_cls == id, -1] = 0

        results = InstanceData()
        results.bboxes = RotatedBoxes(bboxes.detach())
        results.scores = torch.ones_like(cls_score[:, 0])
        results.labels = gt_labels

        return results
    