# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional
import math
import torch
import torch.nn.functional as F
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from mmdet.models.losses import accuracy
from mmdet.models.layers import multiclass_nms
from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.models.utils import empty_instances, multi_apply
from mmdet.models.roi_heads.bbox_heads import Shared2FCBBoxHead
from mmdet.structures.bbox import get_box_tensor, scale_boxes
from mmdet.utils import (ConfigType, OptConfigType)
from torch import Tensor

from mmrotate.registry import MODELS


@MODELS.register_module()
class H2RBoxV2Shared2FCBBoxHead(Shared2FCBBoxHead):
    """H2RBox-v2 Shared2FC BBox head.

    Args:
        loss_bbox_type (str): Set the input type of ``loss_bbox``.
            Defaults to 'normal'.
    """

    def __init__(self,
                 *args,
                 square_cls: list = [],
                 square_resize_cls: list = [],
                 loss_ss: ConfigType = dict(
                     type='mmdet.SmoothL1Loss', loss_weight=0.2, beta=0.1),
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.square_cls = square_cls
        self.square_resize_cls = square_resize_cls
        self.loss_ss = MODELS.build(loss_ss)

    def nested_projection(self, pred, target):
        target_xy1 = target[..., 0:2] - target[..., 2:4] / 2
        target_xy2 = target[..., 0:2] + target[..., 2:4] / 2
        target_projected = torch.cat((target_xy1, target_xy2), -1)
        pred_xy = pred[..., 0:2]
        pred_wh = pred[..., 2:4]
        da = (pred[..., 4] - target[..., 4]).detach()
        cosa = torch.cos(da).abs()
        sina = torch.sin(da).abs()
        pred_wh = torch.matmul(
            torch.stack((cosa, sina, sina, cosa), -1).view(*cosa.shape, 2, 2),
            pred_wh[..., None])[..., 0]
        pred_xy1 = pred_xy - pred_wh / 2
        pred_xy2 = pred_xy + pred_wh / 2
        pred_projected = torch.cat((pred_xy1, pred_xy2), -1)
        return pred_projected, target_projected
    
    def _get_targets_single(self, pos_priors: Tensor, neg_priors: Tensor,
                            pos_gt_bboxes: Tensor, pos_gt_labels: Tensor,
                            pos_assigned_gt_inds: Tensor,
                            cfg: ConfigDict) -> tuple:
        """Calculate the ground truth for proposals in the single image
        according to the sampling results.

        Args:
            pos_priors (Tensor): Contains all the positive boxes,
                has shape (num_pos, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            neg_priors (Tensor): Contains all the negative boxes,
                has shape (num_neg, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_bboxes (Tensor): Contains gt_boxes for
                all positive samples, has shape (num_pos, 4),
                the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_labels (Tensor): Contains gt_labels for
                all positive samples, has shape (num_pos, ).
            cfg (obj:`ConfigDict`): `train_cfg` of R-CNN.

        Returns:
            Tuple[Tensor]: Ground truth for proposals
            in a single image. Containing the following Tensors:

                - labels(Tensor): Gt_labels for all proposals, has
                  shape (num_proposals,).
                - label_weights(Tensor): Labels_weights for all
                  proposals, has shape (num_proposals,).
                - bbox_targets(Tensor):Regression target for all
                  proposals, has shape (num_proposals, 4), the
                  last dimension 4 represents [tl_x, tl_y, br_x, br_y].
                - bbox_weights(Tensor):Regression weights for all
                  proposals, has shape (num_proposals, 4).
        """
        num_pos = pos_priors.size(0)
        num_neg = neg_priors.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_priors.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        assigned_gt_inds = pos_priors.new_zeros(num_samples, 
                                                dtype=torch.long)
        reg_dim = pos_gt_bboxes.size(-1) if self.reg_decoded_bbox \
            else self.bbox_coder.encode_size
        label_weights = pos_priors.new_zeros(num_samples)
        bbox_targets = pos_priors.new_zeros(num_samples, reg_dim)
        bbox_weights = pos_priors.new_zeros(num_samples, reg_dim)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            assigned_gt_inds[:num_pos] = pos_assigned_gt_inds
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_priors, pos_gt_bboxes)
            else:
                # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                # is applied directly on the decoded bounding boxes, both
                # the predicted boxes and regression targets should be with
                # absolute coordinate format.
                pos_bbox_targets = get_box_tensor(pos_gt_bboxes)
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights, assigned_gt_inds

    def get_targets(self,
                    sampling_results: List[SamplingResult],
                    rcnn_train_cfg: ConfigDict,
                    concat: bool = True) -> tuple:
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.

        Almost the same as the implementation in bbox_head, we passed
        additional parameters pos_inds_list and neg_inds_list to
        `_get_targets_single` function.

        Args:
            sampling_results (List[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following list of Tensors:

            - labels (list[Tensor],Tensor): Gt_labels for all
                proposals in a batch, each tensor in list has
                shape (num_proposals,) when `concat=False`, otherwise
                just a single tensor has shape (num_all_proposals,).
            - label_weights (list[Tensor]): Labels_weights for
                all proposals in a batch, each tensor in list has
                shape (num_proposals,) when `concat=False`, otherwise
                just a single tensor has shape (num_all_proposals,).
            - bbox_targets (list[Tensor],Tensor): Regression target
                for all proposals in a batch, each tensor in list
                has shape (num_proposals, 4) when `concat=False`,
                otherwise just a single tensor has shape
                (num_all_proposals, 4), the last dimension 4 represents
                [tl_x, tl_y, br_x, br_y].
            - bbox_weights (list[tensor],Tensor): Regression weights for
                all proposals in a batch, each tensor in list has shape
                (num_proposals, 4) when `concat=False`, otherwise just a
                single tensor has shape (num_all_proposals, 4).
        """
        pos_priors_list = [res.pos_priors for res in sampling_results]
        neg_priors_list = [res.neg_priors for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        pos_assigned_gt_inds_list = [res.pos_assigned_gt_inds for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights, assigned_gt_inds = multi_apply(
            self._get_targets_single,
            pos_priors_list,
            neg_priors_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            pos_assigned_gt_inds_list,
            cfg=rcnn_train_cfg)

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
            assigned_gt_inds = torch.cat(assigned_gt_inds, 0)
        return labels, label_weights, bbox_targets, bbox_weights, assigned_gt_inds

    def loss(self,
             cls_score: Tensor,
             bbox_pred: Tensor,
             rois: Tensor,
             labels: Tensor,
             label_weights: Tensor,
             bbox_targets: Tensor,
             bbox_weights: Tensor,
             assigned_gt_inds: Tensor,
             reduction_override: Optional[str] = None) -> dict:
        """Calculate the loss based on the network predictions and targets.

        Args:
            cls_score (Tensor): Classification prediction
                results of all class, has shape
                (batch_size * num_proposals_single_image, num_classes)
            bbox_pred (Tensor): Regression prediction results,
                has shape
                (batch_size * num_proposals_single_image, 4), the last
                dimension 4 represents [tl_x, tl_y, br_x, br_y].
            rois (Tensor): RoIs with the shape
                (batch_size * num_proposals_single_image, 5) where the first
                column indicates batch id of each RoI.
            labels (Tensor): Gt_labels for all proposals in a batch, has
                shape (batch_size * num_proposals_single_image, ).
            label_weights (Tensor): Labels_weights for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, ).
            bbox_targets (Tensor): Regression target for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, 4),
                the last dimension 4 represents [tl_x, tl_y, br_x, br_y].
            bbox_weights (Tensor): Regression weights for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, 4).
            reduction_override (str, optional): The reduction
                method used to override the original reduction
                method of the loss. Options are "none",
                "mean" and "sum". Defaults to None,

        Returns:
            dict: A dictionary of loss.
        """

        losses = dict()

        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                    bbox_pred = get_box_tensor(bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), self.num_classes,
                        -1)[pos_inds.type(torch.bool),
                            labels[pos_inds.type(torch.bool)]]
                
                pos_labels = labels[pos_inds.type(torch.bool)]
                pos_bbox_targets = bbox_targets[pos_inds.type(torch.bool)]
                square_mask = torch.zeros_like(pos_labels, dtype=torch.bool)
                for c in self.square_cls:
                    square_mask |= (pos_labels == c)
                pos_bbox_pred[square_mask, 4] = pos_bbox_targets[square_mask, 4]

                losses['loss_bbox'] = self.loss_bbox(
                    *self.nested_projection(
                        pos_bbox_pred,
                        pos_bbox_targets),
                    bbox_weights[pos_inds.type(torch.bool), 0],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
                
                # Symmetry-aware learning
                pos_rois = rois[pos_inds.type(torch.bool)]
                pos_assigned_gt_inds = assigned_gt_inds[pos_inds.type(torch.bool)]
                pos_labels = labels[pos_inds.type(torch.bool)]
                ins_bbox_preds = []
                ins_labels = []
                for i, n_gt in enumerate(self.gt_info * 2):
                    idx = pos_assigned_gt_inds[pos_rois[:, 0] == i]
                    bb = pos_bbox_pred[pos_rois[:, 0] == i]
                    ll = pos_labels[pos_rois[:, 0] == i]
                    ins_bb = bb.new_zeros(n_gt, 5).index_reduce_(
                            0, idx, bb, 'amin', include_self=False)
                    ins_ll = ll.new_full((n_gt,), -1).index_reduce_(
                            0, idx, ll, 'amin', include_self=False)
                    ins_bbox_preds.append(ins_bb)
                    ins_labels.append(ins_ll)
                ins_bbox_preds = torch.cat(ins_bbox_preds, 0).view(2, sum(self.gt_info), 5)
                ins_labels = torch.cat(ins_labels, 0).view(2, sum(self.gt_info))
                valid = (ins_labels[0] != -1) & (ins_labels[1] != -1)
                for c in self.square_cls:
                    valid &= (ins_labels[0] != c)
                m, R = self.ss_info
                d_ang = ins_bbox_preds[1, :, 4] + m * ins_bbox_preds[0, :, 4] + R
                d_ang = (d_ang + math.pi / 2) % math.pi - math.pi / 2
                losses['loss_ss'] = self.loss_ss(d_ang, 
                                                 torch.zeros_like(d_ang),
                                                 weight=valid)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()

        return losses
    
    def _predict_by_feat_single(
            self,
            roi: Tensor,
            cls_score: Tensor,
            bbox_pred: Tensor,
            img_meta: dict,
            rescale: bool = False,
            rcnn_test_cfg: Optional[ConfigDict] = None) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            roi (Tensor): Boxes to be transformed. Has shape (num_boxes, 5).
                last dimension 5 arrange as (batch_index, x1, y1, x2, y2).
            cls_score (Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor): Box energies / deltas.
                has shape (num_boxes, num_classes * 4).
            img_meta (dict): image information.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head.
                Defaults to None

        Returns:
            :obj:`InstanceData`: Detection results of each image\
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        results = InstanceData()
        if roi.shape[0] == 0:
            return empty_instances([img_meta],
                                   roi.device,
                                   task_type='bbox',
                                   instance_results=[results],
                                   box_type=self.predict_box_type,
                                   use_box_type=False,
                                   num_classes=self.num_classes,
                                   score_per_cls=rcnn_test_cfg is None)[0]

        # some loss (Seesaw loss..) may have custom activation
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(
                cls_score, dim=-1) if cls_score is not None else None

        img_shape = img_meta['img_shape']
        num_rois = roi.size(0)
        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            num_classes = 1 if self.reg_class_agnostic else self.num_classes
            roi = roi.repeat_interleave(num_classes, dim=0)
            bbox_pred = bbox_pred.view(-1, self.bbox_coder.encode_size)
            bboxes = self.bbox_coder.decode(
                roi[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = roi[:, 1:].clone()
            if img_shape is not None and bboxes.size(-1) == 4:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            bboxes = scale_boxes(bboxes, scale_factor)

        # Get the inside tensor when `bboxes` is a box type
        bboxes = get_box_tensor(bboxes)
        box_dim = bboxes.size(-1)
        bboxes = bboxes.view(num_rois, -1)
        labels = scores.argmax(-1).view(num_rois)

        for id in self.square_cls:
            bboxes[labels == id, -1] = 0
        for id in self.square_resize_cls:
            bboxes[labels == id, 2:4] *= 0.85

        if rcnn_test_cfg is None:
            # This means that it is aug test.
            # It needs to return the raw results without nms.
            results.bboxes = bboxes
            results.scores = scores
        else:
            det_bboxes, det_labels = multiclass_nms(
                bboxes,
                scores,
                rcnn_test_cfg.score_thr,
                rcnn_test_cfg.nms,
                rcnn_test_cfg.max_per_img,
                box_dim=box_dim)
            results.bboxes = det_bboxes[:, :-1]
            results.scores = det_bboxes[:, -1]
            results.labels = det_labels
        return results
