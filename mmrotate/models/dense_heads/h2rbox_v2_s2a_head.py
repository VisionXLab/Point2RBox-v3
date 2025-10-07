# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import math
import copy
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.models.utils import select_single_mlvl
from mmdet.structures.bbox import BaseBoxes, cat_boxes, get_box_tensor
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, OptMultiConfig)
from mmengine.structures import InstanceData
from mmengine.config import ConfigDict
from torch import Tensor
from mmdet.models.task_modules.prior_generators import anchor_inside_flags
from mmdet.models.utils import (filter_scores_and_topk, 
                                images_to_levels, multi_apply, unmap)

from mmrotate.registry import MODELS
from mmrotate.structures.bbox import RotatedBoxes
from ..utils import ORConv2d, RotationInvariantPooling
from .rotated_retina_head import RotatedRetinaHead


@MODELS.register_module()
class H2RBoxV2S2AHead(RotatedRetinaHead):
    r"""An anchor-based head used in `S2A-Net
    <https://ieeexplore.ieee.org/document/9377550>`_.
    """  # noqa: W605
    
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
    
    def _get_targets_single(self,
                            flat_anchors: Union[Tensor, BaseBoxes],
                            valid_flags: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict,
                            gt_instances_ignore: Optional[InstanceData] = None,
                            unmap_outputs: bool = True) -> tuple:
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            flat_anchors (Tensor or :obj:`BaseBoxes`): Multi-level anchors
                of the image, which are concatenated into a single tensor
                or box type of shape (num_anchors, 4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors, ).
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for current image.
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.  Defaults to True.

        Returns:
            tuple:

                - labels (Tensor): Labels of each level.
                - label_weights (Tensor): Label weights of each level.
                - bbox_targets (Tensor): BBox targets of each level.
                - bbox_weights (Tensor): BBox weights of each level.
                - pos_inds (Tensor): positive samples indexes.
                - neg_inds (Tensor): negative samples indexes.
                - sampling_result (:obj:`SamplingResult`): Sampling results.
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg['allowed_border'])
        if not inside_flags.any():
            raise ValueError(
                'There is no valid anchor inside the image boundary. Please '
                'check the image size and anchor sizes, or set '
                '``allowed_border`` to -1 to skip the condition.')
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags]

        pred_instances = InstanceData(priors=anchors)
        assign_result = self.assigner.assign(pred_instances, gt_instances,
                                             gt_instances_ignore)
        # No sampling is required except for RPN and
        # Guided Anchoring algorithms
        sampling_result = self.sampler.sample(assign_result, pred_instances,
                                              gt_instances)

        num_valid_anchors = anchors.shape[0]
        target_dim = gt_instances.bboxes.size(-1) if self.reg_decoded_bbox \
            else self.bbox_coder.encode_size
        bbox_targets = anchors.new_zeros(num_valid_anchors, target_dim)
        bbox_weights = anchors.new_zeros(num_valid_anchors, target_dim)

        # TODO: Considering saving memory, is it necessary to be long?
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        assigned_gt_inds = anchors.new_full((num_valid_anchors, ),
                                            -1,
                                            dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        # `bbox_coder.encode` accepts tensor or box type inputs and generates
        # tensor targets. If regressing decoded boxes, the code will convert
        # box type `pos_bbox_targets` to tensor.
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_priors, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
                pos_bbox_targets = get_box_tensor(pos_bbox_targets)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            labels[pos_inds] = sampling_result.pos_gt_labels
            assigned_gt_inds[pos_inds] = sampling_result.pos_assigned_gt_inds
            if self.train_cfg['pos_weight'] <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg['pos_weight']
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            assigned_gt_inds = unmap(assigned_gt_inds, num_total_anchors,
                                     inside_flags, fill=-1)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result, assigned_gt_inds)

    def loss_by_feat_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            anchors: Tensor, labels: Tensor,
                            label_weights: Tensor, bbox_targets: Tensor,
                            bbox_weights: Tensor, assigned_gt_inds: Tensor,
                            avg_factor: int) -> tuple:
        """Calculate the loss of a single scale level based on the features
        extracted by the detection head.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            avg_factor (int): Average factor that is used to average the loss.

        Returns:
            tuple: loss components.
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=avg_factor)
        # regression loss
        target_dim = bbox_targets.size(-1)
        bbox_targets = bbox_targets.reshape(-1, target_dim)
        bbox_weights = bbox_weights.reshape(-1, target_dim)
        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(-1,
                                                 self.bbox_coder.encode_size)
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, anchors.size(-1))
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
            bbox_pred = get_box_tensor(bbox_pred)

        square_mask = torch.zeros_like(labels, dtype=torch.bool)
        for c in self.square_cls:
            square_mask |= (labels == c)
        bbox_pred[square_mask, 4] = bbox_targets[square_mask, 4]
        # bbox_targets[square_mask, 4] = torch.where(
        #     torch.abs(bbox_targets[square_mask, 4]) < math.pi / 4,
        #     0,
        #     -math.pi / 2)

        loss_bbox = self.loss_bbox(
            *self.nested_projection(
                bbox_pred,
                bbox_targets),
            bbox_weights[:, 0],
            avg_factor=avg_factor)
        
        # Symmetry-aware learning
        labels = labels.view_as(assigned_gt_inds)
        angle_pred = bbox_pred[:, 4].view_as(assigned_gt_inds)
        bg_class_ind = self.num_classes
        pos_inds = (labels >= 0) & (labels < bg_class_ind)

        ins_angle_preds = []
        ins_labels = []
        for i, n_gt in enumerate(self.gt_info * 2):
            idx = assigned_gt_inds[i][pos_inds[i]]
            aa = angle_pred[i][pos_inds[i]]
            ll = labels[i][pos_inds[i]]
            ins_aa = aa.new_zeros(n_gt).index_reduce_(
                    0, idx, aa, 'amin', include_self=False)
            ins_ll = ll.new_full((n_gt,), -1).index_reduce_(
                    0, idx, ll, 'amin', include_self=False)
            ins_angle_preds.append(ins_aa)
            ins_labels.append(ins_ll)
        ins_angle_preds = torch.cat(ins_angle_preds, 0).view(2, sum(self.gt_info))
        ins_labels = torch.cat(ins_labels, 0).view(2, sum(self.gt_info))
        valid = (ins_labels[0] != -1) & (ins_labels[1] != -1)
        for c in self.square_cls:
            valid &= (ins_labels[0] != c)
        m, R = self.ss_info
        d_ang = ins_angle_preds[1, :] + m * ins_angle_preds[0, :] + R
        d_ang = (d_ang + math.pi / 2) % math.pi - math.pi / 2
        loss_ss = self.loss_ss(d_ang, 
                               torch.zeros_like(d_ang),
                               weight=valid)
        
        return loss_cls, loss_bbox, loss_ss

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                has shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         avg_factor, assigned_gt_inds_list) = cls_reg_targets

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(cat_boxes(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox, losses_ss = multi_apply(
            self.loss_by_feat_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            assigned_gt_inds_list,
            avg_factor=avg_factor)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_ss=losses_ss)
    
    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
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
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        if score_factor_list[0] is None:
            # e.g. Retina, FreeAnchor, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, etc.
            with_score_factors = True

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_scores = []
        mlvl_labels = []
        if with_score_factors:
            mlvl_score_factors = []
        else:
            mlvl_score_factors = None
        for level_idx, (cls_score, bbox_pred, score_factor, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list,
                              score_factor_list, mlvl_priors)):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            dim = self.bbox_coder.encode_size
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, dim)
            if with_score_factors:
                score_factor = score_factor.permute(1, 2,
                                                    0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)

            # the `custom_cls_channels` parameter is derived from
            # CrossEntropyCustomLoss and FocalCustomLoss, and is currently used
            # in v3det.
            if getattr(self.loss_cls, 'custom_cls_channels', False):
                scores = self.loss_cls.get_activation(cls_score)
            elif self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                scores = cls_score.softmax(-1)[:, :-1]

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            score_thr = cfg.get('score_thr', 0)

            results = filter_scores_and_topk(
                scores, score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, keep_idxs, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']

            if with_score_factors:
                score_factor = score_factor[keep_idxs]

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

            if with_score_factors:
                mlvl_score_factors.append(score_factor)

        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = cat_boxes(mlvl_valid_priors)
        bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)
        labels = torch.cat(mlvl_labels)

        bboxes = bboxes.tensor
        for id in self.square_cls:
            bboxes[labels == id, -1] = 0
        for id in self.square_resize_cls:
            bboxes[labels == id, 2:4] *= 0.85
        bboxes = RotatedBoxes(bboxes)

        results = InstanceData()
        results.bboxes = bboxes
        results.scores = torch.cat(mlvl_scores)
        results.labels = labels
        if with_score_factors:
            results.score_factors = torch.cat(mlvl_score_factors)

        return self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)
    
    def filter_bboxes(self, cls_scores: List[Tensor],
                      bbox_preds: List[Tensor]) -> List[List[Tensor]]:
        """This function will be used in S2ANet, whose num_anchors=1.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, 5, H, W)

        Returns:
            list[list[Tensor]]: refined rbboxes of each level of each image.
        """
        num_levels = len(cls_scores)
        assert num_levels == len(bbox_preds)
        num_imgs = cls_scores[0].size(0)
        for i in range(num_levels):
            assert num_imgs == cls_scores[i].size(0) == bbox_preds[i].size(0)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_priors(
            featmap_sizes, device=device)

        bboxes_list = [[] for _ in range(num_imgs)]

        for lvl in range(num_levels):
            bbox_pred = bbox_preds[lvl]
            bbox_pred = bbox_pred.permute(0, 2, 3, 1)
            bbox_pred = bbox_pred.reshape(num_imgs, -1, bbox_pred.shape[-1])
            anchors = mlvl_anchors[lvl]

            for img_id in range(num_imgs):
                bbox_pred_i = bbox_pred[img_id]
                decode_bbox_i = self.bbox_coder.decode(anchors, bbox_pred_i)
                bboxes_list[img_id].append(decode_bbox_i.detach())

        return bboxes_list


@MODELS.register_module()
class H2RBoxV2S2ARefineHead(H2RBoxV2S2AHead):
    r"""Rotated Anchor-based refine head. It's a part of the Oriented Detection
    Module (ODM), which produces orientation-sensitive features for
    classification and orientation-invariant features for localization.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        frm_cfg (dict): Config of the feature refine module.
    """  # noqa: W605

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 frm_cfg: dict = None,
                 **kwargs) -> None:
        super().__init__(
            num_classes=num_classes, in_channels=in_channels, **kwargs)
        self.feat_refine_module = MODELS.build(frm_cfg)
        self.bboxes_as_anchors = None

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.or_conv = ORConv2d(
            self.in_channels,
            int(self.feat_channels / 8),
            kernel_size=3,
            padding=1,
            arf_config=(1, 8))
        self.or_pool = RotationInvariantPooling(self.feat_channels, 8)
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = int(self.feat_channels / 8) if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    self.feat_channels,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * self.cls_out_channels,
            3,
            padding=1)
        reg_dim = self.bbox_coder.encode_size
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_base_priors * reg_dim, 3, padding=1)

    def forward_single(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:

            - cls_score (Tensor): Cls scores for a single scale level
              the channels number is num_anchors * num_classes.
            - bbox_pred (Tensor): Box energies / deltas for a single scale
              level, the channels number is num_anchors * 4.
        """
        x = self.or_conv(x)
        reg_feat = x
        cls_feat = self.or_pool(x)
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred

    def loss_by_feat(self,
                     cls_scores: List[Tensor],
                     bbox_preds: List[Tensor],
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None,
                     rois: List[Tensor] = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                has shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            rois (list[Tensor])

        Returns:
            dict: A dictionary of loss components.
        """
        assert rois is not None
        self.bboxes_as_anchors = rois
        return super().loss_by_feat(
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)

    def get_anchors(self,
                    featmap_sizes: List[tuple],
                    batch_img_metas: List[dict],
                    device: Union[torch.device, str] = 'cuda') \
            -> Tuple[List[List[Tensor]], List[List[Tensor]]]:
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            batch_img_metas (list[dict]): Image meta info.
            device (torch.device | str): Device for returned tensors.
                Defaults to cuda.

        Returns:
            tuple:

            - anchor_list (list[list[Tensor]]): Anchors of each image.
            - valid_flag_list (list[list[Tensor]]): Valid flags of each
              image.
        """
        anchor_list = [[
            RotatedBoxes(bboxes_img_lvl).detach()
            for bboxes_img_lvl in bboxes_img
        ] for bboxes_img in self.bboxes_as_anchors]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(batch_img_metas):
            multi_level_flags = self.prior_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        score_factors: Optional[List[Tensor]] = None,
                        rois: List[Tensor] = None,
                        batch_img_metas: Optional[List[dict]] = None,
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
            score_factors (list[Tensor], optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, num_priors * 1, H, W). Defaults to None.
            rois (list[Tensor]):
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
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_scores) == len(bbox_preds)
        assert rois is not None

        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        result_list = []

        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            cls_score_list = select_single_mlvl(
                cls_scores, img_id, detach=True)
            bbox_pred_list = select_single_mlvl(
                bbox_preds, img_id, detach=True)
            if with_score_factors:
                score_factor_list = select_single_mlvl(
                    score_factors, img_id, detach=True)
            else:
                score_factor_list = [None for _ in range(num_levels)]

            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                score_factor_list=score_factor_list,
                mlvl_priors=rois[img_id],
                img_meta=img_meta,
                cfg=cfg,
                rescale=rescale,
                with_nms=with_nms)
            result_list.append(results)
        return result_list

    def feature_refine(self, x: List[Tensor],
                       rois: List[List[Tensor]]) -> List[Tensor]:
        """Refine the input feature use feature refine module.

        Args:
            x (list[Tensor]): feature maps of multiple scales.
            rois (list[list[Tensor]]): input rbboxes of multiple
                scales of multiple images, output by former stages
                and are to be refined.

        Returns:
            list[Tensor]: refined feature maps of multiple scales.
        """
        return self.feat_refine_module(x, rois)

    def refine_bboxes(self, cls_scores: List[Tensor], bbox_preds: List[Tensor],
                      rois: List[List[Tensor]]) -> List[List[Tensor]]:
        """Refine predicted bounding boxes at each position of the feature
        maps. This method will be used in R3Det in refinement stages.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, 5, H, W)
            rois (list[list[Tensor]]): input rbboxes of each level of each
                image. rois output by former stages and are to be refined

        Returns:
            list[list[Tensor]]: best or refined rbboxes of each level of each
            image.
        """
        num_levels = len(cls_scores)
        assert num_levels == len(bbox_preds)

        num_imgs = cls_scores[0].size(0)

        for i in range(num_levels):
            assert num_imgs == cls_scores[i].size(0) == bbox_preds[i].size(0)

        bboxes_list = [[] for _ in range(num_imgs)]

        assert rois is not None
        mlvl_rois = [torch.cat(r) for r in zip(*rois)]

        for lvl in range(num_levels):
            bbox_pred = bbox_preds[lvl]
            rois = mlvl_rois[lvl]
            assert bbox_pred.size(1) == 7
            bbox_pred = bbox_pred.permute(0, 2, 3, 1)
            bbox_pred = bbox_pred.reshape(-1, bbox_pred.shape[-1])
            refined_bbox = self.bbox_coder.decode(rois, bbox_pred)
            refined_bbox = refined_bbox.reshape(num_imgs, -1, 5)
            for img_id in range(num_imgs):
                bboxes_list[img_id].append(refined_bbox[img_id].detach())
        return bboxes_list
