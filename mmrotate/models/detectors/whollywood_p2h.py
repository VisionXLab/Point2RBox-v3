# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
from typing import Tuple, Union

import torch
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.models.utils import unpack_gt_instances
from mmdet.structures import DetDataSample, SampleList
from mmdet.structures.bbox import get_box_tensor
from mmdet.utils import ConfigType, InstanceList, OptConfigType, OptMultiConfig
from torch import Tensor
from torch.nn.functional import grid_sample
from torchvision import transforms

from mmengine.structures import InstanceData
from mmrotate.registry import MODELS
from mmrotate.structures.bbox import RotatedBoxes, rbox2hbox, hbox2rbox
from mmrotate.models.task_modules.synthesis_generators import \
    point2rbox_generator


@MODELS.register_module()
class WhollyWoodP2H(SingleStageDetector):

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 basic_pattern: str = 'basic_patterns/dota',
                 sca_fact: float = 1.0,
                 dense_cls: list = [],
                 square_cls: list = [],
                 use_synthesis: bool = True,
                 use_setrc: bool = True,
                 use_setsk: bool = True,
                 debug: bool = False,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        self.basic_pattern = basic_pattern
        self.sca_fact = sca_fact
        self.dense_cls = dense_cls
        self.square_cls = square_cls
        self.use_synthesis = use_synthesis
        self.debug = debug
        self.basic_pattern = point2rbox_generator.load_basic_pattern(
            self.basic_pattern, use_setrc, use_setsk)

    def add_synthesis(self, batch_inputs, batch_gt_instances):

        def synthesis_single(img, bboxes, labels):
            labels = labels[:, 0:1]
            bb = torch.cat((bboxes, torch.ones_like(labels), labels), -1)
            img, bb = point2rbox_generator.generate_sythesis(
                img, bb, self.sca_fact, *self.basic_pattern, self.dense_cls,
                img.shape[-1])
            ll = bb[:, 6].long()
            instance_data = InstanceData()
            instance_data.labels = torch.stack((ll, torch.zeros_like(ll)), -1)
            square_mask = torch.zeros_like(bb[:, 6], dtype=torch.bool)
            for c in self.square_cls:
                square_mask = torch.logical_or(square_mask, bb[:, 6].long() == c)
            bb[square_mask, 4] = 0
            if hasattr(self.bbox_head, 'use_hbox_output') and self.bbox_head.use_hbox_output:
                instance_data.bboxes = hbox2rbox(rbox2hbox(bb[:, :5]))
            else:
                instance_data.bboxes = bb[:, :5]
            return img, instance_data

        p = ((synthesis_single)(img, gt.bboxes.tensor.cpu(), gt.labels.cpu())
             for (img, gt) in zip(batch_inputs.cpu(), batch_gt_instances))

        img, instance_data = zip(*p)
        batch_inputs = torch.stack(img, 0).to(batch_inputs)
        instance_data = list(instance_data)
        for i, gt in enumerate(instance_data):
            gt.labels = gt.labels.to(batch_gt_instances[i].labels)
            gt.bboxes = RotatedBoxes(gt.bboxes.to(batch_gt_instances[i].bboxes.tensor))
            batch_gt_instances[i] = InstanceData.cat(
                [batch_gt_instances[i], gt])

        return batch_inputs, batch_gt_instances

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList):
        
        batch_gt_instances, _, batch_img_metas = unpack_gt_instances(batch_data_samples)

        # Generate synthetic objects
        if self.use_synthesis:
            batch_inputs, batch_gt_instances = self.add_synthesis(
                batch_inputs, batch_gt_instances)
            
        batch_data_samples = []
        for gt_instances, metas in zip(batch_gt_instances, batch_img_metas):
            data_sample = DetDataSample(
                metainfo=metas)
            data_sample.gt_instances = gt_instances
            batch_data_samples.append(data_sample)

        feat = self.extract_feat(batch_inputs)
        losses = self.bbox_head.loss(feat, batch_data_samples)
        results_list = self.bbox_head.predict(feat, batch_data_samples)
        
        if self.debug:
            import cv2
            import numpy as np
            idx = np.random.randint(100)
            B = batch_inputs.shape[0]
            batch_inputs_plot = batch_inputs[::B]
            for i in range(len(batch_inputs_plot)):
                img = batch_inputs_plot[i].permute(1, 2, 0).cpu().numpy()
                img = np.ascontiguousarray(img[..., (2, 1, 0)] * 58 + 127)
                bb = batch_data_samples[::B][i].gt_instances.bboxes.tensor
                for b in bb.cpu().numpy():
                    point2rbox_generator.plot_one_rotated_box(img, b)
                bb = results_list[::B][i].bboxes.tensor
                for b in bb.cpu().numpy():
                    point2rbox_generator.plot_one_rotated_box(img, b, (0, 255, 0))
                cv2.imwrite(f'debug/{idx}-{i}.png', img)

        return losses
