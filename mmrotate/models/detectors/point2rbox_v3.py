# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
import cv2
import numpy as np
from typing import Tuple, Union, List

import torch
from collections import Counter
from torch import Tensor
import torch.nn.functional as F
from torch.nn.functional import grid_sample
from torchvision import transforms

from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.models.utils import unpack_gt_instances
from mmdet.structures import DetDataSample, SampleList
from mmdet.structures.bbox import get_box_tensor
from mmdet.utils import ConfigType, InstanceList, OptConfigType, OptMultiConfig
from mmengine.structures import InstanceData
from mmrotate.registry import MODELS
from mmrotate.structures.bbox import RotatedBoxes, rbox2hbox, hbox2rbox

from third_parties.ted.ted import TED

from PIL import Image

def save_tensor_as_images(tensor, output_prefix="image"):
    tensor = tensor.detach().cpu().numpy()
    tensor = np.transpose(tensor, (0, 2, 3, 1))

    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min()) * 255
    tensor = tensor.astype(np.uint8)

    for i in range(tensor.shape[0]):
        img = Image.fromarray(tensor[i])
        img.save(f"./vis/{output_prefix}_{i}.png")

def gaussian_2d(xy, mu, sigma, normalize=False):
    dxy = (xy - mu).unsqueeze(-1)
    t0 = torch.exp(-0.5 * dxy.permute(0, 2, 1).bmm(torch.linalg.solve(sigma, dxy)))
    if normalize:
        t0 = t0 / (2 * np.pi * sigma.det().clamp(1e-7).sqrt())
    return t0

def get_single_pattern(image, bbox, label, square_cls):
    if bbox[2] < 16 or bbox[3] < 16 or bbox[2] > 512 or bbox[3] > 512:
        raise

    def obb2poly(obb):
        cx, cy, w, h, t = obb
        dw, dh = (w - 1) / 2, (h - 1) / 2
        cost = np.cos(t)
        sint = np.sin(t)
        mrot = np.float32([[cost, -sint], [sint, cost]])
        poly = np.float32([[-dw, -dh], [dw, -dh], [dw, dh], [-dw, dh]])
        return np.matmul(poly, mrot.T) + np.float32([cx, cy])

    def get_pattern_gaussian(w, h, device):
        w, h = int(w), int(h)
        y, x = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing='ij')
        y = (y - h / 2) / (h / 2)
        x = (x - w / 2) / (w / 2)
        ox, oy = torch.randn(2, device=device).clip(-3, 3) * 0.15
        sx, sy = torch.rand(2, device=device) * 0.5 + 1
        z = torch.exp(-((x - ox) * sx)**2 - ((y - oy) * sy)**2) * 0.5 + 0.5
        return z

    cx, cy, w, h, t = bbox
    w, h = int(w), int(h)
    poly = obb2poly([cx, cy, w, h, t])

    pts1 = poly[0:3]
    pts2 = np.float32([[-1, -1], [1, -1], [1, 1]])
    M = cv2.getAffineTransform(pts1, pts2)
    M = np.concatenate((M, ((0, 0, 1),)), 0)

    H, W = image.shape[1:3]
    T = np.array([[2 / W, 0, -1],
                  [0, 2 / H, -1],
                  [0, 0, 1]])
    theta = T @ np.linalg.inv(M)
    theta = image.new_tensor(theta[:2, :])[None]
    grid = F.affine_grid(theta, [1, 3, h, w], align_corners=True)
    chip = F.grid_sample(image[None], grid, align_corners=True)[0]

    alpha = get_pattern_gaussian(chip.shape[-1], chip.shape[-2], chip.device)[None]
    chip = torch.cat((chip, alpha))
        
    w, h, t = chip.new_tensor((bbox[2] * (0.7 + 0.5 * np.random.rand()), bbox[3] * (0.7 + 0.5 * np.random.rand()), np.pi * np.random.rand()))
    if label in square_cls:
        t *= 0
    cosa = torch.abs(torch.cos(t))
    sina = torch.abs(torch.sin(t))
    sx, sy = int(torch.ceil(cosa * w + sina * h)), int(torch.ceil(sina * w + cosa * h))
    theta = chip.new_tensor(
        [[1 / w * torch.cos(t), 1 / w * torch.sin(t), 0],
        [1 / h * torch.sin(-t), 1 / h * torch.cos(t), 0]])
    theta[:, :2] @= chip.new_tensor([[sx, 0], [0, sy]])
    grid = torch.nn.functional.affine_grid(
        theta[None], (1, 1, sy, sx), align_corners=True)
    chip = torch.nn.functional.grid_sample(
        chip[None], grid, align_corners=True, mode='nearest')[0]
    bbox = np.float32([sx / 2, sy / 2, w.item(), h.item(), t.item()])
    return (chip, bbox, label)


def get_copy_paste_cache(images, bboxes, labels, square_cls, num_copies):
    bboxes = bboxes.cpu().numpy()
    labels = labels.cpu().numpy()
    patterns = []
    for b, l in zip(bboxes, labels):
        try:
            p = get_single_pattern(images, b, l, square_cls)
            patterns.append(p)
            if len(patterns) > num_copies:
                break
        except:
            pass
    return patterns


@MODELS.register_module()
class Point2RBoxV3(SingleStageDetector):

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 rotate_range: Tuple[float, float] = (0.25, 0.75),
                 scale_range: Tuple[float, float] = (0.5, 0.9),
                 ss_prob: float = [0.6, 0.15, 0.25],
                 copy_paste_start_epoch: int = 6,
                 label_assign_pseudo_label_switch_eopch: int = 6,
                 num_copies: int = 10,
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

        self.rotate_range = rotate_range
        self.scale_range = scale_range
        self.ss_prob = ss_prob
        self.copy_paste_start_epoch = copy_paste_start_epoch
        self.label_assign_pseudo_label_switch_eopch = label_assign_pseudo_label_switch_eopch
        self.num_copies = num_copies
        self.debug = debug
        self.copy_paste_cache = None

        self.ted_model = TED()
        for param in self.ted_model.parameters():
            param.requires_grad = False
        self.ted_model.load_state_dict(torch.load('third_parties/ted/ted.pth'))
        self.ted_model.eval()

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.bbox_head.epoch = epoch

    def rotate_crop(
            self,
            batch_inputs: Tensor,
            rot: float = 0.,
            size: Tuple[int, int] = (768, 768),
            batch_gt_instances: InstanceList = None,
            padding: str = 'reflection') -> Tuple[Tensor, InstanceList]:
        """

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            rot (float): Angle of view rotation. Defaults to 0.
            size (tuple[int]): Crop size from image center.
                Defaults to (768, 768).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            padding (str): Padding method of image black edge.
                Defaults to 'reflection'.

        Returns:
            Processed batch_inputs (Tensor) and batch_gt_instances
            (list[:obj:`InstanceData`])
        """
        device = batch_inputs.device
        n, c, h, w = batch_inputs.shape
        size_h, size_w = size
        crop_h = (h - size_h) // 2
        crop_w = (w - size_w) // 2
        if rot != 0:
            cosa, sina = math.cos(rot), math.sin(rot)
            tf = batch_inputs.new_tensor([[cosa, -sina], [sina, cosa]],
                                         dtype=torch.float)
            x_range = torch.linspace(-1, 1, w, device=device)
            y_range = torch.linspace(-1, 1, h, device=device)
            y, x = torch.meshgrid(y_range, x_range)
            grid = torch.stack([x, y], -1).expand([n, -1, -1, -1])
            grid = grid.reshape(-1, 2).matmul(tf).view(n, h, w, 2)
            # rotate
            batch_inputs = grid_sample(
                batch_inputs, grid, 'bilinear', padding, align_corners=True)
            if batch_gt_instances is not None:
                for i, gt_instances in enumerate(batch_gt_instances):
                    gt_bboxes = get_box_tensor(gt_instances.bboxes)
                    xy, wh, a = gt_bboxes[..., :2], gt_bboxes[
                        ..., 2:4], gt_bboxes[..., [4]]
                    ctr = tf.new_tensor([[w / 2, h / 2]])
                    xy = (xy - ctr).matmul(tf.T) + ctr
                    a = a + rot
                    rot_gt_bboxes = torch.cat([xy, wh, a], dim=-1)
                    batch_gt_instances[i].bboxes = RotatedBoxes(rot_gt_bboxes)
        batch_inputs = batch_inputs[..., crop_h:crop_h + size_h,
                                    crop_w:crop_w + size_w]
        if batch_gt_instances is None:
            return batch_inputs
        else:
            for i, gt_instances in enumerate(batch_gt_instances):
                gt_bboxes = get_box_tensor(gt_instances.bboxes)
                xy, wh, a = gt_bboxes[..., :2], gt_bboxes[...,
                                                          2:4], gt_bboxes[...,
                                                                          [4]]
                xy = xy - xy.new_tensor([[crop_w, crop_h]])
                crop_gt_bboxes = torch.cat([xy, wh, a], dim=-1)
                batch_gt_instances[i].bboxes = RotatedBoxes(crop_gt_bboxes)

            return batch_inputs, batch_gt_instances
    
    def prepare_dual_stream_inputs(self,
        single_stream_inputs: Tensor,
        single_stream_targets: InstanceList,
        single_stream_metas: List[dict]
    ) -> Tuple[Tensor, InstanceList, List[dict]]:
        """Prepare dual stream inputs.
        
        Args:
            single_stream_inputs: Input images of shape (N, C, H, W).
            single_stream_targets: It usually includes bboxes
                and labels attributes.
            single_stream_metas: Meta information of each image, e.g.,
                image size, scaling factor, etc.
        
        Returns:
            dual_stream_inputs, dual_stream_targets.
        """
        H, W = single_stream_inputs.shape[2:4]
        sel_p = torch.rand(1)
        if sel_p < self.ss_prob[0]:  # rotate
            # inputs & targets
            rot = math.pi * (
                torch.rand(1).item() *
                (self.rotate_range[1] - self.rotate_range[0]) + self.rotate_range[0])
            batch_gt_aug = copy.deepcopy(single_stream_targets)
            batch_inputs_aug, batch_gt_aug = self.rotate_crop(
                single_stream_inputs, rot, [H, W], batch_gt_aug, 'reflection')
            for gt_instances in batch_gt_aug:
                gt_instances.bids[:, 0] += len(single_stream_targets)
                gt_instances.bids[:, 2] = 1
            # metas
            for img_metas in single_stream_metas:
                img_metas['ss'] = ('rot', rot)
        elif sel_p < self.ss_prob[0] + self.ss_prob[1]:  # flip 
            # inputs
            batch_inputs_aug = transforms.functional.vflip(single_stream_inputs)
            # targets
            batch_gt_aug = copy.deepcopy(single_stream_targets)
            for gt_instances in batch_gt_aug:
                gt_instances.bboxes.flip_([H, W], 'vertical')
                gt_instances.bids[:, 0] += len(single_stream_targets)
                gt_instances.bids[:, 2] = 1
            # metas
            for img_metas in single_stream_metas:
                img_metas['ss'] = ('flp', 0)
        else:  # scale
            # inputs
            sca = (torch.rand(1).item() *
                (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0])
            batch_inputs_aug = transforms.functional.resized_crop(single_stream_inputs,
                0, 0, int(H / sca), int(W / sca), [H, W])
            # targets
            batch_gt_aug = copy.deepcopy(single_stream_targets)
            for gt_instances in batch_gt_aug:
                gt_instances.bboxes.rescale_([sca, sca])
                gt_instances.bids[:, 0] += len(single_stream_targets)
                gt_instances.bids[:, 2] = 1
            # metas
            for img_metas in single_stream_metas:
                img_metas['ss'] = ('sca', sca)     
        dual_stream_inputs = torch.cat((single_stream_inputs, batch_inputs_aug))
        dual_stream_targets = single_stream_targets + batch_gt_aug
        return dual_stream_inputs, dual_stream_targets
    
    def prepare_edges(self):
        """Prepare edges for edge loss"""
        with torch.no_grad():
            mean = self.data_preprocessor.mean
            std = self.data_preprocessor.std
            batch_edges = self.ted_model(self.bbox_head.images_no_copypaste * std + mean)
            self.bbox_head.edges = batch_edges[3].clamp(0)
            # cv2.imwrite('E.png', self.bbox_head.edges[0, 0].cpu().numpy() * 255)
    
    def prepare_copy_paste_step1(self):
        raise NotImplemented
    
    def prepare_copy_paste_step2(self, dual_stream_inputs, dual_stream_targets):
        B, _, H, W = dual_stream_inputs.shape
        aug_begin_id = int(B / 2)
        aug_samples_len = int(B / 2)
        
        for i in range(aug_samples_len):
            gt_instances = dual_stream_targets[aug_begin_id + i]
            patterns = self.copy_paste_cache[i]
            
            bboxes_paste = []
            labels_paste = []
            for p, b, l in patterns:
                h, w = p.shape[1:3]
                ox = np.random.randint(0, W - w)
                oy = np.random.randint(0, H - h)
                dual_stream_inputs[aug_begin_id + i, :, oy:oy + h, ox:ox + w] = \
                    dual_stream_inputs[aug_begin_id + i, :, oy:oy + h, ox:ox + w] \
                    * (1 - p[(3,)]) + p[:3] * p[(3,)]
                bboxes_paste.append(b + np.float32((ox, oy, 0, 0, 0)))
                labels_paste.append(l)
            bboxes = torch.cat((gt_instances.bboxes.tensor, 
                                gt_instances.bboxes.tensor.new_tensor(np.float32(bboxes_paste))))
            labels = torch.cat((gt_instances.labels, 
                                gt_instances.labels.new_tensor(np.int32(labels_paste))))
            bids = torch.cat((gt_instances.bids, 
                              gt_instances.bids.new_tensor((i, 1, 0, 0)).expand(len(labels_paste), -1)))
            gt_instances = InstanceData()
            gt_instances.bboxes = RotatedBoxes(bboxes)
            gt_instances.labels = labels
            gt_instances.bids = bids
            dual_stream_targets[aug_begin_id + i] = gt_instances
        
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        batch_gt_instances, _, batch_img_metas = unpack_gt_instances(batch_data_samples)

        # Set bids for original images and gts
        # bids: long (N, 4) (batch_id based 0, syn, view, obj_id based 1)
        offset = 1
        for i, gt_instances in enumerate(batch_gt_instances):
            blen = len(gt_instances.bboxes)
            bids = gt_instances.labels.new_zeros(blen, 4)
            bids[:, 0] = i
            bids[:, 3] = torch.arange(0, blen, 1) + offset
            gt_instances.bids = bids
            offset += blen

        dual_stream_inputs, dual_stream_targets = self.prepare_dual_stream_inputs(
            batch_inputs, batch_gt_instances, batch_img_metas)
        
        self.bbox_head.images = dual_stream_inputs
        
        self.bbox_head.images_no_copypaste = copy.deepcopy(dual_stream_inputs)
        # Edge prepare
        if self.epoch >= self.bbox_head.edge_loss_start_epoch:
            self.prepare_edges()
            
        # Copy_paste prepare
        if self.copy_paste_cache and len(batch_gt_instances) == len(self.copy_paste_cache):
            self.prepare_copy_paste_step2(dual_stream_inputs, dual_stream_targets)
        
        #save_tensor_as_images(self.bbox_head.images, output_prefix="after_copy_paste")
        #save_tensor_as_images(self.bbox_head.images_no_copypaste, output_prefix="before_copy_paste")
        
        dual_stream_data_samples = [] # gt & meta
        for gt_instances, img_metas in zip(dual_stream_targets, 
                                           batch_img_metas + batch_img_metas):
            data_sample = DetDataSample(metainfo=img_metas)
            data_sample.gt_instances = gt_instances
            dual_stream_data_samples.append(data_sample)
        
        # Prepare pseudo label
        ## Setp1
        feat = self.extract_feat(dual_stream_inputs)
        
        if self.epoch >= self.label_assign_pseudo_label_switch_eopch:
            results_list = self.bbox_head.predict(feat, dual_stream_data_samples)    # img_level -> fpn_level -> gt_obj_level
        else:
            if self.bbox_head.voronoi_type == "standard":
                results_list = self.generate_pseudo_targets(dual_stream_data_samples)
            elif self.bbox_head.voronoi_type in ['gaussian-orientation', 'gaussian-full']:
                results_list_assist = self.bbox_head.predict(feat, dual_stream_data_samples)
                results_list = self.generate_pseudo_targets(dual_stream_data_samples, results_list_assist)


        ## Step2
        for data_sample, results in zip(dual_stream_data_samples, results_list):
            # data_sample.gt_instances.allgt_bboxes = copy.deepcopy(data_sample.gt_instances.bboxes)
            mask = data_sample.gt_instances.bids[:, 1] == 0
            data_sample.gt_instances.bboxes.tensor[mask] = results.bboxes.tensor
            data_sample.gt_instances.labels[mask] = results.labels

        losses = self.bbox_head.loss(feat, dual_stream_data_samples)

        if self.epoch >= self.copy_paste_start_epoch:
            self.copy_paste_cache = []
            for images, instances in zip(dual_stream_inputs, results_list):
                self.copy_paste_cache.append(get_copy_paste_cache(images, 
                                                                  instances.bboxes.tensor, 
                                                                  instances.labels, 
                                                                  self.bbox_head.square_cls,
                                                                  self.num_copies))
            # vis
            if False:
                from PIL import Image, ImageDraw
                def obb2poly(obb):
                    cx, cy, w, h, t = obb
                    dw, dh = (w - 1) / 2, (h - 1) / 2
                    cost = np.cos(t)
                    sint = np.sin(t)
                    mrot = np.float32([[cost, -sint], [sint, cost]])
                    poly = np.float32([[-dw, -dh], [dw, -dh], [dw, dh], [-dw, dh]])
                    return np.matmul(poly, mrot.T) + np.float32([cx, cy])
                
                def draw_quadrilateral(img, points, color="red"):
                    """
                    在图片上绘制四边形
                    :param img: 图片
                    :param points: 四边形四个顶点坐标，格式为[x1, y1, x2, y2, x3, y3, x4, y4]
                    :param outline_color: 边框颜色，默认为红色
                    """
                    draw = ImageDraw.Draw(img)

                    # 绘制四边形边框
                    draw.polygon(list(points), outline=color, fill=None, width=5)  # 注意: 1. 必须是list, 非np.ndarray; 2.color非(0, 255, 0)

                    return img
                
                for i in range(2):
                    for j in range(len(self.copy_paste_cache[i])):
                        p, b, l = self.copy_paste_cache[i][j]  # 只看真实用的
                        img_tensor = p[:3, :, :]
                        img_uint8 = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min()) * 255
                        img_uint8 = img_uint8.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
                        img_pil = Image.fromarray(img_uint8)
                        b_np = obb2poly(b)
                        #img_pil = draw_quadrilateral(img_pil, b_np.reshape(-1), color="green")
                        img_id = batch_data_samples[i].metainfo['img_id']
                        img_pil.save(f'vis/{img_id}-{j}.png')
                    
                    
                    
                    
                    
                    
                    
                
                
        if self.debug:
            from PIL import Image, ImageDraw
            def obb2poly(obb):
                cx, cy, w, h, t = obb
                dw, dh = (w - 1) / 2, (h - 1) / 2
                cost = np.cos(t)
                sint = np.sin(t)
                mrot = np.float32([[cost, -sint], [sint, cost]])
                poly = np.float32([[-dw, -dh], [dw, -dh], [dw, dh], [-dw, dh]])
                return np.matmul(poly, mrot.T) + np.float32([cx, cy])
            
            def draw_quadrilateral(img, points, color="red"):
                draw = ImageDraw.Draw(img)
                draw.polygon(list(points), outline=color, fill=None, width=5)

                return img
            
            for i in range(2):
                for j in range(len(self.copy_paste_cache[i])):
                    p, b, l = self.copy_paste_cache[i][j]
                    img_tensor = p[:3, :, :]
                    img_uint8 = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min()) * 255
                    img_uint8 = img_uint8.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
                    img_pil = Image.fromarray(img_uint8)
                    b_np = obb2poly(b)
                    #img_pil = draw_quadrilateral(img_pil, b_np.reshape(-1), color="green")
                    img_id = batch_data_samples[i].metainfo['img_id']
                    img_pil.save(f'vis/{img_id}-{j}.png')
            def plot_one_rotated_box(img,
                                    obb,
                                    color=[0.0, 0.0, 128],
                                    label=None,
                                    line_thickness=None):
                width, height, theta = obb[2], obb[3], obb[4] / np.pi * 180
                if theta < 0:
                    width, height, theta = height, width, theta + 90
                rect = [(obb[0], obb[1]), (width, height), theta]
                poly = np.intp(np.round(
                    cv2.boxPoints(rect)))  # [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
                cv2.drawContours(
                    image=img, contours=[poly], contourIdx=-1, color=color, thickness=2)
                c1 = (int(obb[0]), int(obb[1]))
                if label:
                    tl = 2
                    tf = max(tl - 1, 1)  # font thickness
                    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                    cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
                    textcolor = [0, 0, 0] if max(color) > 192 else [255, 255, 255]
                    cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, textcolor, thickness=tf, lineType=cv2.LINE_AA)

            for i in range(len(batch_inputs_all)):
                img = batch_inputs_all[i]
                if self.bbox_head.vis[i]:
                    vor, wat = self.bbox_head.vis[i]
                    img[0, wat != wat.max()] += 2
                    img[:, vor != vor.max()] -= 1
                img = img.permute(1, 2, 0).cpu().numpy()
                img = np.ascontiguousarray(img[..., (2, 1, 0)] * 58 + 127)
                bb = batch_data_samples_all[i].gt_instances.bboxes.tensor
                ll = batch_data_samples_all[i].gt_instances.labels
                for b, l in zip(bb.cpu().numpy(), ll.cpu().numpy()):
                    b[2:4] = b[2:4].clip(3)
                    plot_one_rotated_box(img, b)
                if i < len(results_list):
                    bb = results_list[i].bboxes.tensor
                    if hasattr(results_list[i], 'informs'):
                        for b, l in zip(bb.cpu().numpy(), results_list[i].infoms.cpu().numpy()):
                            plot_one_rotated_box(img, b, (0, 255, 0), label=f'{l}')
                    else:
                        for b in bb.cpu().numpy():
                            plot_one_rotated_box(img, b, (0, 255, 0))
                img_id = batch_data_samples_all[i].metainfo['img_id']
                cv2.imwrite(f'debug/{img_id}_{i}.png', img)

        return losses
    
    def voronoi_diagram_watershed(self, mu, sigma, label, image):
        J = len(mu)
        down_sample = 2
        default_sigma = 4096

        # pos_thres = [0.994, 0.994, 0.999, 0.994, 0.994, 0.994, 0.994, 0.95, 0.95, 0.994, 0.95, 0.999, 0.994, 0.994, 0.95]
        # neg_thres = [0.005, 0.005, 0.6, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.6, 0.005, 0.005, 0.005]

        pos_thres = [self.bbox_head.voronoi_thres['default'][0]] * self.bbox_head.num_classes
        neg_thres = [self.bbox_head.voronoi_thres['default'][1]] * self.bbox_head.num_classes
        if 'override' in self.bbox_head.voronoi_thres.keys():
            for item in self.bbox_head.voronoi_thres['override']:
                for cls in item[0]:
                    pos_thres[cls] = item[1][0]
                    neg_thres[cls] = item[1][1]
        D = down_sample
        H, W = image.shape[-2:]
        h, w = H // D, W // D
        x = torch.linspace(0, h, h, device=mu.device)
        y = torch.linspace(0, w, w, device=mu.device)
        xy = torch.stack(torch.meshgrid(x, y, indexing='xy'), -1)
        vor = mu.new_zeros(J, h, w)
        # Get distribution for each instance
        mm = (mu.detach() / D).round()
        if self.bbox_head.voronoi_type == 'standard':
            sg = mu.new_tensor((default_sigma, 0, 0, default_sigma)).reshape(2, 2)
            sg = sg / D ** 2
            for j, m in enumerate(mm):
                vor[j] = gaussian_2d(xy.view(-1, 2), m[None], sg[None]).view(h, w)
        elif self.bbox_head.voronoi_type == 'gaussian-orientation':
            L, V = torch.linalg.eigh(sigma)
            L = L.detach().clone()
            L = L / (L[:, 0:1] * L[:, 1:2]).sqrt() * default_sigma
            sg = V.matmul(torch.diag_embed(L)).matmul(V.permute(0, 2, 1)).detach()
            sg = sg / D ** 2
            for j, (m, s) in enumerate(zip(mm, sg)):
                vor[j] = gaussian_2d(xy.view(-1, 2), m[None], s[None]).view(h, w)
        elif self.bbox_head.voronoi_type == 'gaussian-full':
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
        ridges = torch.conv2d(vor[None].float(), kernel.float(), padding=1)[0] != 0
        vor += 1
        pos_thres = val.new_tensor(pos_thres)
        neg_thres = val.new_tensor(neg_thres)
        vor[val < pos_thres[cls]] = 0
        vor[val < neg_thres[cls]] = J + 1
        vor[ridges] = J + 1

        cls_bg = torch.where(vor == J + 1, self.bbox_head.num_classes, cls)
        cls_bg = torch.where(vor == 0, -1, cls_bg)

        # PyTorch does not support watershed, use cv2
        img_uint8 = (image - image.min()) / (image.max() - image.min()) * 255
        img_uint8 = img_uint8.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
        img_uint8 = cv2.medianBlur(img_uint8, 3)
        markers = vor.detach().cpu().numpy().astype(np.int32)
        markers = vor.new_tensor(cv2.watershed(img_uint8, markers))

        pseudo_info = []
        for j in range(J):
            xy = (markers == j + 1).nonzero()[:, (1, 0)].float()
            if len(xy) == 0:
                pseudo_info.append(mu[j][0].item())  # cx
                pseudo_info.append(mu[j][1].item())  # cy
                pseudo_info.append(0)  # w_half
                pseudo_info.append(0)  # h_half
                pseudo_info.append(0)  # angle, set 0       
                continue
            xy = xy - mu[j]

            obj_w_half = torch.max(torch.abs(xy[:, 0]))
            obj_h_half = torch.max(torch.abs(xy[:, 1]))

            pseudo_info.append(mu[j][0].item())  # cx
            pseudo_info.append(mu[j][1].item())  # cy
            pseudo_info.append(obj_w_half * 2)  # w_half
            pseudo_info.append(obj_h_half * 2)  # h_half
            pseudo_info.append(0)  # angle, set 0
        
        return pseudo_info
    
  
    def generate_pseudo_targets(self, dual_stream_data_samples, results_list_assist=None) -> List:
        """
        
        Args:
            list:
                data_sample = DetDataSample(metainfo=img_metas)
                data_sample.gt_instances = gt_instances

                results_list_assist: support sigma info
        
        Returns:
            list: results = InstanceData()
                  results.bboxes = RotatedBoxes(pseudo_bboxes_selected)  # (obj_gt_num, 5)
                  results.scores = torch.ones_like(scores_all[:, 0])  # (obj_gt_num,)?
                  results.labels = labels_list[0].squeeze(1)  # (obj_gt_num,)?
        """
        results_list = []
        for i in range(len(dual_stream_data_samples)):
            data_sample = dual_stream_data_samples[i]
            bboxes = data_sample.gt_instances.bboxes.tensor  # num_gt, 5
            
            mask = data_sample.gt_instances.bids[:, 1] == 0
            
            
            mu = bboxes[:, :2][mask] # obj_num * 2, cx, cy
            J = len(mu)
            if J == 0:
                results_list.append(InstanceData())
                continue
            label = data_sample.gt_instances.labels[mask] # obj_num
            image = self.bbox_head.images_no_copypaste[i]

            if self.bbox_head.voronoi_type in ['gaussian-orientation', 'gaussian-full']:
                result_assist = results_list_assist[i]
                rbox_preds = result_assist.bboxes.tensor  # (N, 5)
                assert rbox_preds.shape[0] == mask.sum() and rbox_preds.shape[1] == 5

                cos_r = torch.cos(rbox_preds[:, -1])
                sin_r = torch.sin(rbox_preds[:, -1])
                R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
                sigma = R.matmul(torch.diag_embed(rbox_preds[:, 2:4] / 2.0)).matmul(R.permute(0, 2, 1)).view(-1, 2, 2)
            else:
                sigma = None

            if self.bbox_head.loss_voronoi.use_class_specific_watershed:
                pseudo_info = torch.ones(J, 5)
                for cur_class_id in range(self.bbox_head.num_classes):
                    cur_class_mask = label == cur_class_id
                    if not torch.any(cur_class_mask):
                        continue
                    cur_mu = mu[cur_class_mask]
                    cur_label = label[cur_class_mask]

                    if self.bbox_head.voronoi_type in ['gaussian-orientation', 'gaussian-full']:
                        cur_sigma = sigma[cur_class_mask]
                    else:
                        cur_sigma = None

                    cur_pseudo_info = self.voronoi_diagram_watershed(cur_mu, cur_sigma, cur_label, image)
                    cur_pseudo_info = torch.tensor(cur_pseudo_info).view(-1, 5)
                    pseudo_info[cur_class_mask] = cur_pseudo_info
            else:
                pseudo_info = self.voronoi_diagram_watershed(mu, sigma, label, image)
                pseudo_info = torch.tensor(pseudo_info).view(-1, 5)
            
            results = InstanceData()
            results.bboxes = RotatedBoxes(pseudo_info, device=mu.device)  # (obj_gt_num, 5)
            results.scores = torch.ones(J, device=mu.device)  # (obj_gt_num,)?
            results.labels = label  # (obj_gt_num,)?
            
            # if i <= 1 and torch.any(torch.all(pseudo_info[:, 2:] == 0, dim=1)):
            #     print(data_sample.metainfo["img_id"])
            #     assert False
                
            results_list.append(results)
            
        
        return results_list