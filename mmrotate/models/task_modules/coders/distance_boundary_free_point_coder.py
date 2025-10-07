# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.models.task_modules.coders.base_bbox_coder import BaseBBoxCoder
from mmdet.structures.bbox import bbox2distance, distance2bbox

from mmrotate.registry import TASK_UTILS
from mmrotate.structures.bbox import rbox2hbox, hbox2rbox


@TASK_UTILS.register_module()
class DistanceBoundaryFreePointCoder(BaseBBoxCoder):
    """Distance Angle Point BBox coder.

    This coder encodes gt bboxes (x, y, w, h, theta) into (top, bottom, left,
    right, rs, p0, p1, p2) and decode it back to the original.

    Args:
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Defaults to True.
    """

    encode_size = 10

    def __init__(self, clip_border=True, angle_version='oc'):
        super(BaseBBoxCoder, self).__init__()
        self.clip_border = clip_border
        self.angle_version = angle_version

    def encode(self, points, gt_bboxes, max_dis=None, eps=0.1):
        """Encode bounding box to distances.

        Args:
            points (Tensor): Shape (N, 2), The format is [x, y].
            gt_bboxes (Tensor): Shape (N, 5), The format is "xywha"
            max_dis (float): Upper bound of the distance. Default None.
            eps (float): a small value to ensure target < max_dis, instead <=.
                Default 0.1.

        Returns:
            Tensor: Box transformation deltas. The shape is (N, 8).
        """
        assert points.size(0) == gt_bboxes.size(0)
        assert points.size(-1) == 2
        assert gt_bboxes.size(-1) == 5

        hbb = rbox2hbox(gt_bboxes)
        dis = bbox2distance(points, hbb, max_dis, eps)

        tbb = rbox2hbox(gt_bboxes - gt_bboxes.new_tensor([[0, 0, 0, 0, torch.pi / 4]]))
        tis = bbox2distance(points, tbb, max_dis, eps)

        t = gt_bboxes[..., 4] % (torch.pi / 2)
        p0 = torch.cos(4 * t)
        p1 = torch.sin(4 * t)

        return torch.stack([*dis.unbind(dim=-1), *tis.unbind(dim=-1), p0, p1], dim=-1)

    def decode(self, points, pred_bboxes, max_shape=None):
        """Decode distance prediction to bounding box.

        Args:
            points (Tensor): Shape (B, N, 2) or (N, 2).
            pred_bboxes (Tensor): Distance from the given point to 4
                boundaries and angle (left, top, right, bottom, angle).
                Shape (B, N, 8) or (N, 8)
            max_shape (Sequence[int] or torch.Tensor or Sequence[
                Sequence[int]],optional): Maximum bounds for boxes, specifies
                (H, W, C) or (H, W). If priors shape is (B, N, 4), then
                the max_shape should be a Sequence[Sequence[int]],
                and the length of max_shape should also be B.
                Default None.
        Returns:
            Tensor: Boxes with shape (N, 5) or (B, N, 5)
        """
        assert points.size(0) == pred_bboxes.size(0)
        assert points.size(-1) == 2
        assert pred_bboxes.size(-1) == self.encode_size
        if self.clip_border is False:
            max_shape = None
        hbox = distance2bbox(points, pred_bboxes[..., :4], max_shape)
        rbox = hbox2rbox(hbox)
        x, y, W, H = rbox[..., :4].unbind(dim=-1)

        # T = pred_bboxes[..., 4] + pred_bboxes[..., 5]
        tbox = distance2bbox(points, pred_bboxes[..., 4:8], max_shape)
        rtbox = hbox2rbox(tbox)
        xt, yt, Wt, Ht = rtbox[..., :4].unbind(dim=-1)

        p0 = pred_bboxes[..., 8]
        p1 = pred_bboxes[..., 9]
        t = torch.atan2(p1, p0) / 4
        t = t % (torch.pi / 2)

        cosa = torch.cos(t)
        sina = torch.sin(t)
        cosb = torch.cos(t - torch.pi / 4).abs()
        sinb = torch.sin(t - torch.pi / 4).abs()
        m = torch.stack(
            (cosa, sina, sina, cosa, cosb, sinb, sinb, cosb), -1).view(-1, 4, 2)
        b = torch.stack((W, H, Wt, Ht), -1)[..., None]
        # m = torch.stack(
        #     (cosa, sina, sina, cosa), -1).view(-1, 2, 2)
        # b = torch.stack((W, H), -1)[..., None]
        # m = torch.stack(
        #     (cosb, sinb, sinb, cosb), -1).view(-1, 2, 2)
        # b = torch.stack((Wt, Ht), -1)[..., None]
        wh = torch.linalg.lstsq(m, b).solution[:, :, 0].clamp(1e-3)
        
        # i = torch.argmin(wh, dim=0)[0]
        # print(f't {t[i].item():.3f}', 
        #       f'W {W[i].item():.3f}', 
        #       f'H {H[i].item():.3f}', 
        #       f'w {wh[i, 0].item():.3f}', 
        #       f'h {wh[i, 1].item():.3f}')

        return torch.stack([x, y, *wh.unbind(dim=-1), t], dim=-1)
