import torch.nn as nn
from .voronoi_watershed_loss import VoronoiWatershedLoss
from .sam_loss import SamLoss
from mmrotate.registry import MODELS


@MODELS.register_module()
class PgdmLoss(nn.Module):
    """Proposed in Point2RBox-v3, https://arxiv.org/abs/2509.26281.
    1) Sparse scenarios, sam; Dense scenarios, watershed.
    """
    def __init__(self,
                down_sample=2,
                default_sigma=4096,
                loss_weight_watershed=1.0,
                loss_weight_sam=1.0,
                topk=0.95,
                alpha=0.1,
                sam_checkpoint='./mobile_sam.pt',
                sam_type='vit_t',
                sam_device='cuda',
                sam_instance_thr=-1,
                mask_filter_config=None,
                use_class_specific=False,
                debug=False):
        """
        """
        super(PgdmLoss, self).__init__()
        self.use_class_specific = use_class_specific
        self.sam_instance_thr = sam_instance_thr  
        self.voronoi_watershed_loss = VoronoiWatershedLoss(down_sample, loss_weight_watershed, topk, alpha, debug)
        self.sam_loss = SamLoss(sam_checkpoint, sam_type, sam_device, mask_filter_config, loss_weight_sam, debug)
    
    def forward(self, pred,
            label, image, 
            pos_thres, neg_thres, 
            voronoi='gaussian-orientation'):
        mu, sigma = pred
        J = len(sigma)
        if J == 0:
            return sigma.sum()
        
        if J <= self.sam_instance_thr:  # sparse
            loss = self.sam_loss(
                (mu, sigma), label, image)
            self.vis = self.sam_loss.vis
            return loss
        else:  # dense
            loss = self.voronoi_watershed_loss(
                (mu, sigma), label, image, pos_thres, neg_thres, voronoi)  # In fact, pos_thres, neg_thres and voronoi should be
                # defined as object variable; however here retain the inappropriate interfaces solely for compatibility with point2
                # rbox-v2 version.
            self.vis = self.voronoi_watershed_loss.vis
            return loss