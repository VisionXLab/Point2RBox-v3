import torch
import torch.nn as nn
from mmrotate.registry import MODELS
from mmdet.models.losses.utils import weighted_loss


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
            overlap_scale (torch.Tensor, optional): scale matrix for overlap loss with shape(N, N).

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
            overlap_scale=overlap_scale))