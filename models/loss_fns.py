import torch
from torch.nn import functional as F

from protein_learning.common.helpers import safe_normalize

def trig_loss(output, target, target_mask, chi_num_mask):
    sum_sq_diffs = ((output - target) ** 2).sum(dim=-1)
    masked_diffs = sum_sq_diffs * target_mask * chi_num_mask
    loss_torsion = masked_diffs.sum(dim=-1).sum(dim=0)

    w_anglenorm = 0.02
    unmasked_norm_loss = torch.abs((output ** 2).sum(dim=-1) - 1)
    masked_norm_loss = unmasked_norm_loss * target_mask * chi_num_mask
    loss_anglenorm = masked_norm_loss.sum(dim=-1).sum()

    loss = loss_torsion + w_anglenorm * loss_anglenorm
    return loss, loss_torsion, loss_anglenorm

def mse_loss(output, target, target_mask, chi_num_mask):
    output, target = map(
        lambda x : torch.atan2(*safe_normalize(x).unbind(dim=-1)),
        (output, target)
    )
    sq_diffs = ((output - target) ** 2) * target_mask * chi_num_mask
    loss = torch.sum(sq_diffs)
    return loss

def mae_loss(output, target, target_mask, chi_num_mask):
    output, target = map(
        lambda x : torch.atan2(*safe_normalize(x).unbind(dim=-1)),
        (output, target)
    )
    sq_diffs = (output - target).abs() * target_mask * chi_num_mask
    loss = torch.sum(sq_diffs)
    return loss

def huber_loss(output, target, target_mask, chi_num_mask, as_trig=True, w_anglenorm=0.02):
    delta = 1.0
    if not as_trig:
        output, target = map(
            lambda x : torch.atan2(*safe_normalize(x).unbind(dim=-1)),
            (output, target)
        )
        output = output * target_mask * chi_num_mask
        target = target * target_mask * chi_num_mask
        loss = F.huber_loss(output, target, delta=delta, reduction='sum')
        return loss
    else:
        unmasked_loss_torsion = F.huber_loss(
            output,
            target,
            delta=delta,
            reduction='none'
        ).sum(dim=-1)
        loss_torsion = (unmasked_loss_torsion * target_mask * chi_num_mask).sum()

        norms = (output ** 2).sum(-1)
        unmasked_loss_anglenorm = F.huber_loss(
            norms,
            torch.ones(norms.shape, device=norms.device),
            delta=delta,
            reduction='none'
        )
        loss_anglenorm = (unmasked_loss_anglenorm * target_mask * chi_num_mask).sum()

        loss = loss_torsion + w_anglenorm * loss_anglenorm
        return loss, loss_torsion, loss_anglenorm
