import torch
import torch.nn.functional as F


def masked_bce_dice(logits, target, eps=1.0):
    """logits: (B,1,H,W). target: (B,H,W) with -1 = ignore, 0 = dry, 1 = water."""
    logits = logits.squeeze(1)
    valid = target != -1
    if valid.sum() == 0:
        return logits.sum() * 0.0

    t = (target == 1).float()
    bce = F.binary_cross_entropy_with_logits(logits[valid], t[valid])

    p = torch.sigmoid(logits) * valid
    t = t * valid
    inter = (p * t).sum()
    dice = 1 - (2 * inter + eps) / (p.sum() + t.sum() + eps)
    return bce + dice