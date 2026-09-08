from __future__ import annotations

import math
import torch
from torch import Tensor
import torch.nn.functional as F


def wrapped_channel_error(pred: Tensor, target: Tensor, n_channels: int) -> Tensor:
    """Signed shortest circular error in channel units."""
    half = n_channels / 2.0
    return (pred - target + half).remainder(n_channels) - half


def circular_regression_loss(pred_shift: Tensor, target_shift: Tensor, n_channels: int) -> Tensor:
    err_angle = wrapped_channel_error(pred_shift, target_shift, n_channels) * (2 * math.pi / n_channels)
    return (1.0 - torch.cos(err_angle)).mean()


def von_mises_targets(target_shift: Tensor, n_channels: int, kappa: float = 8.0) -> Tensor:
    bins = torch.arange(n_channels, device=target_shift.device, dtype=target_shift.dtype)
    target_angle = target_shift[:, None] * (2 * math.pi / n_channels)
    bin_angle = bins[None, :] * (2 * math.pi / n_channels)
    weights = torch.exp(kappa * torch.cos(bin_angle - target_angle))
    return weights / weights.sum(dim=-1, keepdim=True)


def circular_distribution_loss(logits: Tensor, target_shift: Tensor, kappa: float = 8.0) -> Tensor:
    target = von_mises_targets(target_shift, logits.shape[-1], kappa=kappa)
    return -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def normalized_feature_consistency(aligned: Tensor, reference: Tensor) -> Tensor:
    if aligned.shape != reference.shape:
        raise ValueError(f'Feature shape mismatch: {aligned.shape} vs {reference.shape}')
    a = F.normalize(aligned.flatten(1), dim=1)
    r = F.normalize(reference.flatten(1), dim=1)
    return (1.0 - (a * r).sum(dim=1)).mean()
