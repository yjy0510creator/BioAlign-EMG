from __future__ import annotations

from typing import Dict, Literal, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .cyclic_ops import (
    ContinuousFourierCanonicalizer,
    RandomCyclicMixer,
    SoftCyclicCanonicalizer,
    StraightThroughHardCanonicalizer,
    UniformCyclicMixer,
)


class PerChannelEncoder(nn.Module):
    def __init__(self, out_dim: int = 12):
        super().__init__()
        branch = out_dim
        self.branches = nn.ModuleList([
            nn.Conv1d(1, branch, k, padding=k // 2) for k in (3, 5, 9)
        ])
        self.fuse = nn.Sequential(
            nn.Conv1d(branch * 3, out_dim, 1, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f'Expected [B,C,T], got {tuple(x.shape)}')
        b, c, t = x.shape
        z = x.reshape(b * c, 1, t)
        z = torch.cat([m(z) for m in self.branches], dim=1)
        z = self.fuse(z)
        return z.reshape(b, c, z.shape[1], t)


class CorrespondenceHead(nn.Module):
    def __init__(self, feature_dim: int, n_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(feature_dim, 24, 3, padding=1, padding_mode='circular', bias=False),
            nn.BatchNorm1d(24),
            nn.GELU(),
            nn.Conv1d(24, 12, 3, padding=1, padding_mode='circular'),
            nn.GELU(),
        )
        self.out = nn.Linear(12 * n_channels, n_channels)

    def forward(self, f: Tensor) -> Tensor:
        # f [B,C,D,T] -> pooled [B,D,C]
        z = f.mean(dim=-1).transpose(1, 2)
        z = self.net(z)
        return self.out(z.flatten(1))


class DepthwiseTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        pad = 2 * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=pad, dilation=dilation, groups=channels, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(x + self.net(x))


class TemporalClassifier(nn.Module):
    def __init__(self, n_channels: int, feature_dim: int, n_classes: int, hidden: int = 56):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(n_channels * feature_dim, hidden, 1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[DepthwiseTemporalBlock(hidden, d) for d in (1, 2, 4)])
        self.out = nn.Linear(hidden, n_classes)

    def forward(self, f: Tensor) -> Tuple[Tensor, Tensor]:
        b, c, d, t = f.shape
        z = self.proj(f.reshape(b, c * d, t))
        z = self.blocks(z)
        embedding = z.mean(dim=-1)
        return self.out(embedding), embedding


CanonicalizerName = Literal['none', 'soft', 'hard_ste', 'continuous', 'uniform', 'uniform_safe', 'random']


class BioAlignV2(nn.Module):
    """BioAlign with an identifiable shift head and pattern-preserving safeguards.

    ``return_aux=True`` exposes the latent distribution and pre/post features so
    the mechanism can be tested directly rather than inferred from class scores.
    """
    def __init__(
        self,
        n_channels: int = 8,
        n_classes: int = 7,
        feature_dim: int = 12,
        canonicalizer: CanonicalizerName = 'continuous',
        temperature: float = 1.0,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.encoder = PerChannelEncoder(feature_dim)
        self.shift_head = CorrespondenceHead(feature_dim, n_channels)
        if canonicalizer == 'none':
            self.canonicalizer = None
        elif canonicalizer == 'soft':
            self.canonicalizer = SoftCyclicCanonicalizer(temperature, confidence_residual=True)
        elif canonicalizer == 'hard_ste':
            self.canonicalizer = StraightThroughHardCanonicalizer(temperature)
        elif canonicalizer == 'continuous':
            self.canonicalizer = ContinuousFourierCanonicalizer(temperature, confidence_residual=True)
        elif canonicalizer == 'uniform':
            self.canonicalizer = UniformCyclicMixer(safe_residual=False)
        elif canonicalizer == 'uniform_safe':
            self.canonicalizer = UniformCyclicMixer(safe_residual=True)
        elif canonicalizer == 'random':
            self.canonicalizer = RandomCyclicMixer(seed=3407)
        else:
            raise ValueError(f'Unknown canonicalizer: {canonicalizer}')
        self.canonicalizer_name = canonicalizer
        self.classifier = TemporalClassifier(n_channels, feature_dim, n_classes)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def forward(self, x: Tensor, return_aux: bool = False):
        f_pre = self.encoder(x)
        logits_shift = self.shift_head(f_pre)
        if self.canonicalizer is None:
            f_post = f_pre
            probs = F.softmax(logits_shift, dim=-1)
            aux_align: Dict[str, Tensor] = {
                'probs': probs,
                'entropy': -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(-1) / torch.log(torch.tensor(float(self.n_channels), device=x.device)),
            }
        else:
            f_post, aux_align = self.canonicalizer(f_pre, logits_shift)
        logits_gesture, embedding = self.classifier(f_post)
        if not return_aux:
            return logits_gesture
        aux = dict(aux_align)
        aux.update({
            'shift_logits': logits_shift,
            'features_pre': f_pre,
            'features_post': f_post,
            'embedding': embedding,
        })
        return logits_gesture, aux


class TinyTCN(nn.Module):
    def __init__(self, n_channels: int = 8, n_classes: int = 7, hidden: int = 56):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, hidden, 5, padding=2, bias=False), nn.BatchNorm1d(hidden), nn.GELU(),
            DepthwiseTemporalBlock(hidden, 1),
            DepthwiseTemporalBlock(hidden, 2),
            DepthwiseTemporalBlock(hidden, 4),
        )
        self.out = nn.Linear(hidden, n_classes)

    def forward(self, x: Tensor, return_aux: bool = False):
        z = self.net(x)
        emb = z.mean(-1)
        logits = self.out(emb)
        return (logits, {'embedding': emb}) if return_aux else logits


class CircularConvTCN(nn.Module):
    """Parameter-matched topology-aware baseline without canonicalization."""
    def __init__(self, n_channels: int = 8, n_classes: int = 7, feature_dim: int = 12):
        super().__init__()
        self.encoder = PerChannelEncoder(feature_dim)
        self.spatial = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, 3, padding=1, padding_mode='circular', bias=False),
            nn.BatchNorm1d(feature_dim), nn.GELU(),
        )
        self.classifier = TemporalClassifier(n_channels, feature_dim, n_classes)

    def forward(self, x: Tensor, return_aux: bool = False):
        f = self.encoder(x)
        b, c, d, t = f.shape
        z = f.permute(0, 3, 2, 1).reshape(b * t, d, c)
        z = self.spatial(z)
        f2 = z.reshape(b, t, d, c).permute(0, 3, 2, 1)
        logits, emb = self.classifier(f2)
        return (logits, {'embedding': emb, 'features_post': f2}) if return_aux else logits
