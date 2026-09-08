from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _validate_x(x: Tensor) -> None:
    if x.ndim < 3:
        raise ValueError(f'Expected at least [B,C,T], got {tuple(x.shape)}')
    if x.shape[1] < 2:
        raise ValueError('Cyclic canonicalization requires at least two channels')


def roll_candidates(x: Tensor, inverse: bool = True) -> Tensor:
    """Return all cyclic channel-roll candidates.

    Input:  [B,C,...]
    Output: [B,K,C,...], where K=C.
    A logit at index k denotes an observed +k channel shift.  Canonicalization
    therefore applies -k when ``inverse=True``.
    """
    _validate_x(x)
    c = x.shape[1]
    direction = -1 if inverse else 1
    return torch.stack(
        [torch.roll(x, shifts=direction * k, dims=1) for k in range(c)], dim=1
    )


def normalized_entropy(probs: Tensor, eps: float = 1e-8) -> Tensor:
    k = probs.shape[-1]
    h = -(probs.clamp_min(eps) * probs.clamp_min(eps).log()).sum(dim=-1)
    return h / math.log(k)


def circular_mean_from_probs(probs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Return circular mean in channel units, radians, and resultant length.

    The resultant length is in [0,1] and acts as a confidence measure.  Unlike
    an arithmetic expectation, the circular mean is valid across the 0/K seam.
    """
    if probs.ndim != 2:
        raise ValueError(f'Expected [B,K] probabilities, got {tuple(probs.shape)}')
    k = probs.shape[1]
    angles = torch.arange(k, device=probs.device, dtype=probs.dtype) * (2 * math.pi / k)
    c = (probs * torch.cos(angles)).sum(dim=1)
    s = (probs * torch.sin(angles)).sum(dim=1)
    theta = torch.atan2(s, c).remainder(2 * math.pi)
    resultant = torch.sqrt(c.square() + s.square()).clamp(0.0, 1.0)
    shift_channels = theta * k / (2 * math.pi)
    return shift_channels, theta, resultant


def continuous_circular_shift(x: Tensor, shift_channels: Tensor | float) -> Tensor:
    """Periodically shift the channel axis by a continuous number of channels.

    Uses the Fourier shift theorem, so integer shifts agree with ``torch.roll``
    up to floating-point error and fractional shifts do not assume a 45-degree
    step for an eight-electrode ring.
    """
    _validate_x(x)
    b, c = x.shape[:2]
    shift = torch.as_tensor(shift_channels, dtype=x.dtype, device=x.device)
    if shift.ndim == 0:
        shift = shift.expand(b)
    if shift.shape != (b,):
        raise ValueError(f'shift_channels must be scalar or [B], got {tuple(shift.shape)}')

    spectrum = torch.fft.fft(x, dim=1)
    freq = torch.fft.fftfreq(c, d=1.0, device=x.device).to(x.dtype)
    phase = torch.exp(-2j * math.pi * shift[:, None] * freq[None, :])
    phase = phase.reshape(b, c, *([1] * (x.ndim - 2)))
    return torch.fft.ifft(spectrum * phase, dim=1).real


@dataclass
class CanonicalizationOutput:
    tensor: Tensor
    probs: Tensor
    entropy: Tensor
    confidence: Tensor
    expected_shift_channels: Tensor
    expected_shift_radians: Tensor

    def as_dict(self) -> Dict[str, Tensor]:
        return {
            'probs': self.probs,
            'entropy': self.entropy,
            'confidence': self.confidence,
            'expected_shift_channels': self.expected_shift_channels,
            'expected_shift_radians': self.expected_shift_radians,
        }


class SoftCyclicCanonicalizer(nn.Module):
    """Probability-weighted inverse rolls with uncertainty-safe residual gating.

    The V1 operator could collapse spatial structure when probabilities became
    diffuse.  Here confidence = 1 - normalized entropy.  A uniform distribution
    therefore leaves the input unchanged rather than averaging all channels.
    """
    def __init__(self, temperature: float = 1.0, confidence_residual: bool = True):
        super().__init__()
        if temperature <= 0:
            raise ValueError('temperature must be positive')
        self.temperature = float(temperature)
        self.confidence_residual = bool(confidence_residual)

    def forward(self, x: Tensor, logits: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        _validate_x(x)
        if logits.shape != (x.shape[0], x.shape[1]):
            raise ValueError(f'Expected logits {(x.shape[0], x.shape[1])}, got {tuple(logits.shape)}')
        probs = F.softmax(logits / self.temperature, dim=-1)
        candidates = roll_candidates(x, inverse=True)
        w = probs.reshape(probs.shape[0], probs.shape[1], *([1] * (x.ndim - 1)))
        aligned = (candidates * w).sum(dim=1)
        entropy = normalized_entropy(probs)
        confidence = (1.0 - entropy).clamp(0.0, 1.0)
        if self.confidence_residual:
            gate = confidence.reshape(confidence.shape[0], *([1] * (x.ndim - 1)))
            aligned = x + gate * (aligned - x)
        shift, theta, resultant = circular_mean_from_probs(probs)
        # Resultant is reported separately from entropy-derived confidence.
        info = CanonicalizationOutput(aligned, probs, entropy, confidence, shift, theta).as_dict()
        info['circular_resultant'] = resultant
        return aligned, info


class StraightThroughHardCanonicalizer(nn.Module):
    """Hard inverse roll with a straight-through estimator for the shift head."""
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, x: Tensor, logits: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        probs = F.softmax(logits / self.temperature, dim=-1)
        idx = probs.argmax(dim=-1)
        hard = F.one_hot(idx, num_classes=probs.shape[-1]).to(probs.dtype)
        weights = hard + probs - probs.detach()
        candidates = roll_candidates(x, inverse=True)
        w = weights.reshape(weights.shape[0], weights.shape[1], *([1] * (x.ndim - 1)))
        aligned = (candidates * w).sum(dim=1)
        entropy = normalized_entropy(probs)
        shift, theta, resultant = circular_mean_from_probs(probs)
        return aligned, {
            'probs': probs,
            'hard_index': idx,
            'entropy': entropy,
            'confidence': resultant,
            'expected_shift_channels': shift,
            'expected_shift_radians': theta,
            'circular_resultant': resultant,
        }


class ContinuousFourierCanonicalizer(nn.Module):
    """Continuous cyclic inverse transform derived from a circular distribution."""
    def __init__(self, temperature: float = 1.0, confidence_residual: bool = True):
        super().__init__()
        self.temperature = float(temperature)
        self.confidence_residual = bool(confidence_residual)

    def forward(self, x: Tensor, logits: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        probs = F.softmax(logits / self.temperature, dim=-1)
        shift, theta, resultant = circular_mean_from_probs(probs)
        shifted = continuous_circular_shift(x, -shift)
        if self.confidence_residual:
            gate = resultant.reshape(resultant.shape[0], *([1] * (x.ndim - 1)))
            shifted = x + gate * (shifted - x)
        entropy = normalized_entropy(probs)
        return shifted, {
            'probs': probs,
            'entropy': entropy,
            'confidence': resultant,
            'expected_shift_channels': shift,
            'expected_shift_radians': theta,
            'circular_resultant': resultant,
        }


class UniformCyclicMixer(nn.Module):
    """Negative control: equal weighting of all cyclic candidates."""
    def __init__(self, safe_residual: bool = False):
        super().__init__()
        self.safe_residual = bool(safe_residual)

    def forward(self, x: Tensor, logits: Optional[Tensor] = None) -> Tuple[Tensor, Dict[str, Tensor]]:
        b, c = x.shape[:2]
        probs = torch.full((b, c), 1.0 / c, device=x.device, dtype=x.dtype)
        candidates = roll_candidates(x, inverse=True)
        w = probs.reshape(b, c, *([1] * (x.ndim - 1)))
        mixed = (candidates * w).sum(dim=1)
        if self.safe_residual:
            mixed = x  # uniform distribution has zero directional confidence
        shift, theta, resultant = circular_mean_from_probs(probs)
        return mixed, {
            'probs': probs,
            'entropy': torch.ones(b, device=x.device, dtype=x.dtype),
            'confidence': torch.zeros(b, device=x.device, dtype=x.dtype),
            'expected_shift_channels': shift,
            'expected_shift_radians': theta,
            'circular_resultant': resultant,
        }


class RandomCyclicMixer(nn.Module):
    """Negative control with input-independent random simplex weights."""
    def __init__(self, seed: int = 0):
        super().__init__()
        self.seed = int(seed)
        self.register_buffer('_calls', torch.zeros((), dtype=torch.long), persistent=False)

    def forward(self, x: Tensor, logits: Optional[Tensor] = None) -> Tuple[Tensor, Dict[str, Tensor]]:
        b, c = x.shape[:2]
        gen = torch.Generator(device=x.device)
        gen.manual_seed(self.seed + int(self._calls.item()))
        self._calls += 1
        raw = torch.rand((b, c), generator=gen, device=x.device, dtype=x.dtype).clamp_min(1e-6)
        probs = raw / raw.sum(dim=-1, keepdim=True)
        candidates = roll_candidates(x, inverse=True)
        w = probs.reshape(b, c, *([1] * (x.ndim - 1)))
        mixed = (candidates * w).sum(dim=1)
        entropy = normalized_entropy(probs)
        shift, theta, resultant = circular_mean_from_probs(probs)
        return mixed, {
            'probs': probs,
            'entropy': entropy,
            'confidence': resultant,
            'expected_shift_channels': shift,
            'expected_shift_radians': theta,
            'circular_resultant': resultant,
        }
