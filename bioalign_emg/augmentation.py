from __future__ import annotations

import torch

from .constants import N_CHANNELS


def ring_augment(
    x: torch.Tensor,
    *,
    enable_roll: bool = True,
    gain_jitter: float = 0.15,
    attenuation_prob: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Topology-constrained training perturbation used in the frozen study."""
    if x.ndim != 3 or x.shape[1] != N_CHANNELS:
        raise ValueError(f"Expected [batch,{N_CHANNELS},time], got {tuple(x.shape)}")

    batch = x.shape[0]
    out = x.clone()
    device = x.device
    shift_labels = torch.zeros(batch, dtype=torch.long, device=device)

    if enable_roll:
        shift_labels = torch.randint(0, N_CHANNELS, (batch,), device=device)
        for index in range(batch):
            offset = int(shift_labels[index].item())
            if offset:
                out[index] = torch.roll(out[index], shifts=offset, dims=0)

    gains = 1.0 + gain_jitter * torch.randn(
        batch,
        N_CHANNELS,
        1,
        device=device,
    )
    out = out * gains.clamp(0.65, 1.35)

    if attenuation_prob > 0:
        apply = torch.rand(batch, device=device) < attenuation_prob
        channels = torch.randint(0, N_CHANNELS, (batch,), device=device)
        attenuation = 0.15 + 0.45 * torch.rand(batch, device=device)
        for index in range(batch):
            if bool(apply[index]):
                out[index, int(channels[index].item())] *= attenuation[index]

    return out, shift_labels
