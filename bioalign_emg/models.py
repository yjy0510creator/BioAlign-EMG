from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import N_CHANNELS, N_CLASSES


class DSConv1d(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.10):
        super().__init__()
        self.depth = nn.Conv1d(
            channels,
            channels,
            kernel_size=5,
            padding=2 * dilation,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.point = nn.Conv1d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depth(x)
        y = self.point(y)
        y = self.bn(y)
        y = F.gelu(y)
        return self.dropout(y)


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv1 = DSConv1d(channels, dilation)
        self.conv2 = DSConv1d(channels, dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.conv2(self.conv1(x)))


class TCNBackbone(nn.Module):
    def __init__(self, in_channels: int = N_CHANNELS, hidden: int = 48):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden, 1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResidualTCNBlock(hidden, 1),
            ResidualTCNBlock(hidden, 2),
            ResidualTCNBlock(hidden, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.proj(x))


class PlainTCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TCNBackbone()
        self.head = nn.Linear(48, N_CLASSES)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        embedding = features.mean(dim=-1)
        return {"logits": self.head(embedding), "embedding": embedding}


class SEGate(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        hidden = max(4, channels // 8)
        self.net = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.sigmoid(self.net(x.mean(dim=-1)))
        return x * weights.unsqueeze(-1)


class SETCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TCNBackbone()
        self.se = SEGate(48)
        self.head = nn.Linear(48, N_CLASSES)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.se(self.backbone(x))
        embedding = features.mean(dim=-1)
        return {"logits": self.head(embedding), "embedding": embedding}


class CBAM1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        hidden = max(4, channels // 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )
        self.temporal = nn.Conv1d(2, 1, 7, padding=3)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        average = x.mean(dim=-1)
        maximum = x.amax(dim=-1)
        channel_attention = torch.sigmoid(self.mlp(average) + self.mlp(maximum))
        x = x * channel_attention.unsqueeze(-1)

        average_t = x.mean(dim=1, keepdim=True)
        maximum_t = x.amax(dim=1, keepdim=True)
        temporal_attention = torch.sigmoid(
            self.temporal(torch.cat([average_t, maximum_t], dim=1))
        )
        return x * temporal_attention, channel_attention, temporal_attention.squeeze(1)


class CBAMTCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TCNBackbone()
        self.cbam = CBAM1d(48)
        self.head = nn.Linear(48, N_CLASSES)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features, channel_attention, temporal_attention = self.cbam(
            self.backbone(x)
        )
        embedding = features.mean(dim=-1)
        return {
            "logits": self.head(embedding),
            "embedding": embedding,
            "channel_attention": channel_attention,
            "temporal_attention": temporal_attention,
        }


class PerChannelStem(nn.Module):
    """Encode each ring electrode independently before correspondence recovery."""

    def __init__(self, d: int = 12):
        super().__init__()
        self.d = d
        self.branch3 = nn.Conv1d(1, d, 3, padding=1, bias=False)
        self.branch5 = nn.Conv1d(1, d, 5, padding=2, bias=False)
        self.branch9 = nn.Conv1d(1, d, 9, padding=4, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv1d(d * 3, d, 1, bias=False),
            nn.BatchNorm1d(d),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, time = x.shape
        y = x.reshape(batch * channels, 1, time)
        y = torch.cat(
            [self.branch3(y), self.branch5(y), self.branch9(y)],
            dim=1,
        )
        y = self.fuse(y)
        return y.reshape(batch, channels, self.d, time)


class RingShiftEstimator(nn.Module):
    def __init__(self, d: int = 12, hidden: int = 24):
        super().__init__()
        self.pre = nn.Conv1d(d, hidden, 1)
        self.ring = nn.Conv1d(hidden, hidden, 3, bias=False)
        self.head = nn.Linear(hidden * N_CHANNELS, N_CHANNELS)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features.mean(dim=-1).transpose(1, 2)
        x = F.gelu(self.pre(x))
        x = F.pad(x, (1, 1), mode="circular")
        x = F.gelu(self.ring(x))
        return self.head(x.flatten(1))


def soft_inverse_roll(
    features: torch.Tensor,
    shift_logits: torch.Tensor,
) -> torch.Tensor:
    probabilities = torch.softmax(shift_logits, dim=-1)
    candidates = [
        torch.roll(features, shifts=-offset, dims=1)
        for offset in range(N_CHANNELS)
    ]
    stacked = torch.stack(candidates, dim=1)
    return (
        stacked * probabilities[:, :, None, None, None]
    ).sum(dim=1)


class RingReliabilityGate(nn.Module):
    def __init__(self, d: int = 12, hidden: int = 24):
        super().__init__()
        self.pre = nn.Conv1d(d, hidden, 1)
        self.ring = nn.Conv1d(hidden, hidden, 3, bias=False)
        self.out = nn.Conv1d(hidden, 1, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features.mean(dim=-1).transpose(1, 2)
        x = F.gelu(self.pre(x))
        x = F.pad(x, (1, 1), mode="circular")
        x = F.gelu(self.ring(x))
        return torch.sigmoid(self.out(x)).squeeze(1)


class TimeReliabilityGate(nn.Module):
    def __init__(self, d: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d, d, 5, padding=2, groups=d),
            nn.Conv1d(d, 8, 1),
            nn.GELU(),
            nn.Conv1d(8, 1, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features.mean(dim=1)
        return torch.sigmoid(self.net(x)).squeeze(1)


class BioSelectEMG(nn.Module):
    """Exploratory parent architecture retained for checkpoint compatibility."""

    def __init__(self, d: int = 12, hidden: int = 56):
        super().__init__()
        self.stem = PerChannelStem(d)
        self.shift = RingShiftEstimator(d)
        self.ring_gate = RingReliabilityGate(d)
        self.time_gate = TimeReliabilityGate(d)
        self.project = nn.Sequential(
            nn.Conv1d(N_CHANNELS * d, hidden, 1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        self.tcn = nn.Sequential(
            ResidualTCNBlock(hidden, 1),
            ResidualTCNBlock(hidden, 2),
            ResidualTCNBlock(hidden, 4),
        )
        self.head = nn.Linear(hidden, N_CLASSES)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.stem(x)
        shift_logits = self.shift(features)
        aligned = soft_inverse_roll(features, shift_logits)
        channel_reliability = self.ring_gate(aligned)
        aligned = aligned * channel_reliability[:, :, None, None]
        temporal_reliability = self.time_gate(aligned)
        aligned = aligned * temporal_reliability[:, None, None, :]
        batch, channels, d, time = aligned.shape
        z = aligned.reshape(batch, channels * d, time)
        z = self.tcn(self.project(z))
        embedding = z.mean(dim=-1)
        return {
            "logits": self.head(embedding),
            "embedding": embedding,
            "shift_logits": shift_logits,
            "shift_prob": torch.softmax(shift_logits, dim=-1),
            "channel_reliability": channel_reliability,
            "temporal_reliability": temporal_reliability,
        }


class BioAlignEMG(BioSelectEMG):
    """Final paper/checkpoint-compatible BioAlign model.

    The two gate modules inherited from BioSelectEMG remain registered but are not
    called in this forward path. This preserves the original 33,549-parameter
    state-dict layout and manuscript-reported registered parameter count.
    """

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.stem(x)
        shift_logits = self.shift(features)
        aligned = soft_inverse_roll(features, shift_logits)
        batch, channels, d, time = aligned.shape
        z = aligned.reshape(batch, channels * d, time)
        z = self.tcn(self.project(z))
        embedding = z.mean(dim=-1)
        return {
            "logits": self.head(embedding),
            "embedding": embedding,
            "shift_logits": shift_logits,
            "shift_prob": torch.softmax(shift_logits, dim=-1),
        }


class BioAlignNoCircularAlignment(BioSelectEMG):
    """Full-retraining ablation that bypasses inverse-roll canonicalization."""

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.stem(x)
        shift_logits = self.shift(features)
        aligned = features
        batch, channels, d, time = aligned.shape
        z = aligned.reshape(batch, channels * d, time)
        z = self.tcn(self.project(z))
        embedding = z.mean(dim=-1)
        return {
            "logits": self.head(embedding),
            "embedding": embedding,
            "shift_logits": shift_logits,
            "shift_prob": torch.softmax(shift_logits, dim=-1),
        }


class BioAlignHardArgmax(BioSelectEMG):
    """Optional mechanism control: use one hard inverse roll."""

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.stem(x)
        shift_logits = self.shift(features)
        offsets = shift_logits.argmax(dim=-1)
        aligned = torch.stack(
            [
                torch.roll(features[index], shifts=-int(offsets[index]), dims=0)
                for index in range(len(features))
            ],
            dim=0,
        )
        batch, channels, d, time = aligned.shape
        z = self.tcn(self.project(aligned.reshape(batch, channels * d, time)))
        embedding = z.mean(dim=-1)
        return {
            "logits": self.head(embedding),
            "embedding": embedding,
            "shift_logits": shift_logits,
        }


class BioAlignUniformMixture(BioSelectEMG):
    """Optional mechanism control: equal-weight average of all inverse rolls."""

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.stem(x)
        candidates = [
            torch.roll(features, shifts=-offset, dims=1)
            for offset in range(N_CHANNELS)
        ]
        aligned = torch.stack(candidates, dim=1).mean(dim=1)
        batch, channels, d, time = aligned.shape
        z = self.tcn(self.project(aligned.reshape(batch, channels * d, time)))
        embedding = z.mean(dim=-1)
        return {"logits": self.head(embedding), "embedding": embedding}


class BioAlignEMGCompact(BioAlignEMG):
    """Forward-equivalent clean architecture with inactive gates removed.

    This class has 31,299 registered parameters. It must be retrained before its
    numerical results are compared with the paper's legacy-compatible model.
    """

    def __init__(self, d: int = 12, hidden: int = 56):
        super().__init__(d=d, hidden=hidden)
        del self.ring_gate
        del self.time_gate


MODEL_REGISTRY = {
    "TCN_plain": PlainTCN,
    "TCN_RingAug": PlainTCN,
    "SE_RingAug": SETCN,
    "CBAM_RingAug": CBAMTCN,
    "BioAlign_RingAug": BioAlignEMG,
    "BioAlign_NoAlignment": BioAlignNoCircularAlignment,
    "BioAlign_HardArgmax": BioAlignHardArgmax,
    "BioAlign_UniformMixture": BioAlignUniformMixture,
    "BioAlign_Compact": BioAlignEMGCompact,
}

AUGMENTED_MODELS = {
    "TCN_RingAug",
    "SE_RingAug",
    "CBAM_RingAug",
    "BioAlign_RingAug",
    "BioAlign_NoAlignment",
    "BioAlign_HardArgmax",
    "BioAlign_UniformMixture",
    "BioAlign_Compact",
}


def build_model(name: str) -> nn.Module:
    try:
        return MODEL_REGISTRY[name]()
    except KeyError as exc:
        raise KeyError(
            f"Unknown model {name!r}; choices: {sorted(MODEL_REGISTRY)}"
        ) from exc


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
