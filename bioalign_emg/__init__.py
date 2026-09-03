"""BioAlign-EMG reproducibility package."""

from .constants import (
    FS,
    GESTURES,
    N_CHANNELS,
    N_CLASSES,
    WINDOW_LEN,
    WINDOW_STEP,
)
from .models import (
    BioAlignEMG,
    BioAlignEMGCompact,
    BioAlignNoCircularAlignment,
    CBAMTCN,
    PlainTCN,
    SETCN,
)

__all__ = [
    "FS",
    "GESTURES",
    "N_CHANNELS",
    "N_CLASSES",
    "WINDOW_LEN",
    "WINDOW_STEP",
    "BioAlignEMG",
    "BioAlignEMGCompact",
    "BioAlignNoCircularAlignment",
    "PlainTCN",
    "SETCN",
    "CBAMTCN",
]
