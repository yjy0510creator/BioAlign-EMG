"""Mechanism-validation extension for BioAlign-EMG.

This package separates three claims that must not be conflated:
1) classification robustness under electrode shift;
2) recovery of a cyclic sensor correspondence;
3) preservation of gesture-discriminative structure after canonicalization.
"""

from .cyclic_ops import (
    SoftCyclicCanonicalizer,
    StraightThroughHardCanonicalizer,
    ContinuousFourierCanonicalizer,
    UniformCyclicMixer,
    RandomCyclicMixer,
    circular_mean_from_probs,
    continuous_circular_shift,
)
from .models import BioAlignV2, TinyTCN, CircularConvTCN

__all__ = [
    'SoftCyclicCanonicalizer',
    'StraightThroughHardCanonicalizer',
    'ContinuousFourierCanonicalizer',
    'UniformCyclicMixer',
    'RandomCyclicMixer',
    'circular_mean_from_probs',
    'continuous_circular_shift',
    'BioAlignV2',
    'TinyTCN',
    'CircularConvTCN',
]
