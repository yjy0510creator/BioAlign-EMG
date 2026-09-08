from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch

from .cyclic_ops import continuous_circular_shift
from .signal_features import time_domain_features


def apply_integer_inverse(x: np.ndarray, shift_bins: np.ndarray) -> np.ndarray:
    x=np.asarray(x); shift_bins=np.asarray(shift_bins).astype(int)
    return np.stack([np.roll(a, -int(k), axis=0) for a,k in zip(x,shift_bins)])


def apply_continuous_inverse(x: np.ndarray, shift_channels: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return continuous_circular_shift(torch.as_tensor(x,dtype=torch.float32), -torch.as_tensor(shift_channels,dtype=torch.float32)).cpu().numpy()


def angle_to_integer_bins(angle_deg: np.ndarray, n_channels: int=8) -> np.ndarray:
    return np.rint(np.asarray(angle_deg) / (360.0/n_channels)).astype(int) % n_channels


def angle_to_continuous_channels(angle_deg: np.ndarray, n_channels: int=8) -> np.ndarray:
    return (np.asarray(angle_deg) / 360.0 * n_channels) % n_channels


@dataclass
class XCorrTemplateCorrector:
    templates: Dict[int, np.ndarray]

    @classmethod
    def fit(cls, x: np.ndarray, gesture: np.ndarray) -> 'XCorrTemplateCorrector':
        rms=time_domain_features(x)['RMS']
        templates={int(g): rms[gesture==g].mean(axis=0) for g in np.unique(gesture)}
        return cls(templates)

    def estimate(self, x: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
        rms=time_domain_features(x)['RMS']
        shifts=[]; pseudo_g=[]
        for pattern in rms:
            best=(-np.inf,0,0)
            p0=pattern-pattern.mean()
            for g,t in self.templates.items():
                t0=t-t.mean(); denom=np.linalg.norm(p0)*np.linalg.norm(t0)+1e-12
                for k in range(len(pattern)):
                    score=float(np.dot(np.roll(p0,-k),t0)/denom)
                    if score>best[0]: best=(score,k,g)
            shifts.append(best[1]); pseudo_g.append(best[2])
        return np.asarray(shifts),np.asarray(pseudo_g)

    def correct(self,x:np.ndarray)->Tuple[np.ndarray,np.ndarray]:
        shifts,_=self.estimate(x)
        return apply_integer_inverse(x,shifts),shifts
