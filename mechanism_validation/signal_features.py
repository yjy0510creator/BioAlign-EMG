from __future__ import annotations

import numpy as np


def time_domain_features(x: np.ndarray, zc_threshold: float = 1e-3) -> dict[str, np.ndarray]:
    """Compute standard per-channel sEMG descriptors for [N,C,T]."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f'Expected [N,C,T], got {x.shape}')
    diff = np.diff(x, axis=-1)
    mav = np.mean(np.abs(x), axis=-1)
    rms = np.sqrt(np.mean(x ** 2, axis=-1))
    wl = np.sum(np.abs(diff), axis=-1)
    signs = np.signbit(x)
    zc = np.sum((signs[..., 1:] != signs[..., :-1]) & (np.abs(diff) > zc_threshold), axis=-1)
    var = np.var(x, axis=-1)
    return {'MAV': mav, 'RMS': rms, 'WL': wl, 'ZC': zc.astype(float), 'VAR': var}


def flatten_features(x: np.ndarray) -> np.ndarray:
    feats = time_domain_features(x)
    return np.concatenate([feats[k] for k in ('MAV', 'RMS', 'WL', 'ZC', 'VAR')], axis=1)


def circular_xcorr_shift(reference: np.ndarray, observed: np.ndarray) -> int:
    """Integer shift maximizing cosine similarity for [C] spatial patterns."""
    reference = np.asarray(reference, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if reference.shape != observed.shape or reference.ndim != 1:
        raise ValueError('reference and observed must be same-shape [C] vectors')
    ref = reference - reference.mean()
    obs = observed - observed.mean()
    denom = (np.linalg.norm(ref) * np.linalg.norm(obs)) + 1e-12
    scores = [float(np.dot(ref, np.roll(obs, -k)) / denom) for k in range(len(ref))]
    return int(np.argmax(scores))
