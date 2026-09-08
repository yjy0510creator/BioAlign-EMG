from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import (
    circular_correlation_deg,
    circular_mae_deg,
    circular_rmse_deg,
    fisher_ratio,
    safe_silhouette,
)
from .signal_features import time_domain_features


def synthetic_roll_recovery(expected_shift_channels: np.ndarray, true_shift_channels: np.ndarray, n_channels: int = 8) -> dict:
    pred_deg = np.asarray(expected_shift_channels) * 360.0 / n_channels
    true_deg = np.asarray(true_shift_channels) * 360.0 / n_channels
    pred_bin = np.rint(expected_shift_channels).astype(int) % n_channels
    true_bin = np.rint(true_shift_channels).astype(int) % n_channels
    return {
        'integer_shift_accuracy': float(np.mean(pred_bin == true_bin)),
        'circular_mae_deg': circular_mae_deg(pred_deg, true_deg),
        'circular_rmse_deg': circular_rmse_deg(pred_deg, true_deg),
        'circular_correlation': circular_correlation_deg(pred_deg, true_deg),
    }


def real_angle_agreement(expected_shift_channels: np.ndarray, measured_angle_deg: np.ndarray, n_channels: int = 8) -> dict:
    pred_deg = np.asarray(expected_shift_channels) * 360.0 / n_channels
    true_deg = np.asarray(measured_angle_deg)
    mask = np.isfinite(pred_deg) & np.isfinite(true_deg)
    if mask.sum() == 0:
        return {'n': 0, 'circular_mae_deg': float('nan'), 'circular_rmse_deg': float('nan'), 'circular_correlation': float('nan')}
    return {
        'n': int(mask.sum()),
        'circular_mae_deg': circular_mae_deg(pred_deg[mask], true_deg[mask]),
        'circular_rmse_deg': circular_rmse_deg(pred_deg[mask], true_deg[mask]),
        'circular_correlation': circular_correlation_deg(pred_deg[mask], true_deg[mask]),
    }


def latent_disentanglement(latent: np.ndarray, gesture: np.ndarray, position: np.ndarray, trial: np.ndarray | None = None, seed: int = 0) -> dict:
    """Decode gesture and position from latent outputs with a trial-level split.

    V2.0 split alternating windows, which can leak overlapping windows from the
    same trial into train and test.  V2.1 uses GroupShuffleSplit by trial when
    trial ids are provided.
    """
    latent = np.asarray(latent)
    gesture = np.asarray(gesture)
    position = np.asarray(position).astype(str)
    if trial is None:
        groups = np.arange(len(latent))
    else:
        groups = np.asarray(trial).astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=seed)
    train, test = next(splitter.split(latent, gesture, groups=groups))
    out = {}
    for name, labels in [('gesture', gesture), ('position', position)]:
        if len(np.unique(labels[train])) < 2 or len(np.unique(labels[test])) < 2:
            out[f'{name}_decode_accuracy'] = float('nan')
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
        clf.fit(latent[train], labels[train])
        out[f'{name}_decode_accuracy'] = float(accuracy_score(labels[test], clf.predict(latent[test])))
    return out


def representation_report(embedding_pre: np.ndarray, embedding_post: np.ndarray, gesture: np.ndarray, position: np.ndarray) -> dict:
    return {
        'gesture_fisher_pre': fisher_ratio(embedding_pre, gesture),
        'gesture_fisher_post': fisher_ratio(embedding_post, gesture),
        'gesture_silhouette_pre': safe_silhouette(embedding_pre, gesture),
        'gesture_silhouette_post': safe_silhouette(embedding_post, gesture),
        'position_silhouette_pre': safe_silhouette(embedding_pre, position),
        'position_silhouette_post': safe_silhouette(embedding_post, position),
    }


def _cosine_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    return x @ x.T


def cross_position_similarity(features: np.ndarray, gesture: np.ndarray, position: np.ndarray) -> dict:
    """Measure whether same-gesture samples across positions become closer."""
    features = np.asarray(features, dtype=float)
    gesture = np.asarray(gesture)
    position = np.asarray(position).astype(str)
    sim = _cosine_matrix(features)
    same_gesture = gesture[:, None] == gesture[None, :]
    same_position = position[:, None] == position[None, :]
    eye = np.eye(len(features), dtype=bool)
    same_g_cross_pos = same_gesture & (~same_position) & (~eye)
    diff_g = (~same_gesture) & (~eye)
    return {
        'same_gesture_cross_position_cosine': float(np.nanmean(sim[same_g_cross_pos])) if same_g_cross_pos.any() else float('nan'),
        'different_gesture_cosine': float(np.nanmean(sim[diff_g])) if diff_g.any() else float('nan'),
        'gesture_position_margin': float(np.nanmean(sim[same_g_cross_pos]) - np.nanmean(sim[diff_g])) if same_g_cross_pos.any() and diff_g.any() else float('nan'),
    }


def pre_post_pattern_preservation(features_pre: np.ndarray, features_post: np.ndarray, gesture: np.ndarray, position: np.ndarray) -> dict:
    pre = cross_position_similarity(features_pre, gesture, position)
    post = cross_position_similarity(features_post, gesture, position)
    return {f'pre_{k}': v for k, v in pre.items()} | {f'post_{k}': v for k, v in post.items()} | {
        'delta_same_gesture_cross_position_cosine': post['same_gesture_cross_position_cosine'] - pre['same_gesture_cross_position_cosine'],
        'delta_gesture_position_margin': post['gesture_position_margin'] - pre['gesture_position_margin'],
    }


def gesture_channel_summary(x: np.ndarray, gesture: np.ndarray, position: np.ndarray) -> pd.DataFrame:
    feats = time_domain_features(x)
    rows = []
    for metric, values in feats.items():
        for g in np.unique(gesture):
            for p in np.unique(position):
                m = (gesture == g) & (position == p)
                if not m.any():
                    continue
                channel_mean = values[m].mean(axis=0)
                for ch, val in enumerate(channel_mean):
                    rows.append({'metric': metric, 'gesture': g, 'position': p, 'channel': ch, 'value': float(val)})
    return pd.DataFrame(rows)
