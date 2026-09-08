from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, silhouette_score


def classification_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
    }


def aggregate_trial_probabilities(probs: np.ndarray, y: np.ndarray, trial_ids: Sequence) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs = np.asarray(probs)
    y = np.asarray(y)
    trial_ids = np.asarray(trial_ids)
    unique = np.unique(trial_ids)
    p_out, y_out = [], []
    for t in unique:
        m = trial_ids == t
        labels = np.unique(y[m])
        if len(labels) != 1:
            raise ValueError(f'Trial {t} has multiple labels: {labels}')
        p_out.append(probs[m].mean(axis=0))
        y_out.append(labels[0])
    return np.asarray(p_out), np.asarray(y_out), unique


def circular_signed_error(pred_deg: np.ndarray, true_deg: np.ndarray, period: float = 360.0) -> np.ndarray:
    return (np.asarray(pred_deg) - np.asarray(true_deg) + period / 2) % period - period / 2


def circular_mae_deg(pred_deg, true_deg, period: float = 360.0) -> float:
    return float(np.mean(np.abs(circular_signed_error(pred_deg, true_deg, period))))


def circular_rmse_deg(pred_deg, true_deg, period: float = 360.0) -> float:
    e = circular_signed_error(pred_deg, true_deg, period)
    return float(np.sqrt(np.mean(e ** 2)))


def circular_correlation_deg(alpha_deg, beta_deg) -> float:
    """Jammalamadaka-Sengupta circular-circular correlation."""
    a = np.deg2rad(np.asarray(alpha_deg, dtype=float))
    b = np.deg2rad(np.asarray(beta_deg, dtype=float))
    a0 = np.arctan2(np.sin(a).mean(), np.cos(a).mean())
    b0 = np.arctan2(np.sin(b).mean(), np.cos(b).mean())
    num = np.sum(np.sin(a - a0) * np.sin(b - b0))
    den = np.sqrt(np.sum(np.sin(a - a0) ** 2) * np.sum(np.sin(b - b0) ** 2)) + 1e-12
    return float(num / den)


def fisher_ratio(emb: np.ndarray, labels: np.ndarray) -> float:
    emb = np.asarray(emb, dtype=float)
    labels = np.asarray(labels)
    grand = emb.mean(axis=0)
    between = 0.0
    within = 0.0
    for c in np.unique(labels):
        x = emb[labels == c]
        mu = x.mean(axis=0)
        between += len(x) * np.sum((mu - grand) ** 2)
        within += np.sum((x - mu) ** 2)
    return float(between / (within + 1e-12))


def safe_silhouette(emb: np.ndarray, labels: np.ndarray, max_samples: int = 5000, seed: int = 0) -> float:
    emb = np.asarray(emb)
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float('nan')
    if len(emb) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(emb), max_samples, replace=False)
        emb, labels = emb[idx], labels[idx]
    return float(silhouette_score(emb, labels))
