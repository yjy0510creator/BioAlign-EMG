from __future__ import annotations

import numpy as np

from bioalign_emg.data import make_windows, training_normalization
from bioalign_emg.metrics import trial_level_metrics


def test_window_shape() -> None:
    signal = np.random.default_rng(1).normal(size=(200, 8)).astype(np.float32)
    windows = make_windows(signal)
    assert windows.shape[1:] == (8, 50)


def test_training_normalization_shape() -> None:
    X = np.random.default_rng(2).normal(size=(20, 8, 50)).astype(np.float32)
    mask = np.zeros(20, dtype=bool)
    mask[:10] = True
    mean, std = training_normalization(X, mask)
    assert mean.shape == (1, 8, 1)
    assert std.shape == (1, 8, 1)


def test_trial_aggregation() -> None:
    y = np.repeat([0, 1], 3)
    probability = np.zeros((6, 7), dtype=np.float32)
    probability[:3, 0] = 1
    probability[3:, 1] = 1
    trial_ids = np.repeat(["a", "b"], 3)
    metrics = trial_level_metrics(y, probability, trial_ids)
    assert metrics["accuracy"] == 1.0
