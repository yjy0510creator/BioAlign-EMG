from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .metrics import aggregate_trial_probabilities, classification_metrics
from .signal_features import flatten_features


def _proba_with_all_classes(model, f_test: np.ndarray, all_classes: np.ndarray) -> np.ndarray:
    if hasattr(model, 'predict_proba'):
        p = model.predict_proba(f_test)
        out = np.zeros((len(f_test), len(all_classes)), dtype=float)
        classes = getattr(model, 'classes_', None)
        if classes is None and hasattr(model, 'named_steps'):
            last = list(model.named_steps.values())[-1]
            classes = getattr(last, 'classes_', all_classes)
        for j, cls in enumerate(np.asarray(classes)):
            col = int(np.flatnonzero(all_classes == cls)[0])
            out[:, col] = p[:, j]
        return out
    pred = model.predict(f_test)
    out = np.zeros((len(f_test), len(all_classes)), dtype=float)
    for i, cls in enumerate(pred):
        out[i, int(np.flatnonzero(all_classes == cls)[0])] = 1.0
    return out


def make_classical_models() -> Dict[str, object]:
    return {
        'TD-LDA': make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
        # probability=True is slower but allows the same trial-level probability
        # averaging protocol used by neural models.
        'TD-SVM-RBF': make_pipeline(StandardScaler(), SVC(C=10.0, gamma='scale', probability=True, random_state=42)),
    }


def fit_evaluate_classical(
    data_or_x_train,
    train_idx_or_y_train=None,
    test_idx_or_x_test=None,
    y_test: Optional[np.ndarray] = None,
) -> Dict[str, dict]:
    """Fit time-domain LDA/SVM baselines and report window and trial metrics.

    Preferred V2.1 API:
        fit_evaluate_classical(data, train_idx, test_idx)

    Legacy API is still accepted but only returns window-level metrics because no
    trial ids are available:
        fit_evaluate_classical(x_train, y_train, x_test, y_test)
    """
    if hasattr(data_or_x_train, 'x') and y_test is None:
        data = data_or_x_train
        train_idx = np.asarray(train_idx_or_y_train, dtype=int)
        test_idx = np.asarray(test_idx_or_x_test, dtype=int)
        x_train, y_train = data.x[train_idx], data.gesture[train_idx]
        x_test, yt = data.x[test_idx], data.gesture[test_idx]
        trial = data.trial[test_idx]
    else:
        x_train = np.asarray(data_or_x_train)
        y_train = np.asarray(train_idx_or_y_train)
        x_test = np.asarray(test_idx_or_x_test)
        yt = np.asarray(y_test)
        trial = None

    f_train = flatten_features(x_train)
    f_test = flatten_features(x_test)
    all_classes = np.arange(int(max(np.max(y_train), np.max(yt))) + 1)
    results: Dict[str, dict] = {}
    for name, model in make_classical_models().items():
        model.fit(f_train, y_train)
        pred = model.predict(f_test)
        row = {f'window_{k}': v for k, v in classification_metrics(yt, pred).items()}
        if trial is None:
            row.update(classification_metrics(yt, pred))
        else:
            probs = _proba_with_all_classes(model, f_test, all_classes)
            tprob, ty, _ = aggregate_trial_probabilities(probs, yt, trial)
            tpred = tprob.argmax(axis=1)
            row.update({f'trial_{k}': v for k, v in classification_metrics(ty, tpred).items()})
            # Also expose generic names for the main result table.
            row.update(classification_metrics(ty, tpred))
        results[name] = row
    return results
