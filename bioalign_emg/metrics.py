from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from .constants import N_CLASSES


def evaluate_group(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                pred,
                labels=list(range(N_CLASSES)),
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
    }


@torch.inference_mode()
def predict_probabilities(
    model: torch.nn.Module,
    X: np.ndarray,
    *,
    batch_size: int = 512,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    model.eval().to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X.astype(np.float32))),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    probabilities = []
    predictions = []
    for (xb,) in loader:
        output = model(xb.to(device))
        prob = torch.softmax(output["logits"], dim=1)
        probabilities.append(prob.cpu().numpy())
        predictions.append(prob.argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions), np.concatenate(probabilities)


def trial_level_predictions(
    y: np.ndarray,
    probability: np.ndarray,
    trial_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    true_trial = []
    pred_trial = []
    ids = []
    for trial_id in np.unique(trial_ids):
        mask = trial_ids == trial_id
        labels = np.unique(y[mask])
        if len(labels) != 1:
            raise ValueError(f"Trial {trial_id} has inconsistent labels: {labels}")
        mean_probability = probability[mask].mean(axis=0)
        true_trial.append(int(labels[0]))
        pred_trial.append(int(np.argmax(mean_probability)))
        ids.append(str(trial_id))
    return (
        np.asarray(true_trial, dtype=np.int64),
        np.asarray(pred_trial, dtype=np.int64),
        np.asarray(ids, dtype=str),
    )


def trial_level_metrics(
    y: np.ndarray,
    probability: np.ndarray,
    trial_ids: np.ndarray,
) -> dict[str, float]:
    true_trial, pred_trial, _ = trial_level_predictions(y, probability, trial_ids)
    return evaluate_group(true_trial, pred_trial)
