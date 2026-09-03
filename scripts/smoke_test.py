from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bioalign_emg.augmentation import ring_augment
from bioalign_emg.metrics import trial_level_metrics
from bioalign_emg.models import build_model, count_parameters


def main() -> None:
    expected = {
        "TCN_plain": 16663,
        "TCN_RingAug": 16663,
        "SE_RingAug": 17293,
        "CBAM_RingAug": 17308,
        "BioAlign_RingAug": 33549,
        "BioAlign_NoAlignment": 33549,
        "BioAlign_Compact": 31299,
    }
    x = torch.randn(4, 8, 50)
    y = torch.tensor([0, 1, 2, 3])
    print("Model checks")
    for name, parameter_target in expected.items():
        model = build_model(name)
        output = model(x)
        assert output["logits"].shape == (4, 7), (name, output["logits"].shape)
        parameters = count_parameters(model)
        assert parameters == parameter_target, (name, parameters, parameter_target)
        print(f"  PASS {name:28s} params={parameters:,}")

    augmented, labels = ring_augment(x)
    assert augmented.shape == x.shape
    assert labels.shape == (4,)
    assert labels.min() >= 0 and labels.max() < 8
    print("  PASS RingAug shape and label range")

    model = build_model("BioAlign_RingAug")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x)["logits"], y)
    loss.backward()
    optimizer.step()
    assert np.isfinite(float(loss.item()))
    print(f"  PASS synthetic optimization step, loss={loss.item():.4f}")

    probability = np.full((12, 7), 1 / 7, dtype=np.float32)
    probability[np.arange(12), np.repeat([0, 1, 2], 4)] = 0.8
    probability /= probability.sum(axis=1, keepdims=True)
    trial_ids = np.repeat(["a", "b", "c"], 4)
    labels_true = np.repeat([0, 1, 2], 4)
    metrics = trial_level_metrics(labels_true, probability, trial_ids)
    assert metrics["macro_f1"] > 0
    print("  PASS trial-level probability aggregation")
    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
