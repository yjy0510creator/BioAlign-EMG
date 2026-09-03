from __future__ import annotations

import torch

from bioalign_emg.models import build_model, count_parameters


def test_parameter_counts_and_shapes() -> None:
    expected = {
        "TCN_plain": 16663,
        "SE_RingAug": 17293,
        "CBAM_RingAug": 17308,
        "BioAlign_RingAug": 33549,
        "BioAlign_Compact": 31299,
    }
    x = torch.randn(2, 8, 50)
    for name, target in expected.items():
        model = build_model(name)
        assert count_parameters(model) == target
        assert model(x)["logits"].shape == (2, 7)
