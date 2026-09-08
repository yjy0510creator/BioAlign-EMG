import torch

from mechanism_validation.cyclic_ops import (
    SoftCyclicCanonicalizer,
    StraightThroughHardCanonicalizer,
    UniformCyclicMixer,
    continuous_circular_shift,
)


def test_continuous_integer_shift_matches_roll():
    x = torch.randn(3, 8, 4, 10)
    for k in range(-3, 4):
        y = continuous_circular_shift(x, torch.full((3,), float(k)))
        assert torch.allclose(y, torch.roll(x, shifts=k, dims=1), atol=1e-5, rtol=1e-5)


def test_soft_one_hot_performs_inverse_roll():
    x = torch.randn(2, 8, 3, 5)
    logits = torch.full((2, 8), -20.0)
    logits[:, 3] = 20.0
    y, _ = SoftCyclicCanonicalizer(confidence_residual=False)(x, logits)
    assert torch.allclose(y, torch.roll(x, shifts=-3, dims=1), atol=1e-5)


def test_uniform_control_collapses_channel_identity():
    x = torch.randn(2, 8, 3, 5)
    y, _ = UniformCyclicMixer(safe_residual=False)(x)
    for c in range(1, 8):
        assert torch.allclose(y[:, 0], y[:, c], atol=1e-6)


def test_uniform_safe_residual_preserves_input():
    x = torch.randn(2, 8, 3, 5)
    y, _ = UniformCyclicMixer(safe_residual=True)(x)
    assert torch.allclose(y, x)


def test_hard_ste_passes_gradient_to_logits():
    x = torch.randn(4, 8, 3, 5)
    logits = torch.randn(4, 8, requires_grad=True)
    y, _ = StraightThroughHardCanonicalizer()(x, logits)
    loss = (y ** 2).mean()
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
