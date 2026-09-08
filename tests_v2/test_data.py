import numpy as np
from mechanism_validation.data import StandardWindows, protocol_masks, normalize_from_train


def test_protocol_and_training_only_normalization():
    n = 30
    x = np.random.default_rng(0).normal(size=(n, 8, 50)).astype('float32')
    p = np.array(['p0'] * 9 + ['p1'] * 21)
    r = np.array(['r0', 'r1', 'r2'] * 10)
    d = StandardWindows(x, np.arange(n) % 7, p, r, np.arange(n), np.repeat('h0', n)).validate()
    m = protocol_masks(d)
    assert m['train'].sum() == 6
    assert m['ideal'].sum() == 3
    assert m['shift'].sum() == 21
    dn, mean, std = normalize_from_train(d, m['train'])
    assert np.allclose(dn.x[m['train']].mean(axis=(0, 2)), 0, atol=1e-5)
