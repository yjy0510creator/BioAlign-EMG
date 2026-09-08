import torch
from mechanism_validation.models import BioAlignV2


def test_all_variants_forward_backward():
    x = torch.randn(4, 8, 50)
    y = torch.tensor([0, 1, 2, 3])
    for variant in ['none', 'uniform', 'uniform_safe', 'random', 'hard_ste', 'soft', 'continuous']:
        model = BioAlignV2(canonicalizer=variant)
        logits, aux = model(x, return_aux=True)
        assert logits.shape == (4, 7)
        assert aux['shift_logits'].shape == (4, 8)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
