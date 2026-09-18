"""Label-smoothing loss and accuracy tests."""

import torch
import torch.nn.functional as F

from loss import LabelSmoothingLoss, accuracy

V = 11
PAD = 0


def _random_case(n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, V, generator=g)
    target = torch.randint(1, V, (n,), generator=g)
    return logits, target


def test_zero_smoothing_matches_cross_entropy():
    logits, target = _random_case()
    criterion = LabelSmoothingLoss(smoothing=0.0, vocab_size=V, pad_idx=PAD)
    expected = F.cross_entropy(logits, target)
    assert torch.allclose(criterion(logits, target), expected, atol=1e-5)


def test_all_pad_batch_is_zero_and_finite():
    logits = torch.randn(4, V)
    target = torch.zeros(4, dtype=torch.long)
    criterion = LabelSmoothingLoss(smoothing=0.1, vocab_size=V, pad_idx=PAD)
    loss = criterion(logits, target)
    assert loss.item() == 0.0
    assert torch.isfinite(loss)


def test_pad_positions_do_not_affect_loss():
    logits, target = _random_case(n=5, seed=1)
    criterion = LabelSmoothingLoss(smoothing=0.1, vocab_size=V, pad_idx=PAD)
    base = criterion(logits, target).item()

    # turn the last position into a PAD and corrupt its logits
    target2 = target.clone()
    target2[-1] = PAD
    logits2 = logits.clone()
    logits2[-1] += 100.0
    masked = criterion(logits2, target2)

    expected = criterion(logits2[:4], target2[:4])
    assert torch.allclose(masked, expected, atol=1e-6)
    assert not torch.isclose(masked, torch.tensor(base))


def test_accuracy_counts_only_non_pad():
    # logits engineered so argmax is known: class index = position
    logits = torch.full((1, 4, V), -10.0)
    target = torch.tensor([[3, 4, 5, PAD]])
    for t in range(4):
        logits[0, t, target[0, t] if target[0, t] != PAD else 1] = 10.0
    # predictions: [3, 4, 5, 1]; target non-pad: [3, 4, 5] -> all correct
    assert accuracy(logits, target, pad_idx=PAD).item() == 1.0

    # break one prediction
    logits[0, 1] = -10.0
    logits[0, 1, 9] = 10.0
    assert abs(accuracy(logits, target, pad_idx=PAD).item() - 2.0 / 3.0) < 1e-6


def test_accuracy_all_pad_is_zero():
    logits = torch.randn(1, 3, V)
    target = torch.zeros(1, 3, dtype=torch.long)
    assert accuracy(logits, target, pad_idx=PAD).item() == 0.0
