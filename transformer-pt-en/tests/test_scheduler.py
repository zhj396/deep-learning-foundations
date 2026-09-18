"""Noam schedule: shape, closed form, plateau scale, state round-trip."""

import torch

from scheduler import NoamOpt


def _make(d_model=512, factor=1.0, warmup=100, lr_scale=1.0):
    param = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([param], lr=0.0)
    return NoamOpt(d_model, factor, warmup, opt, lr_scale=lr_scale)


def test_warmup_then_decay_shape():
    s = _make()
    rates = [s.rate(step=i) for i in range(1, 201)]
    warmup_part = rates[:100]
    decay_part = rates[100:]
    assert all(b > a for a, b in zip(warmup_part, warmup_part[1:]))  # strictly up
    assert all(b < a for a, b in zip(decay_part, decay_part[1:]))    # strictly down
    assert max(rates) == rates[99]  # peak at step == warmup


def test_rate_matches_closed_form():
    s = _make(d_model=512, factor=2.0, warmup=400)
    for step in (1, 10, 400, 4000):
        expected = 2.0 * (512 ** -0.5) * min(step ** -0.5, step * (400 ** -1.5))
        assert abs(s.rate(step=step) - expected) < 1e-12


def test_lr_scale_survives_steps():
    """Plateau decay must survive subsequent step() calls."""
    plain = _make()
    scaled = _make()
    for _ in range(5):
        plain.step()
        scaled.step()

    scaled.set_lr_scale(0.5)
    for _ in range(3):
        plain.step()
        scaled.step()

    assert abs(scaled.get_lr() - 0.5 * plain.get_lr()) < 1e-12
    assert abs(scaled.optimizer.param_groups[0]["lr"] - 0.5 * plain.get_lr()) < 1e-12


def test_state_dict_roundtrip():
    s = _make()
    for _ in range(7):
        s.step()
    s.set_lr_scale(0.25)
    state = s.state_dict()

    fresh = _make()
    fresh.load_state_dict(state)
    assert fresh.get_step() == s.get_step()
    assert abs(fresh.get_lr() - s.get_lr()) < 1e-12
    assert fresh.lr_scale == 0.25
