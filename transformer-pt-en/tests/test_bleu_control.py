"""BLEU-driven controller logic without a process group."""

from types import SimpleNamespace

import torch

from bleu_control import (
    broadcast_bleu_state,
    default_bleu_state,
    should_stop,
    update_bleu_state,
)


def _cfg():
    return SimpleNamespace(label_smoothing=0.1, lr_min_scale=0.1, patience=3)


def test_improvement_resets_counters():
    cfg = _cfg()
    state = default_bleu_state(cfg)
    assert update_bleu_state(state, 5.0, cfg)  # first score always improves -1.0
    assert state["best"] == 5.0

    update_bleu_state(state, 4.0, cfg)  # miss
    assert state["patience"] == 1

    assert update_bleu_state(state, 6.0, cfg)  # improvement
    assert state["patience"] == 0
    assert state["no_improve"] == 0
    assert state["plateau"] == 0
    assert state["best"] == 6.0


def test_label_smoothing_halves_after_two_misses():
    cfg = _cfg()
    state = default_bleu_state(cfg)
    update_bleu_state(state, 5.0, cfg)

    update_bleu_state(state, 4.0, cfg)  # miss 1
    assert state["label_smoothing"] == 0.1

    update_bleu_state(state, 4.0, cfg)  # miss 2 -> halve
    assert abs(state["label_smoothing"] - 0.05) < 1e-9
    assert state["no_improve"] == 0  # counter reset after decay


def test_lr_scale_halves_with_floor():
    cfg = _cfg()
    state = default_bleu_state(cfg)
    update_bleu_state(state, 5.0, cfg)
    for _ in range(6):  # 1.0 -> 0.5 -> 0.25 -> 0.125 -> floor 0.1
        update_bleu_state(state, 1.0, cfg)
    assert abs(state["lr_scale"] - cfg.lr_min_scale) < 1e-9


def test_should_stop_at_patience():
    cfg = _cfg()
    state = default_bleu_state(cfg)
    update_bleu_state(state, 5.0, cfg)
    for i in range(cfg.patience):
        update_bleu_state(state, 0.0, cfg)
        if i < cfg.patience - 1:
            assert not should_stop(state, cfg)
    assert should_stop(state, cfg)


def test_broadcast_is_noop_without_process_group():
    cfg = _cfg()
    state = default_bleu_state(cfg)
    out = broadcast_bleu_state(state, torch.device("cpu"))
    assert out == state
    assert out is not state  # returns a copy
