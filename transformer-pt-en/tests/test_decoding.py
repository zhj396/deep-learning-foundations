"""Greedy and beam decoding invariants (tiny random model, CPU)."""

import torch
import torch.nn as nn

from decoding import beam_decode_batch, greedy_decode_batch

BOS, EOS, PAD, VOCAB = 2, 3, 0, 32


def _src(batch=2, length=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(4, VOCAB, (batch, length), generator=g)


def test_greedy_shape_and_bos(tiny_model):
    out = greedy_decode_batch(
        tiny_model, _src(), max_len=7, bos_idx=BOS, eos_idx=EOS, pad_idx=PAD
    )
    assert out.shape[0] == 2
    assert out.shape[1] <= 7
    assert (out[:, 0] == BOS).all()
    assert (out >= 0).all() and (out < VOCAB).all()


def test_greedy_deterministic(tiny_model):
    src = _src(seed=1)
    a = greedy_decode_batch(tiny_model, src, max_len=6, bos_idx=BOS, eos_idx=EOS, pad_idx=PAD)
    b = greedy_decode_batch(tiny_model, src, max_len=6, bos_idx=BOS, eos_idx=EOS, pad_idx=PAD)
    assert torch.equal(a, b)


def test_forced_eos_terminates_immediately(tiny_model):
    """With an output head that always picks EOS, decoding stops after 1 step."""
    eos_out = nn.Linear(tiny_model.out.in_features, VOCAB)
    nn.init.zeros_(eos_out.weight)
    nn.init.zeros_(eos_out.bias)
    with torch.no_grad():
        eos_out.bias[EOS] = 10.0
    tiny_model.out = eos_out

    out = greedy_decode_batch(
        tiny_model, _src(seed=2), max_len=8, bos_idx=BOS, eos_idx=EOS, pad_idx=PAD
    )
    assert out.shape[1] == 2
    assert (out[:, 1] == EOS).all()


def test_beam_size_one_matches_greedy(tiny_model):
    src = _src(seed=3)
    g = greedy_decode_batch(tiny_model, src, max_len=6, bos_idx=BOS, eos_idx=EOS, pad_idx=PAD)
    b = beam_decode_batch(
        tiny_model, src, max_len=6, bos_idx=BOS, eos_idx=EOS, pad_idx=PAD, beam_size=1
    )

    def prefix_before_special(row):
        out = []
        for t in row.tolist():
            if t in (EOS, PAD):
                break
            out.append(t)
        return out

    for gr, br in zip(g, b):
        assert prefix_before_special(gr) == prefix_before_special(br)


def test_beam_output_valid(tiny_model):
    out = beam_decode_batch(
        tiny_model, _src(seed=4), max_len=7,
        bos_idx=BOS, eos_idx=EOS, pad_idx=PAD,
        beam_size=3, length_penalty=0.6,
    )
    assert out.shape[0] == 2
    assert out.shape[1] <= 7
    assert (out[:, 0] == BOS).all()
    assert (out >= 0).all() and (out < VOCAB).all()
