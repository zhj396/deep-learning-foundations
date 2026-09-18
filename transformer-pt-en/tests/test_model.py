"""Model-level tests: masks, positional encoding, causality, weight tying."""

import math

import torch
import torch.nn as nn

from layers import MultiHeadAttention, PositionalEncoding


def test_positional_encoding_values():
    pe = PositionalEncoding(d_model=4, max_len=3)
    table = pe.pe.squeeze(0)  # (3, 4)
    # d_model=4 -> frequencies [1.0, 0.01]; column layout (sin f0, cos f0, sin f1, cos f1)
    assert abs(table[0, 0].item()) < 1e-6                   # sin(0 * 1.0)
    assert abs(table[0, 1].item() - 1.0) < 1e-6             # cos(0 * 1.0)
    assert abs(table[1, 0].item() - math.sin(1.0)) < 1e-6   # sin(1 * 1.0)
    assert abs(table[1, 1].item() - math.cos(1.0)) < 1e-6   # cos(1 * 1.0)
    assert abs(table[1, 2].item() - math.sin(0.01)) < 1e-6  # sin(1 * 0.01)
    assert abs(table[1, 3].item() - math.cos(0.01)) < 1e-6  # cos(1 * 0.01)


def test_positional_encoding_odd_dim():
    pe = PositionalEncoding(d_model=5, max_len=7)
    x = torch.zeros(2, 7, 5)
    assert pe(x).shape == (2, 7, 5)


def test_src_mask_marks_only_pads(tiny_model):
    src = torch.tensor([[5, 6, 0, 0], [7, 0, 0, 0]])
    mask = tiny_model.make_src_mask(src, pad_idx=0)
    assert mask.tolist() == [
        [False, False, True, True],
        [False, True, True, True],
    ]


def test_tgt_mask_is_causal_and_pads_columns(tiny_model):
    tgt = torch.tensor([[2, 5, 6, 0]])  # BOS, two tokens, PAD
    mask = tiny_model.make_tgt_mask(tgt, pad_idx=0)  # (1, 4, 4)
    m = mask[0]
    # strictly upper triangle masked (look-ahead)
    for i in range(4):
        for j in range(4):
            expected = (j > i) or (tgt[0, j] == 0)
            assert bool(m[i, j]) == expected, (i, j)


def test_causality_end_to_end(tiny_model):
    """Changing a future target token must not change earlier logits."""
    src = torch.randint(4, 30, (1, 5))
    tgt = torch.randint(4, 30, (1, 6))
    logits1 = tiny_model(src, tgt)

    tgt2 = tgt.clone()
    tgt2[0, 5] = (tgt2[0, 5] + 1) % 30
    logits2 = tiny_model(src, tgt2)

    # positions 0..4 cannot see position 5
    assert torch.allclose(logits1[0, :5], logits2[0, :5], atol=1e-6)
    # position 5 itself generally does change
    assert not torch.allclose(logits1[0, 5], logits2[0, 5], atol=1e-6)


def test_final_layernorms_present(tiny_model):
    assert isinstance(tiny_model.encoder.norm, nn.LayerNorm)
    assert isinstance(tiny_model.decoder.norm, nn.LayerNorm)


def test_tied_weights(tiny_model):
    assert tiny_model.out.weight is tiny_model.decoder.embed.weight


def test_forward_shapes(tiny_model):
    src = torch.randint(4, 30, (3, 7))
    tgt = torch.randint(4, 30, (3, 5))
    assert tiny_model(src, tgt).shape == (3, 5, tiny_model.out.weight.size(0))


def test_mha_pad_mask_zeroes_attention():
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=8, num_heads=1, dropout=0.0).eval()
    q = k = v = torch.randn(1, 3, 8)
    mask = torch.tensor([[False, False, True]])  # last key is PAD

    _, attn = mha(q, k, v, mask=mask)  # attn: (1, 1, 3, 3)
    assert (attn[0, 0, :, 2] < 1e-6).all()
    # rows still sum to 1 over unmasked keys
    assert torch.allclose(attn.sum(dim=-1), torch.ones_like(attn.sum(dim=-1)), atol=1e-5)
