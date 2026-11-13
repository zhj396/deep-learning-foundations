"""Shared fixtures: sys.path bootstrap, fake SentencePiece, tiny model/cfg.

The train/, data_prep/ and inference/ directories are flat script dirs (no
packages), so tests import them via sys.path.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "train", ROOT / "data_prep", ROOT / "inference"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


class FakeSP:
    """Whitespace-token processor duck-typing SentencePieceProcessor.

    Special ids are configurable so tests can prove nothing depends on
    the 0/1/2/3 defaults.
    """

    def __init__(self, words=None, pad_id=0, unk_id=1, bos_id=2, eos_id=3):
        base = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
        self._words = list(words or ["a", "b", "c", "d", "e", "ola", "mundo", "hello", "world"])
        self._vocab = base + self._words
        self._tok2id = {t: i for i, t in enumerate(self._vocab)}
        self._pad, self._unk, self._bos, self._eos = pad_id, unk_id, bos_id, eos_id

    def pad_id(self):
        return self._pad

    def unk_id(self):
        return self._unk

    def bos_id(self):
        return self._bos

    def eos_id(self):
        return self._eos

    def encode(self, text, out_type=int):
        return [self._tok2id.get(w, self._unk) for w in text.split()]

    def decode(self, ids):
        specials = {self._pad, self._unk, self._bos, self._eos}
        toks = [self._vocab[i] for i in ids if 0 <= i < len(self._vocab) and i not in specials]
        return " ".join(toks)

    def get_piece_size(self):
        return len(self._vocab)


@pytest.fixture
def fake_sp():
    return FakeSP()


@pytest.fixture
def tiny_cfg():
    """Small but valid config for fast CPU tests."""
    return SimpleNamespace(
        d_model=16,
        num_layers=2,
        num_heads=2,
        d_ff=32,
        vocab_size=32,
        dropout=0.0,
        max_len=64,
        tie_weights=True,
        pad_idx=0,
        unk_idx=1,
        bos_idx=2,
        eos_idx=3,
        label_smoothing=0.1,
        lr_min_scale=0.1,
        patience=3,
        seed=0,
    )


@pytest.fixture
def tiny_model(tiny_cfg):
    from model import Transformer

    torch.manual_seed(0)
    m = Transformer(
        src_vocab=tiny_cfg.vocab_size,
        tgt_vocab=tiny_cfg.vocab_size,
        d_model=tiny_cfg.d_model,
        num_layers=tiny_cfg.num_layers,
        num_heads=tiny_cfg.num_heads,
        d_ff=tiny_cfg.d_ff,
        dropout=tiny_cfg.dropout,
        max_len=tiny_cfg.max_len,
        tie_weights=tiny_cfg.tie_weights,
    )
    m.eval()
    return m
