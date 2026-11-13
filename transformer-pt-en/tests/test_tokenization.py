"""Translator mirrors train/dataset.py encoding rules (FakeSP, no spm needed)."""

from conftest import FakeSP
from tokenization import Translator


def test_encode_wraps_bos_eos():
    sp = FakeSP()
    tr = Translator(processor=sp)
    ids = tr.encode("ola mundo")
    assert ids[0] == tr.bos_idx
    assert ids[-1] == tr.eos_idx
    assert ids[1:-1] == sp.encode("ola mundo", out_type=int)


def test_decode_cuts_at_eos_and_strips_specials():
    sp = FakeSP()
    tr = Translator(processor=sp)
    hello = sp.encode("hello world", out_type=int)
    # BOS, "hello", EOS, then junk that must be ignored, plus trailing PAD
    ids = [tr.bos_idx, hello[0], tr.eos_idx, hello[1], tr.pad_idx]
    assert tr.decode(ids) == "hello"


def test_decode_full_roundtrip():
    sp = FakeSP()
    tr = Translator(processor=sp)
    ids = tr.encode("hello world")
    assert tr.decode(ids) == "hello world"


def test_nondefault_ids_from_processor():
    """Special ids come from the processor, never hardcoded."""
    sp = FakeSP(pad_id=10, unk_id=11, bos_id=12, eos_id=13)
    tr = Translator(processor=sp)
    assert (tr.pad_idx, tr.unk_idx, tr.bos_idx, tr.eos_idx) == (10, 11, 12, 13)

    ids = tr.encode("a")
    assert ids[0] == 12 and ids[-1] == 13
    assert tr.decode_params() == {"bos_idx": 12, "eos_idx": 13, "pad_idx": 10}


def test_ctor_requires_source():
    try:
        Translator()
        assert False, "expected ValueError"
    except ValueError:
        pass
