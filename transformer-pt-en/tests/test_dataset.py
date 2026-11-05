"""Dataset encoding/decoding and Collator padding/truncation."""

from conftest import FakeSP
from dataset import Collator, TranslationDataset


def _write_pair_files(tmp_path):
    src = tmp_path / "src.txt"
    tgt = tmp_path / "tgt.txt"
    src.write_text("ola mundo\na b c\n", encoding="utf-8")
    tgt.write_text("hello world\na b\n", encoding="utf-8")
    return src, tgt


def test_encode_wraps_bos_eos(tmp_path, fake_sp):
    src, tgt = _write_pair_files(tmp_path)
    ds = TranslationDataset(src, tgt, sp_processor=fake_sp)
    ids = ds.encode("ola mundo")
    assert ids[0] == fake_sp.bos_id()
    assert ids[-1] == fake_sp.eos_id()
    assert ids[1:-1] == fake_sp.encode("ola mundo", out_type=int)


def test_decode_filters_specials(tmp_path, fake_sp):
    src, tgt = _write_pair_files(tmp_path)
    ds = TranslationDataset(src, tgt, sp_processor=fake_sp)
    ids = (
        [fake_sp.bos_id()]
        + fake_sp.encode("hello world", out_type=int)
        + [fake_sp.eos_id(), fake_sp.pad_id()]
    )
    assert ds.decode(ids) == "hello world"


def test_getitem_and_len(tmp_path, fake_sp):
    src, tgt = _write_pair_files(tmp_path)
    ds = TranslationDataset(src, tgt, sp_processor=fake_sp)
    assert len(ds) == 2
    s, t = ds[0]
    assert s[0] == ds.bos_idx
    assert t[-1] == ds.eos_idx


def test_mismatched_file_lengths_raise(tmp_path, fake_sp):
    src = tmp_path / "src.txt"
    tgt = tmp_path / "tgt.txt"
    src.write_text("a b c\nd e f\n", encoding="utf-8")
    tgt.write_text("x y z\n", encoding="utf-8")
    try:
        TranslationDataset(src, tgt, sp_processor=fake_sp)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_collator_pads_to_batch_max():
    coll = Collator(pad_idx=0, eos_idx=3, max_len=None)
    batch = [([2, 5, 3], [2, 6, 3]), ([2, 7, 8, 9, 3], [2, 10, 3])]
    src_t, tgt_t = coll(batch)
    assert src_t.shape == (2, 5)
    assert tgt_t.shape == (2, 3)
    assert src_t[0].tolist() == [2, 5, 3, 0, 0]
    assert tgt_t[0].tolist() == [2, 6, 3]


def test_collator_truncates_keeping_eos():
    coll = Collator(pad_idx=0, eos_idx=3, max_len=3)
    batch = [([2, 5, 6, 7, 3], [2, 8, 3])]
    src_t, _ = coll(batch)
    assert src_t.shape[1] == 3
    assert src_t[0].tolist() == [2, 5, 3]


def test_nondefault_special_ids_respected(tmp_path):
    """Special ids must come from the processor, not globals."""
    sp = FakeSP(pad_id=10, unk_id=11, bos_id=12, eos_id=13)
    src, tgt = _write_pair_files(tmp_path)
    ds = TranslationDataset(src, tgt, sp_processor=sp)
    assert (ds.pad_idx, ds.unk_idx, ds.bos_idx, ds.eos_idx) == (10, 11, 12, 13)
    ids = ds.encode("a b")
    assert ids[0] == 12 and ids[-1] == 13
