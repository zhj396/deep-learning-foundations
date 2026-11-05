"""Config invariants for the probe/report knobs (config.py has no other suite).

Everything here is checked on the dataclass itself: the CLI is generated
from its fields, so one test of the fields and one of the generated flags
covers both surfaces.
"""

import pytest

from config import Config


def test_probe_defaults_keep_selection_off_test():
    cfg = Config()
    assert cfg.bleu_split == "valid", "selection must not read the reported split"
    assert cfg.final_test is True, "the reported number needs one full-split pass"
    assert cfg.bleu_min_delta > 0.0, "a strict > would let BLEU noise defer plateaus"
    assert cfg.bleu_max_len <= cfg.max_len, "the probe cap cannot exceed the model's range"


def test_unknown_split_is_rejected_at_construction():
    with pytest.raises(ValueError, match="bleu_split"):
        Config(bleu_split="train")


def test_both_supported_splits_construct():
    for split in ("valid", "test"):
        assert Config(bleu_split=split).bleu_split == split


def test_json_roundtrip_keeps_the_probe_knobs(tmp_path):
    cfg = Config(bleu_split="test", bleu_min_delta=0.25, final_test=False)
    path = tmp_path / "config.json"
    cfg.save_json(str(path))

    back = Config.load(str(path))
    assert back.bleu_split == "test"
    assert back.bleu_min_delta == 0.25
    assert back.final_test is False


def test_cli_exposes_the_probe_knobs():
    cfg = Config.from_argparse(["--bleu_split", "test", "--no-final_test"])
    assert cfg.bleu_split == "test"
    assert cfg.final_test is False
