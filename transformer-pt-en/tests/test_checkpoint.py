"""Checkpoint save/load round-trip on CPU, without a process group."""

import os
from types import SimpleNamespace

import torch

from checkpoint import load_checkpoint, save_checkpoint
from scheduler import NoamOpt


def _make_sched(model, d_model=16, warmup=10):
    opt = torch.optim.Adam(model.parameters(), lr=0.0)
    return NoamOpt(d_model, 1.0, warmup, opt)


def test_roundtrip(tmp_path, tiny_cfg, tiny_model):
    sched = _make_sched(tiny_model)
    for _ in range(3):
        sched.step()
    sched.set_lr_scale(0.5)

    bleu_state = {
        "best": 12.5, "no_improve": 1, "plateau": 0, "patience": 1,
        "label_smoothing": 0.05, "lr_scale": 0.5,
    }
    history = {
        "train_loss": [1.0], "train_acc": [0.5], "val_loss": [1.0],
        "val_acc": [0.5], "bleu_scores": [[1, 12.5]], "lr": [0.5],
    }
    cfg = SimpleNamespace(label_smoothing=0.1, seed=0)

    path = str(tmp_path / "ck.pt")
    save_checkpoint(
        0, tiny_model, sched, epoch=7, path=path,
        bleu_state=bleu_state, history=history, cfg=cfg,
    )

    # atomic save leaves no temp files behind
    assert os.listdir(tmp_path) == ["ck.pt"]

    # fresh model (different init) + scheduler restore everything
    from model import Transformer

    torch.manual_seed(123)
    model2 = Transformer(
        src_vocab=tiny_cfg.vocab_size, tgt_vocab=tiny_cfg.vocab_size,
        d_model=tiny_cfg.d_model, num_layers=tiny_cfg.num_layers,
        num_heads=tiny_cfg.num_heads, d_ff=tiny_cfg.d_ff,
        dropout=0.0, max_len=tiny_cfg.max_len, tie_weights=True,
    )
    sched2 = _make_sched(model2)

    start_epoch, bleu2, hist2 = load_checkpoint(
        model2, sched2, path, torch.device("cpu"), rank=0,
        cfg=SimpleNamespace(label_smoothing=0.1),
    )

    assert start_epoch == 8
    assert bleu2 == bleu_state
    assert hist2 == history
    assert sched2.get_step() == 3
    assert abs(sched2.get_lr() - sched.get_lr()) < 1e-12
    for p1, p2 in zip(tiny_model.parameters(), model2.parameters()):
        assert torch.equal(p1.cpu(), p2.cpu())


def test_payload_is_weights_only_loadable(tmp_path, tiny_model):
    """The saved payload must work with torch's SAFE loader (no pickle exec)."""
    sched = _make_sched(tiny_model)
    path = str(tmp_path / "ck.pt")
    save_checkpoint(
        0, tiny_model, sched, epoch=1, path=path,
        bleu_state={"best": -1.0}, history={}, cfg=SimpleNamespace(),
    )
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    assert ckpt["epoch"] == 1


def test_missing_checkpoint_returns_defaults(tmp_path, tiny_model):
    sched = _make_sched(tiny_model)
    start_epoch, bleu, history = load_checkpoint(
        tiny_model, sched, str(tmp_path / "nope.pt"),
        torch.device("cpu"), rank=0,
        cfg=SimpleNamespace(label_smoothing=0.1),
    )
    assert start_epoch == 1
    assert bleu["best"] == -1.0
    assert history["train_loss"] == []


def test_non_master_rank_skips_save(tmp_path, tiny_model):
    sched = _make_sched(tiny_model)
    path = str(tmp_path / "ck.pt")
    save_checkpoint(
        1, tiny_model, sched, epoch=1, path=path,
        bleu_state={}, history={}, cfg=SimpleNamespace(),
    )
    assert not os.path.exists(path)
