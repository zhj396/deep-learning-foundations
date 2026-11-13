"""Backend contract tests: torch checkpoint serving + ONNX export parity.

Artifacts are built in tmp_path (tiny model, no downloads); the module-level
ARTIFACTS_DIR in backends.py is monkeypatched to point at them.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import backends as backends_mod
from backends import DecodeParams
from checkpoint import save_checkpoint


def _tiny_cfg_dict(tiny_cfg):
    return {
        "vocab_size": tiny_cfg.vocab_size,
        "d_model": tiny_cfg.d_model,
        "num_layers": tiny_cfg.num_layers,
        "num_heads": tiny_cfg.num_heads,
        "d_ff": tiny_cfg.d_ff,
        "max_len": tiny_cfg.max_len,
        "tie_weights": True,
        "pad_idx": 0,
        "bos_idx": 2,
        "eos_idx": 3,
    }


@pytest.fixture
def artifacts(tmp_path, tiny_cfg, tiny_model, monkeypatch):
    """Checkpoint-only artifact tree; ARTIFACTS_DIR redirected to it."""
    root = tmp_path / "artifacts"
    model_dir = root / "base"
    model_dir.mkdir(parents=True)

    save_checkpoint(
        0, tiny_model, None, epoch=1, path=str(model_dir / "checkpoint.pt"),
        bleu_state={}, history={}, cfg=SimpleNamespace(**_tiny_cfg_dict(tiny_cfg)),
    )
    # the artifacts contract requires the spm model alongside the weights;
    # backends never parse it (tokenization.Translator does), so a stub is
    # enough for backend-level tests
    (model_dir / "spm_en_pt.model").write_bytes(b"stub-for-artifact-contract")

    monkeypatch.setattr(backends_mod, "ARTIFACTS_DIR", root)
    return model_dir


@pytest.fixture
def onnx_artifacts(artifacts):
    """artifacts + exported ONNX graphs."""
    pytest.importorskip("onnxruntime")
    from export_onnx import export_checkpoint_to_onnx

    export_checkpoint_to_onnx(artifacts / "checkpoint.pt", artifacts, validate=False)
    return artifacts


def test_torch_backend_greedy(artifacts, tiny_cfg):
    backend = backends_mod.get_backend("torch")
    assert backend.available_models() == ["base"]

    src = np.array([[2, 5, 6, 3, 0, 0]], dtype=np.int64)
    params = DecodeParams(max_len=8)

    out1 = backend.translate("base", src, params)
    out2 = backend.translate("base", src, params)
    assert out1 == out2                       # eval mode -> deterministic
    assert out1[0][0] == 2                    # starts with BOS
    assert all(0 <= t < tiny_cfg.vocab_size for t in out1[0])
    assert out1[0][1] == 3 or len(out1[0]) <= 8

    info = backend.describe("base")
    assert info["modes"] == ["greedy", "beam"]
    assert info["config"]["d_model"] == tiny_cfg.d_model
    assert "base" in backend.cached_models()


def test_torch_backend_beam_runs(artifacts):
    backend = backends_mod.get_backend("torch")
    src = np.array([[2, 5, 6, 3]], dtype=np.int64)
    out = backend.translate("base", src, DecodeParams(mode="beam", max_len=8, beam_size=3))
    assert out[0][0] == 2
    assert len(out) == 1


def test_missing_model_raises(artifacts):
    backend = backends_mod.get_backend("torch")
    with pytest.raises(FileNotFoundError, match="missing"):
        backend.translate("nonexistent", np.zeros((1, 2), dtype=np.int64), DecodeParams())


def test_onnx_export_writes_valid_graphs(onnx_artifacts):
    import onnx

    for name in ("encoder.onnx", "decoder_step.onnx"):
        path = onnx_artifacts / name
        assert path.exists()
        onnx.checker.check_model(onnx.load(str(path)))

    # both backends see the model
    assert "base" in backends_mod.get_backend("onnx").available_models()


def test_onnx_greedy_matches_torch_greedy(onnx_artifacts):
    """The numpy greedy loop must reproduce train/decoding.py exactly."""
    torch_out = backends_mod.get_backend("torch").translate(
        "base",
        np.array([[2, 5, 6, 3], [2, 7, 8, 0]], dtype=np.int64),
        DecodeParams(max_len=10),
    )
    ort_out = backends_mod.get_backend("onnx").translate(
        "base",
        np.array([[2, 5, 6, 3], [2, 7, 8, 0]], dtype=np.int64),
        DecodeParams(max_len=10),
    )
    assert ort_out == torch_out


def test_onnx_backend_rejects_beam(onnx_artifacts):
    backend = backends_mod.get_backend("onnx")
    with pytest.raises(ValueError, match="greedy"):
        backend.translate("base", np.array([[2, 5, 3]], dtype=np.int64),
                          DecodeParams(mode="beam"))
    assert backend.describe("base")["modes"] == ["greedy"]
