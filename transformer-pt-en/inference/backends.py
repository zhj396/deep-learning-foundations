# -*- coding: utf-8 -*-
"""Model loading + decoding backends for the translation service.

Two interchangeable backends over the same deployment contract:

    torch (default)  artifacts/<model>/checkpoint.pt                  via PyTorch
    onnx             artifacts/<model>/{encoder,decoder_step}.onnx    via ONNX Runtime

Heavy runtimes are imported LAZILY inside the backend classes, so a service
started with BACKEND=onnx never imports torch -- the property that makes the
ONNX path deployable to torch-free edge environments.

Both backends expose:
    translate(model_name, src_ids, params) -> list[list[int]]
        src_ids: padded (B, S) int64 token ids (BOS ... EOS already added
        by tokenization.Translator); returns raw decoded id sequences
    describe(model_name) -> dict          decode signature for /model-info
    available_models()   -> list[str]     artifact dirs this backend can serve
    cached_models()      -> list[str]

The torch backend reuses train/decoding.py verbatim (greedy AND beam); the
ONNX backend reimplements the greedy loop over numpy (greedy only -- beam on
edge hardware is a latency non-starter, and one decoding implementation per
mode keeps the parity testable).
"""

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger("inference")

# Deployment contract shared with training (copy checkpoints here, or export
# with ../train/export_onnx.py; override the root with ARTIFACTS_DIR)
ARTIFACTS_DIR = Path(
    os.environ.get("ARTIFACTS_DIR", Path(__file__).resolve().parent.parent / "artifacts")
)

# train/ holds the model definition and the reference decoding routines
TRAIN_DIR = Path(__file__).resolve().parent.parent / "train"

SPM_FILENAME = "spm_en_pt.model"


@dataclass(frozen=True)
class DecodeParams:
    """Everything a decode call needs beyond the source ids.

    Callers fill bos_idx/eos_idx/pad_idx from the spm model via
    Translator.decode_params(); the defaults below only mirror the ids
    data_prep trains into every spm model (pad/unk/bos/eos = 0/1/2/3).
    """

    mode: str = "greedy"          # greedy | beam
    max_len: int = 128            # max generated sequence length (incl. BOS)
    beam_size: int = 4
    length_penalty: float = 0.6
    bos_idx: int = 2
    eos_idx: int = 3
    pad_idx: int = 0


def _ensure_train_on_path() -> None:
    if str(TRAIN_DIR) not in sys.path:
        sys.path.insert(0, str(TRAIN_DIR))


class BackendBase:
    """Common interface + model caching for all serving backends."""

    name = "base"

    def __init__(self) -> None:
        self._cache: Dict[str, object] = {}
        self._lock = Lock()

    # -- to be implemented by subclasses ------------------------------------
    @property
    def required_files(self) -> Tuple[str, ...]:
        """Files under artifacts/<model>/ this backend needs (besides the spm model)."""
        raise NotImplementedError

    def _load(self, model_name: str, model_dir: Path):
        raise NotImplementedError

    def _translate(self, handle, src_ids: np.ndarray, params: DecodeParams) -> List[List[int]]:
        raise NotImplementedError

    def _describe(self, handle) -> Dict:
        raise NotImplementedError

    # -- shared implementation ----------------------------------------------
    def _required(self) -> Tuple[str, ...]:
        return (*self.required_files, SPM_FILENAME)

    def load(self, model_name: str):
        """Load (and cache) a model; thread-safe."""
        with self._lock:
            if model_name in self._cache:
                return self._cache[model_name]

            model_dir = ARTIFACTS_DIR / model_name
            missing = [f for f in self._required() if not (model_dir / f).exists()]
            if missing:
                raise FileNotFoundError(
                    f"Model {model_name!r} is missing {missing} under {model_dir}. "
                    f"Available models: {self.available_models() or 'none'}. "
                    f"Copy a training checkpoint + {SPM_FILENAME} into the artifacts dir "
                    f"(or set ARTIFACTS_DIR); the onnx backend additionally needs "
                    f"train/export_onnx.py to have run."
                )

            logger.info(f"[{self.name}] Loading model: {model_name}")
            handle = self._load(model_name, model_dir)
            self._cache[model_name] = handle
            logger.info(f"[{self.name}] Model {model_name} loaded and cached")
            return handle

    def translate(self, model_name: str, src_ids: np.ndarray,
                  params: DecodeParams) -> List[List[int]]:
        """Decode one padded batch of source ids; returns id sequences."""
        return self._translate(self.load(model_name), src_ids, params)

    def describe(self, model_name: str) -> Dict:
        """Decode signature for /model-info."""
        return self._describe(self.load(model_name))

    def available_models(self) -> List[str]:
        """Models on disk that this backend can serve."""
        if not ARTIFACTS_DIR.exists():
            return []
        return sorted(
            d.name for d in ARTIFACTS_DIR.iterdir()
            if d.is_dir() and all((d / f).exists() for f in self._required())
        )

    def cached_models(self) -> List[str]:
        return sorted(self._cache.keys())


class TorchCkptBackend(BackendBase):
    """PyTorch checkpoint backend.

    Rebuilds the Transformer from the config dict EMBEDDED in the checkpoint
    (train/checkpoint.py stores it), loads weights with the safe loader, and
    reuses train/decoding.py verbatim so served outputs are bit-identical to
    in-training BLEU probes.
    """

    name = "torch"

    @property
    def required_files(self) -> Tuple[str, ...]:
        return ("checkpoint.pt",)

    def _load(self, model_name: str, model_dir: Path):
        import torch  # lazy: the onnx serving path never imports torch

        _ensure_train_on_path()
        from model import Transformer

        ckpt = torch.load(str(model_dir / "checkpoint.pt"), map_location="cpu", weights_only=True)
        cfg = ckpt.get("config") or {}
        if "vocab_size" not in cfg or "model_state" not in ckpt:
            raise ValueError(
                f"Checkpoint for {model_name!r} lacks config.vocab_size or model_state; "
                f"was it saved by train/checkpoint.py?"
            )

        model = Transformer(
            src_vocab=cfg["vocab_size"],
            tgt_vocab=cfg["vocab_size"],
            d_model=cfg.get("d_model", 512),
            num_layers=cfg.get("num_layers", 6),
            num_heads=cfg.get("num_heads", 8),
            d_ff=cfg.get("d_ff", 2048),
            dropout=cfg.get("dropout", 0.1),
            max_len=cfg.get("max_len", 512),
            tie_weights=cfg.get("tie_weights", True),
        )
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.eval()
        return {"model": model, "config": cfg}

    def _translate(self, handle, src_ids: np.ndarray, params: DecodeParams) -> List[List[int]]:
        import torch

        _ensure_train_on_path()
        from decoding import beam_decode_batch, greedy_decode_batch

        src = torch.from_numpy(np.ascontiguousarray(src_ids, dtype=np.int64))
        common = dict(
            max_len=params.max_len,
            bos_idx=params.bos_idx,
            eos_idx=params.eos_idx,
            pad_idx=params.pad_idx,
            device=torch.device("cpu"),
        )
        if params.mode == "greedy":
            out = greedy_decode_batch(handle["model"], src, **common)
        elif params.mode == "beam":
            out = beam_decode_batch(
                handle["model"], src,
                beam_size=params.beam_size,
                length_penalty=params.length_penalty,
                **common,
            )
        else:
            raise ValueError(f"Unknown decode mode: {params.mode!r} (greedy | beam)")
        return out.tolist()

    def _describe(self, handle) -> Dict:
        cfg = handle["config"]
        model = handle["model"]
        return {
            "backend": self.name,
            "modes": ["greedy", "beam"],
            "parameters": sum(p.numel() for p in model.parameters()),
            "config": {
                k: cfg.get(k)
                for k in ("vocab_size", "d_model", "num_layers", "num_heads",
                          "d_ff", "max_len", "tie_weights")
            },
        }


class OrtBackend(BackendBase):
    """ONNX Runtime backend (CPU-optimized, torch-free).

    Greedy decoding is a numpy loop over the two exported graphs, mirroring
    train/decoding.py step for step (finished sequences are frozen with PAD,
    the loop breaks when every row hit EOS).
    """

    name = "onnx"

    @property
    def required_files(self) -> Tuple[str, ...]:
        return ("encoder.onnx", "decoder_step.onnx")

    def _load(self, model_name: str, model_dir: Path):
        import onnxruntime as ort  # lazy: only this backend pulls ORT in

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "4"))
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = ["CPUExecutionProvider"]
        encoder = ort.InferenceSession(
            str(model_dir / "encoder.onnx"), sess_options=sess_options, providers=providers
        )
        decoder = ort.InferenceSession(
            str(model_dir / "decoder_step.onnx"), sess_options=sess_options, providers=providers
        )
        return {"encoder": encoder, "decoder": decoder}

    def _translate(self, handle, src_ids: np.ndarray, params: DecodeParams) -> List[List[int]]:
        if params.mode != "greedy":
            raise ValueError(
                f"ONNX backend supports greedy decoding only (got {params.mode!r}). "
                f"Start the service with BACKEND=torch for beam search."
            )

        src = np.ascontiguousarray(src_ids, dtype=np.int64)
        batch = src.shape[0]

        memory = handle["encoder"].run(None, {"src_tokens": src})[0]

        ys = np.full((batch, 1), params.bos_idx, dtype=np.int64)
        finished = np.zeros(batch, dtype=bool)

        for _ in range(params.max_len - 1):
            logits = handle["decoder"].run(
                None, {"ys": ys, "memory": memory, "src_tokens": src}
            )[0]
            nxt = logits.argmax(axis=-1)
            nxt[finished] = params.pad_idx

            ys = np.concatenate([ys, nxt[:, None]], axis=1)
            finished |= nxt == params.eos_idx
            if finished.all():
                break

        return ys.tolist()

    def _describe(self, handle) -> Dict:
        def io_dict(session):
            return {
                "inputs": {
                    i.name: {"shape": list(i.shape), "dtype": i.type}
                    for i in session.get_inputs()
                },
                "outputs": {
                    o.name: {"shape": list(o.shape), "dtype": o.type}
                    for o in session.get_outputs()
                },
            }

        return {
            "backend": self.name,
            "modes": ["greedy"],
            "encoder": io_dict(handle["encoder"]),
            "decoder_step": io_dict(handle["decoder"]),
        }


BACKENDS = {cls.name: cls for cls in (TorchCkptBackend, OrtBackend)}


def get_backend(name: str) -> BackendBase:
    """Instantiate a backend by name ('torch' | 'onnx')."""
    name = (name or "torch").lower()
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend: {name!r}. Available: {sorted(BACKENDS)}")
    return BACKENDS[name]()
