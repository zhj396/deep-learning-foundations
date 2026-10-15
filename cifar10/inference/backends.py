# -*- coding: utf-8 -*-
"""
Model loading backends for the CIFAR-10 inference service.

Two interchangeable backends over the same deployment contract:

    onnx  (default)  artifacts/<model>/model.onnx      via ONNX Runtime
    tf               artifacts/<model>/saved_model/    via tf.saved_model

Heavy runtimes are imported LAZILY inside the backend classes, so a service
started with BACKEND=onnx never imports TensorFlow — the property that makes
the ONNX path deployable to TF-free edge environments.

Both backends expose the same interface:
    predict(model_name, batch) -> np.ndarray   # softmax probabilities
    describe(model_name)       -> dict         # input/output signature
    available_models()         -> list[str]
"""

import logging
import os
from pathlib import Path
from threading import Lock
from typing import Dict, List

import numpy as np

logger = logging.getLogger("inference")

# Deployment contract shared with training (train/train.py exports here)
ARTIFACTS_DIR = Path(
    os.environ.get("ARTIFACTS_DIR", Path(__file__).resolve().parent.parent / "artifacts")
)


class BackendBase:
    """Common interface + model caching for all serving backends."""

    name = "base"

    def __init__(self) -> None:
        self._cache: Dict[str, object] = {}
        self._lock = Lock()

    # -- to be implemented by subclasses ------------------------------------
    @property
    def artifact_relpath(self) -> Path:
        """Model file/dir (relative to artifacts/<model>/) this backend serves."""
        raise NotImplementedError

    def _load(self, model_name: str, model_path: Path):
        raise NotImplementedError

    def _predict(self, handle, batch: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _describe(self, handle) -> Dict:
        raise NotImplementedError

    # -- shared implementation ----------------------------------------------
    def _check_input_contract(self, rank4_shape, model_name: str) -> None:
        """Fail fast if the model does not match the NHWC (batch, H, W, 3) contract."""
        if len(rank4_shape) != 4 or rank4_shape[-1] != 3:
            raise ValueError(
                f"Model {model_name}: expected NHWC input (batch, H, W, 3), "
                f"got {list(rank4_shape)}. Re-export with train/export_onnx.py."
            )

    def load(self, model_name: str):
        """Load (and cache) a model; thread-safe."""
        with self._lock:
            if model_name in self._cache:
                return self._cache[model_name]

            model_path = ARTIFACTS_DIR / model_name / self.artifact_relpath
            exists = (
                (model_path / "saved_model.pb").exists()
                if model_path.is_dir() else model_path.exists()
            )
            if not exists:
                raise FileNotFoundError(
                    f"No {self.name} model for {model_name!r} at {model_path}. "
                    f"Available models: {self.available_models() or 'none'}. "
                    f"Train with ../train/train.py or set ARTIFACTS_DIR."
                )

            logger.info(f"[{self.name}] Loading model: {model_name} -> {model_path}")
            handle = self._load(model_name, model_path)
            self._cache[model_name] = handle
            logger.info(f"[{self.name}] Model {model_name} loaded and cached")
            return handle

    def predict(self, model_name: str, batch: np.ndarray) -> np.ndarray:
        """Run one batch; returns softmax probabilities (N, num_classes)."""
        return self._predict(self.load(model_name), batch)

    def describe(self, model_name: str) -> Dict:
        """Input/output signature for /model-info."""
        return self._describe(self.load(model_name))

    def available_models(self) -> List[str]:
        """Models on disk that this backend can serve."""
        if not ARTIFACTS_DIR.exists():
            return []
        return sorted(
            d.name for d in ARTIFACTS_DIR.iterdir()
            if d.is_dir() and self._artifact_exists(d / self.artifact_relpath)
        )

    @staticmethod
    def _artifact_exists(path: Path) -> bool:
        return (path / "saved_model.pb").exists() if path.is_dir() else path.exists()

    def cached_models(self) -> List[str]:
        return sorted(self._cache.keys())


class OrtBackend(BackendBase):
    """ONNX Runtime backend (CPU-optimized, no TensorFlow dependency)."""

    name = "onnx"

    @property
    def artifact_relpath(self) -> Path:
        return Path("model.onnx")

    def _load(self, model_name: str, model_path: Path):
        import onnxruntime as ort  # lazy: keeps TF-free deployments TF-free

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        session = ort.InferenceSession(
            str(model_path), sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        inp = session.get_inputs()[0]
        self._check_input_contract(inp.shape, model_name)
        return session

    def _predict(self, session, batch: np.ndarray) -> np.ndarray:
        input_name = session.get_inputs()[0].name
        return session.run(None, {input_name: batch})[0]

    def _describe(self, session) -> Dict:
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        return {
            "inputs": {i.name: {"shape": list(i.shape), "dtype": i.type} for i in inputs},
            "outputs": [o.name for o in outputs],
        }


class TfSavedModelBackend(BackendBase):
    """TensorFlow SavedModel backend (serving_default signature)."""

    name = "tf"

    @property
    def artifact_relpath(self) -> Path:
        return Path("saved_model")

    def _load(self, model_name: str, model_path: Path):
        import tensorflow as tf  # lazy: only this backend pulls TF in

        loaded = tf.saved_model.load(str(model_path))
        infer = loaded.signatures["serving_default"]
        shape = infer.structured_input_signature[1]
        first = next(iter(shape.values()))
        self._check_input_contract(first.shape, model_name)
        return infer

    def _predict(self, infer, batch: np.ndarray) -> np.ndarray:
        import tensorflow as tf

        output = infer(tf.constant(batch))
        return list(output.values())[0].numpy()

    def _describe(self, infer) -> Dict:
        return {
            "inputs": {
                k: {"shape": str(v.shape), "dtype": str(v.dtype)}
                for k, v in infer.structured_input_signature[1].items()
            },
            "outputs": list(infer.structured_outputs.keys()),
        }


BACKENDS = {cls.name: cls for cls in (OrtBackend, TfSavedModelBackend)}


def get_backend(name: str) -> BackendBase:
    """Instantiate a backend by name ('onnx' | 'tf')."""
    name = (name or "onnx").lower()
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend: {name!r}. Available: {sorted(BACKENDS)}")
    return BACKENDS[name]()
