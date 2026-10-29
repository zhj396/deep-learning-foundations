# -*- coding: utf-8 -*-
"""
Inference service test suite - Flask test client + a synthetic ONNX model.

Covers the serving layer only (no CIFAR download, no pretrained weights):

1. Preprocessing contract (preprocessing.py): unit scaling, caffe BGR mean
   subtraction, and the EfficientNet pass-through, asserted numerically.
2. DynamicBatcher (batcher.py): intra-request batching through one
   predict_fn call, per-item error propagation without killing the worker,
   and the wait() timeout safety net.
3. HTTP surface (app.py) via Flask's test client against a tiny synthetic
   ONNX logistic-regression model: /health, /model-info, /predict
   (single, multi-image, ZIP, bad file), /predict_from_dir with upload-dir
   path confinement, and CSV download.

app.py reads artifacts through backends.ARTIFACTS_DIR and stores uploads in
tempfile.gettempdir() subdirectories, so the test run points the
service at a per-run temp artifacts directory and real PNGs exercise
the full preprocess -> batcher -> ONNX path with no TensorFlow anywhere.
"""

import io
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, TensorProto
import pytest
from PIL import Image

import app as service
from backends import OrtBackend
from batcher import DynamicBatcher, RequestItem
from preprocessing import (
    KERAS_CAFFE_BGR_MEAN,
    MODEL_SPECS,
    preprocess_image,
)

# ---------------------------------------------------------------------------
# Synthetic ONNX model: NHWC float32 (N, 4, 4, 3) -> 10 logits
# MatMul over flattened pixels + bias; deterministic, weight-free fixture.
# ---------------------------------------------------------------------------


def _build_synthetic_onnx(path: Path, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    weights = rng.normal(scale=0.1, size=(4 * 4 * 3, 10)).astype(np.float32)
    bias = rng.normal(scale=0.01, size=(10,)).astype(np.float32)

    initializers = [
        helper.make_tensor("weights", TensorProto.FLOAT, weights.shape,
                           weights.flatten().tolist()),
        helper.make_tensor("bias", TensorProto.FLOAT, bias.shape, bias.tolist()),
    ]
    nodes = [
        helper.make_node("Reshape", ["input", "shape_2d"], ["flat"]),
        helper.make_node("MatMul", ["flat", "weights"], ["logits"]),
        helper.make_node("Add", ["logits", "bias"], ["output_0"]),
    ]
    graph = helper.make_graph(
        nodes, "synthetic_clf",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 4, 4, 3])],
        [helper.make_tensor_value_info("output_0", TensorProto.FLOAT, ["N", 10])],
        initializer=initializers,
    )
    shape_const = helper.make_tensor(
        "shape_2d", TensorProto.INT64, [2], [-1, 4 * 4 * 3])
    graph.initializer.append(shape_const)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9  # keep the fixture loadable across ORT versions
    onnx.save(model, str(path))


def _png_bytes(seed: int = 0, size: int = 28) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def artifacts():
    """Synthetic model dropped into the ARTIFACTS_DIR chosen by conftest.py
    (set before app.py was imported, so the service only sees this model),
    plus a matching preprocessing spec (4x4, pass-through)."""
    from backends import ARTIFACTS_DIR
    model_dir = ARTIFACTS_DIR / "synth"
    model_dir.mkdir(exist_ok=True)
    _build_synthetic_onnx(model_dir / "model.onnx")
    MODEL_SPECS["synth"] = (4, lambda x: x)  # resize to 4x4, raw float32
    yield ARTIFACTS_DIR
    MODEL_SPECS.pop("synth", None)
    return ARTIFACTS_DIR


@pytest.fixture(scope="module")
def client(artifacts):
    """Flask test client pointed at the temp artifacts dir.

    app.py resolves upload/result directories at import time, so the
    ARTIFACTS_DIR override happens on first import of this test module's
    app instance (env is set in conftest.py before the import).
    """
    service.app.config["TESTING"] = True
    with service.app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# 1. Preprocessing contract
# ---------------------------------------------------------------------------


class TestPreprocessing:
    def test_unit_scale_cnn(self):
        x = np.full((32, 32, 3), 128.0)
        out = MODEL_SPECS["cnn"][1](x)
        assert out.shape == (32, 32, 3)
        np.testing.assert_allclose(out, 128.0 / 255.0)

    def test_caffe_bgr_resnet50(self):
        # distinct channel values: RGB -> BGR swap + mean subtraction
        x = np.zeros((224, 224, 3), dtype=np.float32)
        x[..., 0] = 200.0  # R -> lands last after swap
        x[..., 2] = 100.0  # B -> lands first after swap
        out = MODEL_SPECS["resnet50"][1](x)
        np.testing.assert_allclose(out[..., 0], 100.0 - KERAS_CAFFE_BGR_MEAN[0])
        np.testing.assert_allclose(out[..., 1], 0.0 - KERAS_CAFFE_BGR_MEAN[1])
        np.testing.assert_allclose(out[..., 2], 200.0 - KERAS_CAFFE_BGR_MEAN[2])

    def test_efficientnet_passthrough(self):
        x = np.full((224, 224, 3), 255.0, dtype=np.float32)
        out = MODEL_SPECS["efficientnetb0"][1](x)
        np.testing.assert_array_equal(out, x)  # raw [0, 255], unchanged

    def test_resizes_and_returns_batch(self):
        out = preprocess_image("cnn", _png_bytes(1, size=28))
        assert out.shape == (1, 32, 32, 3)
        assert out.dtype == np.float32

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model type"):
            preprocess_image("nope", _png_bytes())


# ---------------------------------------------------------------------------
# 2. DynamicBatcher: batching, failure safety, timeout
# ---------------------------------------------------------------------------


class TestBatcher:
    def test_intra_request_batching_single_call(self):
        calls = []

        def predict_fn(batch):
            calls.append(batch.shape[0])
            return np.tile(np.arange(10, dtype=np.float32) / 10, (batch.shape[0], 1))

        batcher = DynamicBatcher(predict_fn, poll_interval_s=0.02)
        items = [RequestItem(np.zeros((1, 4, 4, 3), np.float32), f"{i}.png")
                 for i in range(5)]
        batcher.submit(items)
        results = [batcher.wait(it) for it in items]
        assert len(results) == 5
        assert calls == [5]  # one backend call for the whole request
        for it in items:
            assert it.error is None

    def test_error_does_not_kill_worker(self):
        state = {"fail": True}

        def predict_fn(batch):
            if state["fail"]:
                raise RuntimeError("boom")
            return np.ones((batch.shape[0], 10), np.float32) / 10

        batcher = DynamicBatcher(predict_fn)
        bad = [RequestItem(np.zeros((1, 4, 4, 3), np.float32), "bad.png")]
        batcher.submit(bad)
        with pytest.raises(RuntimeError, match="boom"):
            batcher.wait(bad[0])
        assert batcher.alive  # worker survived the exception

        state["fail"] = False
        good = [RequestItem(np.zeros((1, 4, 4, 3), np.float32), "good.png")]
        batcher.submit(good)
        result = batcher.wait(good[0])  # same worker serves the next batch
        np.testing.assert_allclose(result, 0.1)

    def test_wait_timeout(self):
        def predict_fn(batch):  # never returns
            time.sleep(1.0)
            return np.ones((batch.shape[0], 10), np.float32) / 10

        batcher = DynamicBatcher(predict_fn, wait_timeout_s=0.05)
        item = RequestItem(np.zeros((1, 4, 4, 3), np.float32), "slow.png")
        batcher.submit([item])
        with pytest.raises(TimeoutError):
            batcher.wait(item)

    def test_multi_request_items_merge_into_batches(self):
        # Four separate requests (each a 3-item submit) share one batcher:
        # items still queued from earlier requests merge into the same calls.
        calls, lock = [], threading.Lock()

        def predict_fn(batch):
            with lock:
                calls.append(batch.shape[0])
            time.sleep(0.02)  # widen the poll window so merging is observable
            return np.ones((batch.shape[0], 10), np.float32) / 10

        batcher = DynamicBatcher(predict_fn, poll_interval_s=0.02)
        per_request = []
        for _ in range(4):
            items = [RequestItem(np.zeros((1, 4, 4, 3), np.float32), f"{i}.png")
                     for i in range(3)]
            batcher.submit(items)
            per_request.append(items)
        all_items = [it for group in per_request for it in group]
        for it in all_items:
            batcher.wait(it)
        assert sum(calls) == 12  # every image predicted exactly once
        assert max(calls) > 1    # and at least one batch merged >1 item


# ---------------------------------------------------------------------------
# 3. HTTP surface via Flask test client
# ---------------------------------------------------------------------------


class TestHealthAndModelInfo:
    def test_health_lists_synthetic_model(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.get_json()
        assert body["status"] == "healthy"
        assert body["backend"] == "onnx"
        assert "synth" in body["models_available"]

    def test_model_info_signature(self, client):
        r = client.get("/model-info?model=synth")
        assert r.status_code == 200
        sig = r.get_json()["signature"]
        assert sig["inputs"]["input"]["shape"] == ["N", 4, 4, 3]

    def test_model_info_unknown_model(self, client):
        r = client.get("/model-info?model=missing")
        assert r.status_code == 500
        assert r.get_json()["status"] == "error"


class TestPredict:
    def test_single_image(self, client):
        r = client.post("/predict", data={
            "image": (io.BytesIO(_png_bytes(2)), "img.png", "image/png"),
            "model": "synth", "top_k": "3",
        })
        assert r.status_code == 200
        body = r.get_json()
        assert body["status"] == "success" and body["total"] == 1
        top = body["results"][0]["top"]
        assert len(top) == 3
        # confidences are softmax-free logits from the fixture but still
        # descending and named with the 10 CIFAR classes
        assert top[0]["confidence"] >= top[-1]["confidence"]
        assert {t["class"] for t in top} <= set(service.CLASS_NAMES)
        assert "csv_url" in body

    def test_multi_image_batched(self, client):
        r = client.post("/predict", data={
            "image": [(io.BytesIO(_png_bytes(10 + i)), f"{i}.png", "image/png")
                      for i in range(4)],
            "model": "synth", "top_k": "1",
        })
        assert r.status_code == 200
        body = r.get_json()
        assert body["total"] == 4
        assert [res["filename"] for res in body["results"]] == \
            [f"{i}.png" for i in range(4)]

    def test_zip_batch(self, client):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for i in range(3):
                zf.writestr(f"{i}.png", _png_bytes(20 + i))
            zf.writestr("notes.txt", b"not an image")  # filtered out, not extracted
        r = client.post("/predict", data={
            "zip": (io.BytesIO(buf.getvalue()), "batch.zip", "application/zip"),
            "model": "synth",
        })
        assert r.status_code == 200
        body = r.get_json()
        assert body["total"] == 3  # txt member excluded

    def test_invalid_extension_rejected(self, client):
        r = client.post("/predict", data={
            "image": (io.BytesIO(b"MZ..."), "evil.exe", "application/octet-stream"),
            "model": "synth",
        })
        assert r.status_code == 400

    def test_unknown_model_reports_per_file_failure(self, client):
        # valid image, but preprocessing for a nonexistent model fails:
        # no result succeeded, so the request fails with per-file details
        r = client.post("/predict", data={
            "image": (io.BytesIO(_png_bytes(3)), "img.png", "image/png"),
            "model": "missing-model",
        })
        assert r.status_code == 400  # no result succeeded
        body = r.get_json()
        assert body["status"] == "error"
        assert body["details"][0]["filename"] == "img.png"

    def test_csv_download(self, client):
        r = client.post("/predict", data={
            "image": (io.BytesIO(_png_bytes(4)), "img.png", "image/png"),
            "model": "synth", "top_k": "2",
        })
        csv_url = r.get_json()["csv_url"]
        r2 = client.get("/" + csv_url.split("//", 1)[1].split("/", 1)[1])
        assert r2.status_code == 200
        lines = r2.get_data(as_text=True).strip().splitlines()
        assert lines[0] == "filename,class,confidence,rank"
        assert len(lines) == 3  # header + 2 rows (top_k=2)


class TestPredictFromDir:
    def test_path_confinement(self, client, tmp_path):
        # write one file inside the upload dir and one outside it
        inside = service.UPLOAD_DIR / "inside_test.png"
        inside.write_bytes(_png_bytes(5))
        outside = tmp_path / "outside.png"
        outside.write_bytes(_png_bytes(6))
        try:
            r = client.post("/predict_from_dir", json={
                "model_name": "synth",
                "files": [str(inside), str(outside)],
            })
            assert r.status_code == 200
            body = r.get_json()
            # the outside path is silently dropped by the confinement filter
            assert body["total"] == 1
            assert body["results"][0]["filename"] == "inside_test.png"
        finally:
            inside.unlink(missing_ok=True)

    def test_all_paths_outside_rejected(self, client, tmp_path):
        outside = tmp_path / "x.png"
        outside.write_bytes(_png_bytes(7))
        r = client.post("/predict_from_dir", json={
            "model_name": "synth", "files": [str(outside)],
        })
        assert r.status_code == 400
        assert "rejected" in r.get_json()["error"]

    def test_empty_body(self, client):
        r = client.post("/predict_from_dir", json={})
        assert r.status_code == 400


class TestZipSlipProtection:
    def test_malicious_zip_entry_rejected(self, client):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("../../evil.png", _png_bytes(8))
        r = client.post("/upload", data={
            "files": (io.BytesIO(buf.getvalue()), "attack.zip", "application/zip"),
        })
        assert r.status_code == 200  # upload succeeds but extraction failed
        body = r.get_json()
        assert body["skipped"] == ["attack.zip"]


class TestBackendContract:
    def test_ort_backend_rejects_non_nhwc(self):
        # a model whose input is (N, 3, 4, 4) violates the NHWC contract
        from backends import ARTIFACTS_DIR
        bad_dir = ARTIFACTS_DIR / "bad"
        bad_dir.mkdir(exist_ok=True)
        _build_synthetic_onnx(bad_dir / "model.onnx", seed=1)
        import onnx as _onnx
        m = _onnx.load(str(bad_dir / "model.onnx"))
        m.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 3
        m.graph.input[0].type.tensor_type.shape.dim[2].dim_value = 4
        m.graph.input[0].type.tensor_type.shape.dim[3].dim_value = 4
        _onnx.save(m, str(bad_dir / "model.onnx"))
        backend = OrtBackend()
        with pytest.raises(ValueError, match="NHWC"):
            backend.load("bad")

    def test_no_tensorflow_imported(self):
        import sys
        assert "tensorflow" not in sys.modules  # ONNX path stays TF-free
