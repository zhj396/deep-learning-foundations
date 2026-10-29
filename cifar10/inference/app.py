# -*- coding: utf-8 -*-
"""
CIFAR-10 inference service (single app, pluggable backends)
===========================================================
Serves the models exported by ../train/train.py through the shared artifacts
directory (default ../artifacts, override with ARTIFACTS_DIR):

    artifacts/<model_name>/saved_model/   BACKEND=tf    (tf.saved_model)
    artifacts/<model_name>/model.onnx     BACKEND=onnx  (ONNX Runtime, default)

Select the backend with the BACKEND environment variable. The ONNX backend
never imports TensorFlow (lazy imports in backends.py), which keeps that
deployment path TF-free for edge devices.

Features:
- Web UI (upload -> select -> predict -> CSV download) at /
- Single-image, multi-image, and ZIP batch inference
- Dynamic batching: items from concurrent requests are merged into single
  backend calls (see batcher.py), with per-item errors + timeouts
- Preprocessing mirrors the training pipeline (preprocessing.py)
- Security: secure_filename, zip-slip-safe extraction, upload-dir path
  confinement, request size limit, debug off by default

Endpoints:
    GET  /                     Web UI
    GET  /health               Backend, cached/available models, batcher liveness
    GET  /model-info?model=X   Input/output signature
    POST /upload               Store images/ZIP server-side (field: files)
    GET  /uploaded_files       List stored files
    POST /predict              Inference (fields: image | zip, model, top_k)
    POST /predict_from_dir     Inference on uploaded paths (JSON)
    GET  /download/<name>      Download a result CSV
"""

import csv
import gc
import io
import logging
import os
import tempfile
import threading
import uuid
import zipfile
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Dict, List

import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename

from backends import ARTIFACTS_DIR, get_backend
from batcher import DynamicBatcher, RequestItem
from preprocessing import preprocess_image

# ==========================================================
# Configuration
# ==========================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("inference")

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB per request
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

BACKEND = get_backend(os.environ.get("BACKEND", "onnx"))

UPLOAD_DIR = Path(tempfile.gettempdir()) / "cifar10_inference_uploads"
RESULT_DIR = Path(tempfile.gettempdir()) / "cifar10_inference_results"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

# One dynamic batcher per model, lazily created (see get_batcher)
BATCHERS: Dict[str, DynamicBatcher] = {}
BATCHERS_LOCK = threading.Lock()


# ==========================================================
# Helpers
# ==========================================================
def collect_garbage(func):
    """Run gc.collect() after the request (numpy/PIL buffers add up in batches)."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        finally:
            gc.collect()
    return wrapper


def allowed_file(filename: str) -> bool:
    """Check the file extension against the image whitelist."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def safe_extract_zip(zf: zipfile.ZipFile, dest: Path, accept=None) -> None:
    """Extract a ZIP rejecting entries that escape dest (zip-slip protection).

    Uses the stdlib ``filter='data'`` on Python >= 3.12 and falls back to a
    manual path-containment check on older versions. When ``accept`` is
    given, only members passing it are written to dest.
    """
    dest = dest.resolve()
    for member in zf.infolist():
        target = (dest / member.filename).resolve()
        if not str(target).startswith(str(dest) + os.sep) and target != dest:
            raise ValueError(f"Unsafe path in ZIP (possible zip-slip): {member.filename}")
    members = [m for m in zf.infolist() if accept is None or accept(m.filename)]
    try:
        zf.extractall(dest, members=members, filter="data")  # Python >= 3.12
    except TypeError:
        zf.extractall(dest, members=members)  # older Python: containment already checked above


def build_top_k(pred: np.ndarray, top_k: int) -> List[Dict]:
    """Convert one probability vector into a top-k JSON result."""
    return [
        {"class": CLASS_NAMES[j], "confidence": round(float(pred[j]), 4)}
        for j in np.argsort(pred)[::-1][:top_k]
    ]


def save_results_to_csv(results: List[Dict]) -> str:
    """Persist top-k results as CSV; returns a unique file name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"results_{timestamp}_{uuid.uuid4().hex[:6]}.csv"
    csv_path = RESULT_DIR / csv_filename

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "class", "confidence", "rank"])
        for r in results:
            for rank, pred in enumerate(r["top"], 1):
                writer.writerow([r["filename"], pred["class"], pred["confidence"], rank])

    return csv_filename


def get_batcher(model_name: str) -> DynamicBatcher:
    """Return the (lazily created) batcher for a model, recreating it if its
    worker thread has died for any reason."""
    with BATCHERS_LOCK:
        batcher = BATCHERS.get(model_name)
        if batcher is not None and batcher.alive:
            return batcher
        if batcher is not None:
            logger.error(f"Batcher worker for {model_name} died; recreating")
        batcher = DynamicBatcher(lambda batch: BACKEND.predict(model_name, batch))
        BATCHERS[model_name] = batcher
        return batcher


def run_batched(model_name: str, payloads, top_k: int):
    """Preprocess (name, bytes) payloads, run them through the batcher, and
    collect top-k results plus per-file failure details."""
    batcher = get_batcher(model_name)

    items: List[RequestItem] = []
    failures: List[Dict] = []
    for name, data in payloads:
        try:
            items.append(RequestItem(preprocess_image(model_name, data), name))
        except Exception as e:
            logger.warning(f"Skipping unprocessable file {name}: {e}")
            failures.append({"filename": name, "error": str(e)})

    if not items:
        return [], failures

    # Submit everything first, then wait: one multi-image request becomes as
    # few backend calls as the batch size allows.
    batcher.submit(items)

    results = []
    for item in items:
        try:
            results.append({"filename": item.filename,
                            "top": build_top_k(batcher.wait(item), top_k)})
        except Exception as e:
            failures.append({"filename": item.filename, "error": str(e)})

    return results, failures


def payloads_from_paths(file_paths: List[str]):
    """Yield (filename, bytes) for readable image paths."""
    for file_path in file_paths:
        p = Path(file_path)
        if not p.is_file():
            logger.warning(f"File not found: {file_path}")
            continue
        if not allowed_file(p.name):
            logger.warning(f"Unsupported file type: {file_path}")
            continue
        yield p.name, p.read_bytes()


# ==========================================================
# Flask App
# ==========================================================
app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"status": "error",
                    "error": f"Request exceeds the {MAX_FILE_SIZE // 1024 // 1024} MB limit"}), 413


# ==========================================================
# Routes
# ==========================================================
@app.route("/")
def index():
    """Web UI: upload -> select -> configure -> predict -> CSV download."""
    return render_template("flask_infer_frontend.html")


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "backend": BACKEND.name,
        "artifacts_dir": str(ARTIFACTS_DIR),
        "models_available": BACKEND.available_models(),
        "models_cached": BACKEND.cached_models(),
        "batchers_alive": {name: b.alive for name, b in BATCHERS.items()},
    })


@app.route("/model-info", methods=["GET"])
def model_info():
    """Model input/output signature. Query param: ?model=cnn"""
    model_name = request.args.get("model", "cnn").lower()
    try:
        return jsonify({"model": model_name, "backend": BACKEND.name,
                        "signature": BACKEND.describe(model_name)})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/upload", methods=["POST"])
def upload_files():
    """Store uploaded images (field: files) or ZIP archives server-side.

    Filenames are sanitized (secure_filename) and ZIP members are checked
    against path traversal before extraction.
    """
    try:
        if "files" not in request.files:
            return jsonify({"status": "error", "error": "No files field in the request"}), 400

        saved_files: List[str] = []
        skipped: List[str] = []

        for file in request.files.getlist("files"):
            raw_name = file.filename or ""
            if not raw_name:
                continue
            is_zip = raw_name.lower().endswith(".zip")
            if not is_zip and not allowed_file(raw_name):
                skipped.append(raw_name)
                continue

            filename = secure_filename(raw_name)
            if not filename:
                skipped.append(raw_name)
                continue

            if is_zip:
                try:
                    with zipfile.ZipFile(io.BytesIO(file.read())) as zf:
                        safe_extract_zip(zf, UPLOAD_DIR, accept=allowed_file)
                        extracted = [
                            str(UPLOAD_DIR / name) for name in zf.namelist()
                            if allowed_file(name) and (UPLOAD_DIR / name).is_file()
                        ]
                    saved_files.extend(extracted)
                    logger.info(f"Extracted {len(extracted)} image(s) from ZIP")
                except Exception as e:
                    logger.warning(f"ZIP extraction failed for {raw_name}: {e}")
                    skipped.append(raw_name)
            else:
                save_path = UPLOAD_DIR / filename
                file.save(save_path)
                saved_files.append(str(save_path))

        return jsonify({
            "status": "success",
            "message": f"Saved {len(saved_files)} file(s)",
            "files": saved_files,
            "skipped": skipped,
        })
    except Exception as e:
        logger.exception("Upload failed")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/uploaded_files", methods=["GET"])
def get_uploaded_files():
    """List image files currently stored in the upload directory."""
    try:
        files = [
            {"name": p.name, "path": str(p), "size": p.stat().st_size}
            for p in sorted(UPLOAD_DIR.iterdir())
            if p.is_file() and allowed_file(p.name)
        ]
        return jsonify({"status": "success", "files": files})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/predict", methods=["POST"])
@collect_garbage
def predict():
    """Inference on uploaded images: multipart 'image' field(s) or a 'zip'.

    Form fields: model (default cnn), top_k (default 3).
    """
    try:
        if not request.files:
            return jsonify({"status": "error", "error": "No uploaded files"}), 400

        model_name = request.form.get("model", "cnn").lower()
        top_k = int(request.form.get("top_k", 3))

        if "image" in request.files:
            valid_files = [
                f for f in request.files.getlist("image")
                if f.filename and allowed_file(f.filename)
            ]
            if not valid_files:
                return jsonify({"status": "error", "error": "No valid image files"}), 400
            payloads = [(f.filename, f.read()) for f in valid_files]

        elif "zip" in request.files:
            zip_file = request.files["zip"]
            if not (zip_file.filename or "").lower().endswith(".zip"):
                return jsonify({"status": "error", "error": "Only ZIP files accepted here"}), 400
            with zipfile.ZipFile(io.BytesIO(zip_file.read())) as zf:
                payloads = [(n, zf.read(n)) for n in zf.namelist() if allowed_file(n)]
            if not payloads:
                return jsonify({"status": "error", "error": "ZIP contains no images"}), 400

        else:
            return jsonify({
                "status": "error",
                "error": "Upload with field name 'image' (images) or 'zip' (archive)",
            }), 400

        results, failures = run_batched(model_name, payloads, top_k)
        if not results:
            return jsonify({"status": "error", "error": "No image could be processed",
                            "details": failures}), 400

        payload = {
            "status": "success",
            "backend": BACKEND.name,
            "model": model_name,
            "total": len(results),
            "csv_url": url_for("download_result",
                               filename=save_results_to_csv(results), _external=True),
            "results": results,
        }
        if failures:
            payload["failed"] = failures
        return jsonify(payload)

    except Exception as e:
        logger.exception("Prediction request failed")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/predict_from_dir", methods=["POST"])
@collect_garbage
def predict_from_dir():
    """Inference on server-side file paths (used by the Web UI after /upload).

    JSON body: {model_name, top_k, files: [<paths>]}
    Paths are restricted to the upload directory.
    """
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"status": "error", "error": "Empty JSON body"}), 400

        model_name = str(data.get("model_name", "cnn")).lower()
        top_k = int(data.get("top_k", 3))
        file_paths = data.get("files", [])
        if not file_paths:
            return jsonify({"status": "error", "error": "No file paths provided"}), 400

        # Security: only serve files inside the upload directory
        upload_root = str(UPLOAD_DIR.resolve())
        safe_paths = [
            p for p in file_paths
            if str(Path(p).resolve()).startswith(upload_root + os.sep)
        ]
        rejected = len(file_paths) - len(safe_paths)
        if rejected:
            logger.warning(f"Rejected {rejected} path(s) outside the upload directory")
        if not safe_paths:
            return jsonify({"status": "error",
                            "error": "All paths rejected (must be inside the upload dir)"}), 400

        logger.info(f"Processing {len(safe_paths)} file(s) with model {model_name} "
                    f"[backend={BACKEND.name}]")
        results, failures = run_batched(model_name, payloads_from_paths(safe_paths), top_k)
        if not results:
            return jsonify({"status": "error", "error": "No file could be processed",
                            "details": failures}), 400

        payload = {
            "status": "success",
            "backend": BACKEND.name,
            "model": model_name,
            "total": len(results),
            "csv_url": url_for("download_result",
                               filename=save_results_to_csv(results), _external=True),
            "results": results,
        }
        if failures:
            payload["failed"] = failures
        return jsonify(payload)

    except Exception as e:
        logger.exception("predict_from_dir failed")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/download/<path:filename>")
def download_result(filename: str):
    """Download a result CSV produced by /predict or /predict_from_dir."""
    return send_from_directory(RESULT_DIR, filename, as_attachment=True)


# ==========================================================
# Entry Point
# ==========================================================
if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    logger.info(f"Starting CIFAR-10 inference service [backend={BACKEND.name}]")
    logger.info(f"Artifacts dir: {ARTIFACTS_DIR}")
    logger.info(f"Models available: {BACKEND.available_models()}")
    logger.info(f"Upload dir:    {UPLOAD_DIR}")
    logger.info(f"Result dir:    {RESULT_DIR}")
    logger.info(f"Listening on:  http://{host}:{port}")
    # threaded=True is required for dynamic batching: concurrent requests are
    # what the batcher merges into single backend calls.
    # debug is never enabled by default (the Werkzeug debugger allows RCE).
    app.run(host=host, port=port, debug=os.environ.get("FLASK_DEBUG") == "1", threaded=True)
