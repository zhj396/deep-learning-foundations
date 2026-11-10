# -*- coding: utf-8 -*-
"""pt->en translation service (single app, pluggable backends)
==============================================================
Serves checkpoints trained by ../train through the shared artifacts
directory (default ../artifacts, override with ARTIFACTS_DIR):

    artifacts/<model>/checkpoint.pt                BACKEND=torch (default)
    artifacts/<model>/{encoder,decoder_step}.onnx  BACKEND=onnx (torch-free,
                                                   exported by ../train/export_onnx.py)

Features:
- Web UI (text or .txt file -> translation) at /
- Dynamic batching: sentences from concurrent requests are merged into
  single decode calls (batcher.py), with per-item errors + timeouts;
  autoregressive decoding amortises especially well over batches
- Tokenization mirrors training exactly (tokenization.py)
- Greedy on both backends; beam search on the torch backend
- Security: model-name whitelist, secure_filename, request size limit,
  debug off by default

Endpoints:
    GET  /                  Web UI
    GET  /health            Backend, models, batcher liveness
    GET  /model-info?model= Decode signature
    POST /translate         JSON {"text"|"texts", model, mode, ...} (or form field "text")
    POST /translate_file    Multipart .txt, one sentence per line (field: file)
    GET  /download/<name>   Result file produced by /translate_file
"""

import logging
import os
import re
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from flask import Flask, jsonify, render_template, request, send_from_directory, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename

from backends import ARTIFACTS_DIR, DecodeParams, get_backend
from batcher import DynamicBatcher, RequestItem
from tokenization import Translator

# ==========================================================
# Configuration
# ==========================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("inference")

MAX_FILE_SIZE = 10 * 1024 * 1024   # 10 MB per request (text only)
MAX_LINES = 10_000                 # per /translate_file request
DEFAULT_MAX_LEN = int(os.environ.get("MAX_LEN", "128"))
# model names are joined into artifact paths: strict whitelist, no traversal
MODEL_NAME_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")

BACKEND = get_backend(os.environ.get("BACKEND", "torch"))

RESULT_DIR = Path(tempfile.gettempdir()) / "pten_inference_results"
RESULT_DIR.mkdir(exist_ok=True)

# One batcher per (model, decode params); one Translator per model. Both are
# lazily created and guarded by locks (Flask runs threaded).
BATCHERS: Dict[tuple, DynamicBatcher] = {}
TRANSLATORS: Dict[str, Translator] = {}
_REGISTRY_LOCK = threading.Lock()


# ==========================================================
# Helpers
# ==========================================================
def get_translator(model_name: str) -> Translator:
    with _REGISTRY_LOCK:
        translator = TRANSLATORS.get(model_name)
        if translator is None:
            spm_path = ARTIFACTS_DIR / model_name / "spm_en_pt.model"
            translator = Translator(sp_model_path=spm_path)
            TRANSLATORS[model_name] = translator
        return translator


def get_batcher(model_name: str, params: DecodeParams) -> DynamicBatcher:
    """Return the (lazily created) batcher for these exact decode settings,
    recreating it if its worker thread has died for any reason."""
    key = (model_name, params.mode, params.max_len,
           params.beam_size, params.length_penalty)
    with _REGISTRY_LOCK:
        batcher = BATCHERS.get(key)
        if batcher is not None and batcher.alive:
            return batcher
        if batcher is not None:
            logger.error(f"Batcher worker for {key} died; recreating")
        translator = TRANSLATORS[model_name]  # set by run_batched before this call
        BATCHERS[key] = DynamicBatcher(
            lambda arr: BACKEND.translate(model_name, arr, params),
            pad_idx=translator.pad_idx,
        )
        return BATCHERS[key]


def params_from_dict(data: dict) -> Tuple[DecodeParams, str, str]:
    """Validate/extract (params, model_name, error) from a request payload."""
    model_name = str(data.get("model", "base")).lower()
    if not MODEL_NAME_RE.fullmatch(model_name):
        return None, model_name, f"Invalid model name: {model_name!r}"

    mode = str(data.get("mode", "greedy")).lower()
    if mode not in ("greedy", "beam"):
        return None, model_name, f"Invalid mode: {mode!r} (greedy | beam)"
    if mode == "beam" and BACKEND.name == "onnx":
        return None, model_name, "Beam search needs BACKEND=torch (the onnx backend is greedy-only)"

    try:
        params = DecodeParams(
            mode=mode,
            max_len=int(data.get("max_len", DEFAULT_MAX_LEN)),
            beam_size=int(data.get("beam_size", 4)),
            length_penalty=float(data.get("length_penalty", 0.6)),
        )
    except (TypeError, ValueError) as e:
        return None, model_name, f"Invalid decode parameter: {e}"

    if not 2 <= params.max_len <= 1024:
        return None, model_name, "max_len must be within [2, 1024]"
    if params.mode == "beam" and not 1 <= params.beam_size <= 16:
        return None, model_name, "beam_size must be within [1, 16]"

    return params, model_name, None


def run_batched(model_name: str, texts: List[str], params: DecodeParams):
    """Encode, submit to the dynamic batcher, decode; order-preserving.

    Returns (results, failures) where results carry {"index", "text",
    "translation"} and failures {"index", "text", "error"}.
    """
    translator = get_translator(model_name)
    # special-token ids come from the spm model, never from the request
    params = DecodeParams(**{**params.__dict__, **translator.decode_params()})

    if model_name not in BACKEND.available_models():
        raise FileNotFoundError(
            f"No {BACKEND.name} model {model_name!r}. Available: "
            f"{BACKEND.available_models() or 'none'}"
        )

    batcher = get_batcher(model_name, params)

    items: List[Tuple[int, RequestItem]] = []
    failures: List[Dict] = []
    for i, text in enumerate(texts):
        if not text or not text.strip():
            failures.append({"index": i, "text": "", "error": "empty sentence"})
            continue
        items.append((i, RequestItem(translator.encode(text), text)))

    if items:
        # submit everything first, then wait: one multi-sentence request
        # becomes as few decode calls as the batch size allows
        batcher.submit([item for _, item in items])

    results: List[Dict] = []
    for i, item in items:
        try:
            ids = batcher.wait(item)
            results.append({"index": i, "text": item.text,
                            "translation": translator.decode(ids)})
        except Exception as e:
            failures.append({"index": i, "text": item.text[:200], "error": str(e)})

    results.sort(key=lambda r: r["index"])
    failures.sort(key=lambda f: f["index"])
    return results, failures


def save_results_file(results: List[Dict], failures: List[Dict], n_lines: int) -> str:
    """Write line-aligned translations (failures as empty lines); returns the file name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"translations_{timestamp}_{uuid.uuid4().hex[:6]}.txt"
    by_index = {r["index"]: r["translation"] for r in results}
    lines = [by_index.get(i, "") for i in range(n_lines)]
    (RESULT_DIR / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return filename


# ==========================================================
# Flask App
# ==========================================================
app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE
app.config["JSON_SORT_KEYS"] = False


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"status": "error",
                    "error": f"Request exceeds the {MAX_FILE_SIZE // 1024 // 1024} MB limit"}), 413


# ==========================================================
# Routes
# ==========================================================
@app.route("/")
def index():
    """Web UI: text or file -> translation."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "backend": BACKEND.name,
        "modes": ["greedy"] if BACKEND.name == "onnx" else ["greedy", "beam"],
        "artifacts_dir": str(ARTIFACTS_DIR),
        "models_available": BACKEND.available_models(),
        "models_cached": BACKEND.cached_models(),
        "batchers_alive": {"|".join(map(str, k)): b.alive for k, b in BATCHERS.items()},
        "default_max_len": DEFAULT_MAX_LEN,
    })


@app.route("/model-info", methods=["GET"])
def model_info():
    """Decode signature. Query param: ?model=base"""
    model_name = request.args.get("model", "base").lower()
    if not MODEL_NAME_RE.fullmatch(model_name):
        return jsonify({"status": "error", "error": f"Invalid model name: {model_name!r}"}), 400
    try:
        return jsonify({"model": model_name, "backend": BACKEND.name,
                        "signature": BACKEND.describe(model_name)})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/translate", methods=["POST"])
def translate():
    """Translate text. JSON body: {"text": str | "texts": [str], "model", "mode", ...}
    (a form field "text" also works for quick curl checks)."""
    t0 = time.time()
    try:
        data = request.get_json(silent=True)
        if data is None:
            form_text = request.form.get("text")
            if not form_text:
                return jsonify({"status": "error",
                                "error": "Send JSON {'text': ...} or a form field 'text'"}), 400
            data = {"text": form_text, **request.form.to_dict()}

        texts = data.get("texts")
        if texts is None and data.get("text") is not None:
            texts = [data["text"]]
        if not texts or not isinstance(texts, list):
            return jsonify({"status": "error",
                            "error": "Provide 'text' (string) or 'texts' (list of strings)"}), 400
        texts = [str(t) for t in texts]

        params, model_name, err = params_from_dict(data)
        if err:
            return jsonify({"status": "error", "error": err}), 400

        results, failures = run_batched(model_name, texts, params)
        if not results:
            return jsonify({"status": "error", "error": "No sentence could be translated",
                            "details": failures}), 400

        payload = {
            "status": "success",
            "backend": BACKEND.name,
            "model": model_name,
            "mode": params.mode,
            "total": len(results),
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "translations": results,
        }
        if failures:
            payload["failed"] = failures
        return jsonify(payload)

    except Exception as e:
        logger.exception("Translation request failed")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/translate_file", methods=["POST"])
def translate_file():
    """Translate a .txt file (one sentence per line). Multipart field: file.
    Form fields: model, mode, max_len, beam_size, length_penalty."""
    t0 = time.time()
    try:
        if "file" not in request.files:
            return jsonify({"status": "error", "error": "No file field in the request"}), 400
        upload = request.files["file"]
        raw_name = upload.filename or ""
        if not raw_name.lower().endswith(".txt"):
            return jsonify({"status": "error", "error": "Only .txt files are accepted"}), 400
        filename = secure_filename(raw_name)

        content = upload.read().decode("utf-8", errors="replace")
        lines = [line.strip() for line in content.splitlines()]
        if len(lines) > MAX_LINES:
            return jsonify({"status": "error",
                            "error": f"File has {len(lines)} lines; the limit is {MAX_LINES}"}), 400

        params, model_name, err = params_from_dict(request.form.to_dict())
        if err:
            return jsonify({"status": "error", "error": err}), 400

        results, failures = run_batched(model_name, lines, params)
        if not results:
            return jsonify({"status": "error", "error": "No line could be translated",
                            "details": failures}), 400

        out_name = save_results_file(results, failures, len(lines))
        payload = {
            "status": "success",
            "backend": BACKEND.name,
            "model": model_name,
            "mode": params.mode,
            "source_file": filename,
            "total": len(results),
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "download_url": url_for("download_result", filename=out_name, _external=True),
            "preview": results[:50],
        }
        if failures:
            payload["failed"] = failures[:50]
        return jsonify(payload)

    except Exception as e:
        logger.exception("translate_file failed")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/download/<path:filename>")
def download_result(filename: str):
    """Download a result file produced by /translate_file."""
    return send_from_directory(RESULT_DIR, filename, as_attachment=True)


# ==========================================================
# Entry Point
# ==========================================================
if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    logger.info(f"Starting pt->en translation service [backend={BACKEND.name}]")
    logger.info(f"Artifacts dir: {ARTIFACTS_DIR}")
    logger.info(f"Models available: {BACKEND.available_models()}")
    logger.info(f"Result dir:    {RESULT_DIR}")
    logger.info(f"Listening on:  http://{host}:{port}")
    # threaded=True is required for dynamic batching: concurrent requests are
    # what the batcher merges into single decode calls.
    # debug is never enabled by default (the Werkzeug debugger allows RCE).
    app.run(host=host, port=port, debug=os.environ.get("FLASK_DEBUG") == "1", threaded=True)
