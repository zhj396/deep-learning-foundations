# Inference Service — One App, Two Backends

Single Flask service for the models exported by [`../train/`](../train/), with a
**pluggable serving backend** selected at startup:

| `BACKEND` | Runtime | Model file | Notes |
|---|---|---|---|
| `onnx` (default) | ONNX Runtime (CPU-optimized) | `artifacts/<model>/model.onnx` | **Zero TensorFlow import** — deployable to TF-free edge devices |
| `tf` | TensorFlow `saved_model` | `artifacts/<model>/saved_model/` | Native Keras graph, `serving_default` signature |

Both backends serve the *same* artifacts and produce the same predictions
up to float rounding, so you can A/B the runtimes without touching the model.

## Model Layout (deployment contract)

```
cifar10/artifacts/
└── <model_name>/
    ├── saved_model/      ← BACKEND=tf
    ├── model.onnx        ← BACKEND=onnx (NHWC float32, dynamic batch, opset 17)
    └── model.keras / model.tflite / config.json
```

Override the location with `ARTIFACTS_DIR` (default `../artifacts`).

## Install & Run

```bash
# ONNX backend (no TensorFlow needed)
pip install onnxruntime flask flask-cors pillow numpy
BACKEND=onnx python app.py            # http://127.0.0.1:5000

# TF backend
pip install tensorflow flask flask-cors pillow numpy
BACKEND=tf python app.py
```

Environment: `BACKEND` (onnx|tf), `HOST`, `PORT`, `ARTIFACTS_DIR`, `FLASK_DEBUG=1`.
Web UI: http://localhost:5000

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web UI (upload → select → predict → CSV) |
| GET | `/health` | Backend name, cached/available models, batcher liveness |
| GET | `/model-info?model=cnn` | Input/output signature of the active backend |
| POST | `/upload` | Store images/ZIP server-side (field: `files`) |
| GET | `/uploaded_files` | List stored files |
| POST | `/predict` | Inference (fields: `image` or `zip`, plus `model`, `top_k`) |
| POST | `/predict_from_dir` | Inference on uploaded paths (JSON: `model_name`, `top_k`, `files`) |
| GET | `/download/<name>` | Download a result CSV |

## curl Examples

```bash
curl http://localhost:5000/health | jq
curl "http://localhost:5000/model-info?model=cnn" | jq

# Single image
curl -X POST -F "model=cnn" -F "image=@cat.png" -F "top_k=3" \
     http://localhost:5000/predict | jq

# Multiple images in one request (batched into as few backend calls as possible)
curl -X POST -F "model=cnn" -F "image=@0.png" -F "image=@1.png" \
     http://localhost:5000/predict | jq

# ZIP batch
curl -X POST -F "model=cnn" -F "zip=@images.zip" http://localhost:5000/predict | jq

# Web-UI flow: upload, then predict on server-side paths
curl -X POST -F "files=@cat.png" http://localhost:5000/upload | jq
curl -X POST -H "Content-Type: application/json" \
     -d '{"model_name":"cnn","top_k":3,"files":["/tmp/cifar10_inference_uploads/cat.png"]}' \
     http://localhost:5000/predict_from_dir | jq
```

Per-file failures are reported inline (`failed`) instead of failing the request.

## Dynamic Batching — How It Works

```
request A ─┐                       ┌─ backend call (batch of N)
request B ─┼─► queue ─► worker ────┤   every ~5 ms poll window,
request C ─┘  (mutex)   thread     └─ results routed back via per-item Events
```

- Flask runs `threaded=True`: concurrent requests enqueue `RequestItem`s and
  block on per-item `threading.Event`s; the worker drains up to 32 items per call.
- One multi-image request submits **all** its items before waiting → intra-request
  batching, not serialized single-image calls.
- **Failure safety**: inference exceptions attach to the affected
  items and their events are always set → immediate errors, no hangs; a 30 s wait
  timeout is the second safety net; `/health` reports liveness and a dead worker
  is recreated on the next request.

## Preprocessing Parity (train/serve contract)

Implemented once, in pure NumPy ([preprocessing.py](preprocessing.py)),
mirroring the training pipeline:

| Model | Size | NumPy implementation | Equivalent Keras function |
|-------|------|----------------------|---------------------------|
| `cnn`, `resnet` | 32×32 | `x / 255` | — (dataset.py plain scaling) |
| `resnet50` | 224×224 | RGB→BGR, subtract `[103.939, 116.779, 123.680]` | `resnet50.preprocess_input` (caffe mode, exact match) |
| `efficientnetb0` | 224×224 | **pass-through** — feed raw `[0, 255]` | `efficientnet.preprocess_input` (documented no-op placeholder) |

> Note: Keras bakes EfficientNet's normalization **into the model
> graph** (`Rescaling(1/255)` + imagenet stddev correction, see
> `keras/src/applications/efficientnet.py`), so its `preprocess_input` is a
> pass-through and the exported ONNX/SavedModel expect raw `[0, 255]` input.
> ResNet50 has NO internal preprocessing — serving must do the caffe-style mean
> subtraction itself. Preprocessing must match the training pipeline exactly,
> otherwise accuracy degrades silently.

## Security Notes

- ZIP extraction is zip-slip protected (`filter='data'` + containment check)
- Upload filenames sanitized with `secure_filename`
- `/predict_from_dir` only serves paths inside the upload directory
- Request size capped (`MAX_CONTENT_LENGTH`, JSON 413 handler)
- `debug=True` never enabled by default (Werkzeug debugger = RCE risk)
