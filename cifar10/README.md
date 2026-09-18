# CIFAR-10: Multi-Backbone Classification and Local Serving

CIFAR-10 classification with four switchable backbones, a reproducible training
pipeline, and a Flask serving app with pluggable TensorFlow / ONNX Runtime
backends.

```
┌────────────────────────────┐      ┌─────────────────────────┐      ┌──────────────────────────────┐
│ train/                     │      │ artifacts/<model>/      │      │ inference/                   │
│  CIFAR-10 → tf.data        │────▶│  model.keras            │────▶│  one Flask app, :5000        │
│  4 backbones, MirroredStr. │      │  saved_model/           │      |  BACKEND=tf    → SavedModel  │
│  MixUp/CutMix, 2-phase TL  │      │  model.onnx ────────────┼────▶│  BACKEND=onnx  → ORT (no TF) │
│  → evaluate → export       │      │  model.tflite           │      │  + dynamic batching, web UI  │
└────────────────────────────┘      │  config.json            │      └──────────────────────────────┘
                                    └─────────────────────────┘
```

## Backbones

| Model | Type | Input | Notes |
|---|---|---|---|
| `cnn` | from scratch | 32×32 | 3× (Conv-BN-ReLU ×2, pool, dropout) + MLP head |
| `resnet` | from scratch | 32×32 | custom 6-block ResNet, GAP head, L2 1e-4 |
| `resnet50` | ImageNet transfer | 224×224 | frozen backbone → fine-tune top 20–30%, BN kept frozen |
| `efficientnetb0` | ImageNet transfer | 224×224 | same two-phase recipe |

## Accuracy

Test accuracy on the official CIFAR-10 test split (10,000 images), best
checkpoint by `val_accuracy`, seed 42. *Table fills in as training runs complete.*

| Backbone | Epochs | Augmentation | Test Acc | Test Loss |
|---|---|---|---|---|
| `cnn` | 50 | flip + MixUp/CutMix | _TBD_ | |
| `resnet` | 50 | flip + MixUp/CutMix | _TBD_ | |
| `resnet50` (frozen) | 30 | flip | _TBD_ | |
| `resnet50` (+fine-tune 20%) | 30 + 20 | flip | _TBD_ | |
| `efficientnetb0` (frozen) | 30 | flip | _TBD_ | |
| `efficientnetb0` (+fine-tune 20%) | 30 + 20 | flip | _TBD_ | |

## Quick Start

```bash
# One-time cleanup: remove any pre-existing/duplicate TF installs BEFORE
# installing requirements (pip install never removes them, and leftover
# copies cause the cuFFT/cuDNN/cuBLAS "factory already registered" errors
# and mixed-version tf.data warnings).
pip uninstall -y tensorflow tensorflow-gpu tensorflow-cpu keras tf-keras

# Installs the pinned training stack: TF 2.18 + Keras 3.6 (Linux, CUDA extra).
# Do NOT upgrade Keras past 3.6 for multi-GPU runs: Keras 3.7/3.8 (bundled
# with TF >=2.19) have a MirroredStrategy bug where training loss/accuracy
# are reported as NaN mid-epoch while the model keeps training normally
# (validation metrics stay correct). See keras-team/keras#21467.
pip install -r requirements.txt

# Verify the pinned versions (expect: 2.18.x 3.6.0)
python -c "import tensorflow as tf, keras; print(tf.__version__, keras.__version__)"

# Train (multi-GPU auto-detected via MirroredStrategy)
cd train

# From-scratch backbones (32x32)
python train.py --model-type cnn    --epochs 50 --horizontal-flip --use-mix
python train.py --model-type resnet --epochs 50 --horizontal-flip --use-mix

# Transfer-learning backbones (224x224, two-phase: frozen backbone ->
# fine-tune the last 20% of layers at a lower LR)
python train.py --model-type resnet50       --epochs 30 --fine-epochs 20 --fine-tune-ratio 0.2 --horizontal-flip
python train.py --model-type efficientnetb0 --epochs 30 --fine-epochs 20 --fine-tune-ratio 0.2 --horizontal-flip

# LR schedules (--lr-mode; default is plateau)
#   plateau: ReduceLROnPlateau on val_loss — halve LR after `--patience`
#     stagnant epochs, down to `--min-lr`
#   cosine: SGDR-style cosine decay with warm restarts — `--cosine-restart-epochs`
#     per cycle, each cycle 2x longer, peak LR 0.9x per restart
python train.py --model-type cnn --epochs 50 --horizontal-flip --use-mix \
    --lr-mode cosine --cosine-restart-epochs 10 --init-lr 1e-3 --min-lr 1e-6
python train.py --model-type cnn --epochs 50 --horizontal-flip --use-mix \
    --lr-mode plateau --factor 0.5 --patience 3 --min-lr 1e-6

# Serve — ONNX Runtime backend (default; no TensorFlow dependency)
cd ../inference
BACKEND=onnx python app.py          # http://127.0.0.1:5000

# Serve — TensorFlow SavedModel backend (same app, same artifacts)
BACKEND=tf python app.py
```

Platform notes:

- `tensorflow[and-cuda]` in `requirements.txt` is Linux-only. On Windows,
  install `tensorflow==2.18.*` instead (the Windows pip package has no CUDA
  extra) plus a matching local CUDA/cuDNN if a GPU is used.
- On Kaggle the preinstalled TF (2.18.x) should NOT be replaced. Kaggle images
  may ship a newer Keras; pin it in the notebook's first cell instead
  (`!pip install -q keras==3.6.0` — session-scoped, so keep it in every
  session), then verify with
  `python -c "import tensorflow as tf, keras; print(tf.__version__, keras.__version__)"`.
  Alternative sidestep: run single-GPU (`CUDA_VISIBLE_DEVICES=0`), which avoids
  the MirroredStrategy NaN-metrics bug entirely.

Details: [train/README.md](train/README.md) · [inference/README.md](inference/README.md)

## Implementation Notes

- **Reproducibility**: seed 42 + stratified split + deterministic ops; resolved
  config saved next to every exported model.
- **tf.data order**: shuffle raw uint8 *before* resize/normalize (the shuffle
  buffer holds 3 KB uint8 images instead of ~600 KB resized float32 at
  224×224); augmentation layers called with explicit `training=True` inside
  `map()`.
- **MixUp/CutMix**: fully in TF ops (`tf.cond` + stateless gamma-ratio beta
  sampling) so the mix ratio is re-drawn per batch under graph tracing.
- **Transfer learning**: same loss across both phases, fresh callback state per
  phase, unfreeze pattern with BatchNorm kept frozen.
- **Train/serve parity**: serving preprocessing mirrors training exactly —
  caffe-style mean subtraction for ResNet50, pass-through `[0,255]` for
  EfficientNet (its normalization is baked into the model graph).
- **Serving**: dynamic batching with per-item error handling, timeouts, and
  liveness in `/health`; zip-slip-safe uploads and path confinement.

## Operational Notes

- **Version pin (multi-GPU)**: Keras 3.7/3.8 (bundled with TF >=2.19) have a
  MirroredStrategy bug where *reported* training loss/accuracy are NaN mid-epoch
  while validation metrics stay correct and the model keeps training normally
  ([keras#21467](https://github.com/keras-team/keras/issues/21467)). The mirrored
  `keras.metrics.Mean` state (`total`/`count`) overflows to NaN when summed across
  replicas. Fix: pin TF 2.18 + Keras 3.6 (see Quick Start), or run single-GPU.
  Checkpointing/early stopping/LR reduction all monitor `val_*` metrics, so an
  affected run's artifacts remain trustworthy — only the train-side log lines and
  training curves are garbage. Verify both packages with
  `python -c "import tensorflow as tf, keras; print(tf.__version__, keras.__version__)"`
  (the `keras` package alone can be the culprit).
- **Kaggle pip installs are session-scoped**: `!pip install keras==3.6.0` only
  affects the current notebook session. Keep the pin in the notebook's first cell
  so every run gets it, and launch training *after* that cell executes.
- **Benign startup noise**: the cuFFT/cuDNN/cuBLAS "factory already registered"
  errors and the `use_unbounded_threadpool` MapDataset NodeDef warning
  (`Unknown attributes will be ignored`) appear in clean single-TF runtimes —
  they are artifacts of plugin loading and
  `ds.map(num_parallel_calls=AUTOTUNE)` graph construction. Silence them with
  `TF_CPP_MIN_LOG_LEVEL=3` if desired.

*Accuracy takeaways to be completed after the training runs.*
