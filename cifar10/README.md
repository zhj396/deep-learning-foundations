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
checkpoint by `val_accuracy`, seed 42. Recorded on Kaggle Tesla T4 ×2
(MirroredStrategy, 2 replicas, global batch 128):

| Backbone | Recipe | Test Acc |
|---|---|---|
| `cnn` | scratch, 50-epoch cap, flip + MixUp/CutMix | 86.78% |
| `resnet` | scratch, 50-epoch cap, flip + MixUp/CutMix | 86.67% |
| `resnet50` | transfer, frozen 30 epochs + fine-tune top 20% | 94.53% |
| `efficientnetb0` | transfer, frozen 30 epochs + fine-tune top 20% | 94.35% |

The two from-scratch backbones plateau at ~86.7% regardless of architecture
size (0.55M vs 0.73M parameters). ImageNet transfer reaches ~94.4%, with the
frozen-backbone phase and the fine-tune phase each contributing roughly half
of the gain over from-scratch training. Per-model early stopping, validation
curves, and the ONNX re-evaluation of all four exported models are in
[train/README.md](train/README.md).

## Quick Start

```bash
# One-time cleanup: remove any pre-existing/duplicate TF installs BEFORE
# installing requirements (pip install never removes them, and leftover
# copies cause the cuFFT/cuDNN/cuBLAS "factory already registered" errors
# and mixed-version tf.data warnings).
pip uninstall -y tensorflow tensorflow-gpu tensorflow-cpu keras tf-keras

# Installs the training stack with the version floors from requirements.txt:
# TensorFlow >= 2.18 + Keras >= 3.10 (Linux, CUDA extra). Keras 3.7-3.9 have
# a MirroredStrategy bug that reports training loss and accuracy as NaN
# mid-epoch while the model keeps training normally (validation metrics stay
# correct), so multi-GPU runs need >= 3.10. See keras-team/keras#21467.
pip install -r requirements.txt

# Verify both packages (floors: TF 2.18.x+, Keras 3.10+)
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
  install the plain CPU build (`tensorflow>=2.18`) — Windows-native TensorFlow
  has no GPU support; use WSL2, which follows the Linux path, for CUDA.
- On Kaggle the preinstalled TensorFlow should NOT be replaced. Print both
  versions in the notebook's first cell
  (`python -c "import tensorflow as tf, keras; print(tf.__version__, keras.__version__)"`);
  if Keras falls in the 3.7-3.9 range, install a newer one there
  (`!pip install -q "keras>=3.10"` — session-scoped, so keep it in every
  session). Alternative sidestep: run single-GPU (`CUDA_VISIBLE_DEVICES=0`),
  which avoids the MirroredStrategy NaN-metrics bug entirely.

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

- **Version floor (multi-GPU)**: Keras 3.7-3.9 have a MirroredStrategy bug
  where *reported* training loss/accuracy are NaN mid-epoch while validation
  metrics stay correct and the model keeps training normally
  ([keras#21467](https://github.com/keras-team/keras/issues/21467)). The mirrored
  `keras.metrics.Mean` state (`total`/`count`) overflows to NaN when summed across
  replicas. Fix: the `keras>=3.10` floor in `requirements.txt` (see Quick Start),
  or run single-GPU. Checkpointing/early stopping/LR reduction all monitor
  `val_*` metrics, so an affected run's artifacts remain trustworthy — only the
  train-side log lines and training curves are garbage. Verify both packages with
  `python -c "import tensorflow as tf, keras; print(tf.__version__, keras.__version__)"`
  (the `keras` package alone can be the culprit).
- **Kaggle pip installs are session-scoped**: `!pip install -q "keras>=3.10"`
  only affects the current notebook session. Keep it in the notebook's first cell
  so every run gets it, and launch training *after* that cell executes.
- **Benign startup noise**: the cuFFT/cuDNN/cuBLAS "factory already registered"
  errors and the `use_unbounded_threadpool` MapDataset NodeDef warning
  (`Unknown attributes will be ignored`) appear in clean single-TF runtimes —
  they are artifacts of plugin loading and
  `ds.map(num_parallel_calls=AUTOTUNE)` graph construction. Silence them with
  `TF_CPP_MIN_LOG_LEVEL=3` if desired.
