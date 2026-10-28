# CIFAR-10 Training — Multi-Backbone Classification (TensorFlow / Keras 3)

Training pipeline for the CIFAR-10 project: data loading, augmentation (incl. MixUp/CutMix),
four switchable backbones, LR scheduling, two-phase transfer learning, multi-GPU training,
and deployment export (.keras / SavedModel / TFLite / **ONNX**).

## Project Structure

```
train/
├── config.py         # Dataclass config: CLI args, YAML/JSON files, .env, seeding
├── dataset.py        # CIFAR-10 loading, stratified split, tf.data pipeline
├── data_aug.py       # Augmentation builder + graph-safe MixUp/CutMix
├── model.py          # CIFARModel factory: cnn / resnet / resnet50 / efficientnetb0
├── callbacks.py      # TrainingMonitor (logger), ModelCheckpoint, EarlyStopping
├── scheduler.py      # Cosine-decay-restarts or ReduceLROnPlateau
├── utils.py          # Training curves + prediction previews (headless, Agg backend)
├── export_onnx.py    # SavedModel -> ONNX (also runnable standalone)
├── train.py          # Entry point: train -> fine-tune -> evaluate -> export
└── README.md
```

Outputs (paths anchored to `cifar10/`, independent of the cwd; on Colab/Kaggle
anchored to `/content` resp. `/kaggle/working`):

```
cifar10/
├── checkpoints/<model>/              # intermediate training files
│   ├── best.keras                    # best checkpoint (by val_accuracy)
│   ├── figures/                      # accuracy.png, loss.png, predictions.png
│   └── logs/train.log                # full training log
└── artifacts/<model>/                # deployment-ready exports
    ├── model.keras                   # native Keras format
    ├── saved_model/                  # TF SavedModel directory
    ├── model.tflite
    ├── model.onnx
    └── config.json                   # exact resolved config for reproducibility
```

## Environment

- Python ≥ 3.10 (Keras 3 era TensorFlow)
- TensorFlow ≥ 2.18, plus: `pip install -r ../requirements.txt`
- Runs on CPU, single GPU, or multi-GPU (auto-detected)
- CI: GitHub Actions (`.github/workflows/ci.yml`) — flake8 + CPU smoke tests
  (`test_smoke.py`, no dataset/weights download) on Python 3.11/3.12, on pushes
  to main and on PRs (plus manual `workflow_dispatch` runs)

## Quick Start

```bash
# Pipeline smoke test (CPU-friendly): 1 batch/epoch, exports all formats
python train.py --mode sanity --model-type cnn --epochs 2

# From-scratch models
python train.py --model-type cnn    --epochs 50 --horizontal-flip --use-mix
python train.py --model-type resnet --epochs 50 --horizontal-flip --use-mix

# Transfer learning: frozen backbone first, then fine-tune the last 20% of layers
python train.py --model-type resnet50       --epochs 30 --fine-epochs 20 --fine-tune-ratio 0.2 --horizontal-flip
python train.py --model-type efficientnetb0 --epochs 30 --fine-epochs 20 --fine-tune-ratio 0.2 --horizontal-flip
```

Full CLI reference: `python train.py --help`. Config files also work:
`python train.py --config my_config.yaml`.

### Data loading (local-first, download fallback)

CIFAR-10 is resolved in this order on every run:

1. `cifar10/data/cifar-10-batches-py/` — extracted batch files (fastest)
2. `cifar10/data/cifar-10-python.tar.gz` — local archive, extracted in place
3. Online download via `tf.keras.utils.get_file` (cached by filename into
   `~/.keras/datasets`, so the archive is fetched only once); it is then
   copied to `cifar10/data/` and extracted in place, so later runs resolve
   through steps 1–2 with no network access.

To seed an offline machine, copy `cifar-10-python.tar.gz` into `cifar10/data/`.
`data/` and `*.tar.gz` are gitignored.

## Key Design Points

### Reproducibility

- `seed=42` by default → identical **stratified** train/val split, weight init, and shuffling
  across backbones and runs.
- `tf.config.experimental.enable_op_determinism()` is enabled with the seed
  (comment out in `config.py::_set_seed` to trade determinism for GPU speed).
- The resolved config is saved next to the exported model (`artifacts/<model>/config.json`).

### tf.data pipeline order

`slice uint8 → shuffle → resize/normalize → augment → batch → mix`:
shuffling happens on raw 3 KB uint8 images, **before** the resize to 224×224,
so the shuffle buffer holds 3 KB uint8 images instead of ~600 KB resized
float32 images for the transfer-learning models.

### Augmentation inside tf.data

- Random* layers are called with explicit `training=True` inside `map()`.
- MixUp/CutMix uses `tf.cond` + stateless gamma-ratio beta sampling (no NumPy RNG),
  so λ is re-drawn per batch.
- Brightness assumes [0, 1]-scaled images (RandomBrightness gets
  `value_range=(0, 1)`); it is skipped for the ImageNet-pretrained backbones,
  whose preprocessing does not produce unit-scaled inputs (resnet50:
  mean-subtracted; efficientnetb0: raw [0, 255] with normalization baked
  into the graph). Contrast is scale-relative and applies to all backbones.

### Two-phase transfer learning

1. **Phase 1** — backbone frozen (`training=False` keeps pretrained BatchNorm
   statistics), only the head trains at `init_lr`.
2. **Phase 2** (`--fine-tune-ratio` / `--fine-tune-at`) — unfreeze the top fraction of
   backbone layers at `fine_tune_lr`; **BatchNorm layers stay frozen** everywhere.
   Phase 2 uses fresh callbacks and the **same loss**
   (incl. label smoothing) so monitored values remain comparable across phases.

### Multi-GPU (MirroredStrategy)

With 2+ visible GPUs (e.g. Kaggle T4 ×2), training automatically runs under
`tf.distribute.MirroredStrategy`: synchronous data parallelism, variables mirrored
per device, gradients all-reduced each step. `--batch-size` is the **global** batch
(split evenly across replicas). Single GPU / CPU falls back to the default strategy
with no code-path differences.

### LR scheduling

| `--lr-mode` | Behavior |
|---|---|
| `plateau` (default) | `ReduceLROnPlateau` on `val_loss` (factor 0.5, patience 3) |
| `cosine` | `CosineDecayRestarts`: cycle = `--cosine-restart-epochs` (default 10), each next cycle 2× longer, peak decays 10%/cycle, floor `min_lr` |

Checkpoint/EarlyStopping monitor `val_accuracy`; ReduceLROnPlateau monitors
`val_loss`.

## Configuration Parameters (defaults)

| Parameter | Default | Description |
|---|---|---|
| `seed` | 42 | Reproducibility seed (None disables) |
| `val_split_ratio` | 0.2 | Stratified validation split |
| `model_type` | cnn | `cnn` / `resnet` / `resnet50` / `efficientnetb0` |
| `input_shape` | auto | 32×32×3 (cnn/resnet), 224×224×3 (backbones); override with `--input-shape` |
| `batch_size` | 128 | Global batch size |
| `epochs` / `patience` | 30 / 3 | Training length / LR-reduction patience |
| `label_smoothing` | 0.1 | Applied in **both** training phases |
| `lr_mode` / `init_lr` / `min_lr` / `factor` | plateau / 1e-3 / 1e-6 / 0.5 | LR schedule |
| `mode` | full | `sanity` = 1-batch smoke test |
| `fine_epochs` / `fine_tune_lr` | 20 / 1e-5 | Phase-2 length / LR |
| `fine_tune_ratio` / `fine_tune_at` | None | Unfreeze last X% / from layer index |
| `use_mix` | False | MixUp + CutMix (batch level) |
| `augmentation` | None | Dict via `--horizontal-flip`, `--rotation`, ... flags |

Recommended augmentation for CIFAR-10: `--horizontal-flip` (+ small `--width-shift 0.1
--height-shift 0.1`); vertical flips and large rotations hurt (CIFAR has a canonical
up direction).

## Fine-tuning Reference

| Backbone | Layers | Recommended unfreeze | LR |
|---|---|---|---|
| ResNet50 | ~175 | last 20–30% (`--fine-tune-ratio 0.2`) | 1e-5 |
| EfficientNetB0 | ~237 | last 20–25% | 1e-5 |

## ONNX Export

`train.py` exports ONNX automatically. To convert an existing SavedModel later
(e.g. trained on Kaggle, downloaded locally):

```bash
python export_onnx.py --model-type cnn                      # uses ../artifacts/cnn/saved_model
python export_onnx.py --saved-model /path/to/saved_model --output /path/model.onnx
```

I/O contract: NHWC float32, dynamic batch — identical to the SavedModel.

## Visualization Outputs

- `checkpoints/<model>/figures/accuracy.png`, `loss.png` — train/val curves (smoothed; raw values in `logs/train.log`)
- `checkpoints/<model>/figures/predictions.png` — first 10 test images with true/pred labels + confidence
