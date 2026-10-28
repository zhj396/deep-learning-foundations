# -*- coding: utf-8 -*-
"""
CIFAR-10 Training and Model Export Script

Features:
- tf.data pipeline with augmentation and MixUp/CutMix (see dataset.py)
- Multi-GPU data parallelism via MirroredStrategy (auto-detected, e.g. Kaggle T4 x2)
- Two-phase transfer learning for pretrained backbones:
  phase 1 trains the head with a frozen backbone, phase 2 fine-tunes the top
  fraction of backbone layers (BatchNorm layers stay frozen)
- Exports deployment artifacts to <root>/artifacts/<model_type>/:
  model.keras, saved_model/, model.tflite, model.onnx, config.json

Quick start:
    python train.py --mode sanity --model-type cnn --epochs 2   # pipeline smoke test
    python train.py --model-type cnn --epochs 50 --horizontal-flip --use-mix
"""

import os

# TensorFlow environment settings (must be set before importing TF)
os.environ.setdefault("TF_GPU_THREAD_MODE", "gpu_private")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # suppress INFO/WARNING C++ logs

import tensorflow as tf

from callbacks import create_training_callbacks
from config import Config, parse_args
from dataset import CIFARDataLoader
from export_onnx import export_saved_model_to_onnx
from model import CIFARModel
from scheduler import create_optimizer_and_scheduler
from utils import VisualizationUtils

# Layer names of the keras.applications backbones as embedded in the wrapper
# model (used to locate them for fine-tuning).
BACKBONE_LAYER_NAMES = {
    "resnet50": "resnet50",
    "efficientnetb0": "efficientnetb0",
}


def setup_tf_runtime(cfg: Config) -> tf.distribute.Strategy:
    """Configure GPU memory growth and pick a distribution strategy.

    Returns MirroredStrategy when 2+ GPUs are visible (synchronous data
    parallelism: the global batch is split across replicas, gradients are
    all-reduced each step), otherwise the default single-device strategy.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            cfg.logger.info(f"Found {len(gpus)} GPU(s); memory growth enabled")
        except RuntimeError as e:
            # Memory growth must be set before the GPU is initialized
            cfg.logger.warning(f"GPU configuration warning: {e}")
    else:
        cfg.logger.info("No GPU detected, running on CPU")

    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        cfg.logger.info(
            f"MirroredStrategy active: {strategy.num_replicas_in_sync} replicas "
            f"(global batch {cfg.batch_size} -> "
            f"{cfg.batch_size // strategy.num_replicas_in_sync} per GPU)"
        )
    else:
        strategy = tf.distribute.get_strategy()

    return strategy


def _make_loss_and_metrics(cfg: Config):
    """Loss/metrics used in BOTH training phases so monitored values stay comparable."""
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=cfg.label_smoothing)
    metrics = [tf.keras.metrics.CategoricalAccuracy(name="accuracy")]
    return loss, metrics


def _compile_model(model: tf.keras.Model, cfg: Config, optimizer) -> None:
    """Compile with the shared loss/metrics (identical in both training phases)."""
    loss, metrics = _make_loss_and_metrics(cfg)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


def _fine_tune_backbone(model: tf.keras.Model, cfg: Config) -> None:
    """Unfreeze the top fraction of the backbone for fine-tuning.

    Follows the canonical Keras transfer-learning pattern:
    1. set base_model.trainable = True (re-enables weight tracking)
    2. freeze the early layers (generic low-level features)
    3. keep BatchNormalization layers frozen everywhere: their pretrained
       moving statistics are valuable and small noisy batch updates harm them
    """
    backbone_name = BACKBONE_LAYER_NAMES[cfg.model_type.lower()]
    try:
        base_model = model.get_layer(backbone_name)
    except ValueError:
        raise RuntimeError(
            f"Could not locate backbone layer {backbone_name!r} in the model. "
            f"Fine-tuning requires a transfer-learning model_type."
        ) from None

    n_layers = len(base_model.layers)
    if cfg.fine_tune_ratio is not None:
        start_idx = int(n_layers * (1.0 - cfg.fine_tune_ratio))
        cfg.logger.info(
            f"Unfreezing last {cfg.fine_tune_ratio * 100:.1f}% of backbone "
            f"({n_layers - start_idx}/{n_layers} layers, from index {start_idx})"
        )
    else:
        start_idx = cfg.fine_tune_at
        cfg.logger.info(f"Unfreezing backbone layers from index {start_idx} of {n_layers}")

    if not 0 <= start_idx < n_layers:
        raise ValueError(f"fine-tune start index {start_idx} out of range [0, {n_layers})")

    base_model.trainable = True
    trainable_count = 0
    for i, layer in enumerate(base_model.layers):
        if i < start_idx:
            layer.trainable = False
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False  # preserve pretrained moving statistics
        else:
            layer.trainable = True
            trainable_count += 1

    cfg.logger.info(f"Backbone unfrozen: {trainable_count} trainable layers "
                    f"(BatchNorm layers kept frozen)")


def _export_artifacts(best_model: tf.keras.Model, cfg: Config) -> None:
    """Export all deployment formats into artifacts/<model_type>/.

    A failure in an optional format (TFLite/ONNX) is logged but does not
    abort the run: training results (.keras/SavedModel) are already safe.
    """
    art_dir = cfg.artifacts_dir / cfg.model_type
    art_dir.mkdir(parents=True, exist_ok=True)
    saved_model_dir = art_dir / "saved_model"

    # 1. Native Keras format (primary artifact)
    keras_path = art_dir / "model.keras"
    best_model.save(keras_path)
    cfg.logger.info(f"Saved .keras model: {keras_path}")

    # 2. TF SavedModel (standard TF serving format)
    # Keras 3's model.export() (ExportArchive) writes a SavedModel with a
    # "serve" endpoint aliased to "serving_default", the signature that
    # serving tools and tf2onnx consume. The raw tf.saved_model.save
    # fallback covers Keras 2.x where model.export does not exist.
    try:
        best_model.export(str(saved_model_dir))
    except AttributeError:
        tf.saved_model.save(best_model, str(saved_model_dir))
    cfg.logger.info(f"Saved SavedModel: {saved_model_dir}")

    # 3. TFLite (edge deployment)
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        tflite_model = converter.convert()
        tflite_path = art_dir / "model.tflite"
        tflite_path.write_bytes(tflite_model)
        cfg.logger.info(f"Saved TFLite model: {tflite_path}")
    except Exception as e:
        cfg.logger.error(f"TFLite export failed (non-fatal): {e}")

    # 4. ONNX (served by ONNX Runtime)
    try:
        export_saved_model_to_onnx(saved_model_dir, art_dir / "model.onnx")
    except Exception as e:
        cfg.logger.error(f"ONNX export failed (non-fatal): {e}. "
                         f"Install tf2onnx or run export_onnx.py manually.")

    # 5. Resolved configuration (reproducibility: exact params next to the model)
    cfg.save(art_dir / "config.json")


def train(cfg: Config) -> None:
    """Full training pipeline: data -> model -> train -> (fine-tune) -> evaluate -> export."""
    strategy = setup_tf_runtime(cfg)

    # ===============================
    # Data
    # ===============================
    dataset = CIFARDataLoader(cfg)

    # drop_remainder=True on the training pipeline means every epoch has
    # exactly floor(N / batch) steps (the last partial batch is dropped).
    steps_per_epoch = max(1, len(dataset.x_train) // cfg.batch_size)

    # ===============================
    # Model + optimizer (inside strategy scope so variables are mirrored)
    # ===============================
    with strategy.scope():
        model = CIFARModel(cfg).build()
        optimizer, scheduler_callbacks = create_optimizer_and_scheduler(cfg, steps_per_epoch)
        _compile_model(model, cfg, optimizer)

    model.summary()
    trainable_params = sum(
        tf.size(v).numpy() for v in model.trainable_weights
    )
    cfg.logger.info(f"Trainable parameters: {trainable_params:,}")

    best_model_path = str(cfg.ckpt_dir / "best.keras")
    callback_list = scheduler_callbacks + create_training_callbacks(cfg, best_model_path)

    # ===============================
    # Training (phase 1)
    # ===============================
    if cfg.mode == "sanity":
        cfg.logger.info("=" * 60)
        cfg.logger.info("Running sanity check (1 batch train / 1 batch val per epoch)")
        cfg.logger.info("=" * 60)
        history = model.fit(
            dataset.train_dataset().take(1),
            validation_data=dataset.val_dataset().take(1),
            epochs=cfg.epochs,
            verbose=0,  # TrainingMonitor logs every epoch to console + file
            callbacks=callback_list,
        )
    else:
        cfg.logger.info("=" * 60)
        cfg.logger.info("Starting full training")
        cfg.logger.info("=" * 60)
        history = model.fit(
            dataset.train_dataset(),
            validation_data=dataset.val_dataset(),
            epochs=cfg.epochs,
            verbose=0,
            callbacks=callback_list,
        )

    # ===============================
    # Fine-tuning (phase 2, pretrained backbones only)
    # ===============================
    wants_fine_tune = (
        cfg.fine_tune_at is not None
        or (cfg.fine_tune_ratio is not None and cfg.fine_tune_ratio > 0)
    )
    if cfg.mode == "full" and cfg.model_type.lower() in BACKBONE_LAYER_NAMES and wants_fine_tune:
        cfg.logger.info("=" * 60)
        cfg.logger.info("Entering fine-tuning phase...")
        cfg.logger.info("=" * 60)

        _fine_tune_backbone(model, cfg)

        # Fresh callbacks for phase 2: EarlyStopping/ModelCheckpoint start
        # from clean state (no best value / wait counter carried over).
        # Same loss (incl. label smoothing) as phase 1 so val_loss is
        # comparable across the two phases.
        ft_callback_list = create_training_callbacks(cfg, best_model_path)

        initial_epoch = len(history.history.get("loss", []))
        total_epochs = initial_epoch + cfg.fine_epochs
        cfg.logger.info(
            f"Fine-tuning from epoch {initial_epoch} to {total_epochs} "
            f"(constant lr={cfg.fine_tune_lr})"
        )

        with strategy.scope():
            _compile_model(model, cfg,
                           tf.keras.optimizers.Adam(learning_rate=cfg.fine_tune_lr))

        history_fine = model.fit(
            dataset.train_dataset(),
            validation_data=dataset.val_dataset(),
            epochs=total_epochs,
            initial_epoch=initial_epoch,
            verbose=0,
            callbacks=ft_callback_list,
        )

        for metric, values in history_fine.history.items():
            history.history.setdefault(metric, []).extend(values)

        cfg.logger.info("Fine-tuning phase completed")

    # ===============================
    # Load best checkpoint and evaluate on the official test split
    # ===============================
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            f"Best model checkpoint not found: {best_model_path}. "
            f"Training may have failed before the first epoch completed."
        )

    cfg.logger.info("Loading best checkpoint (by val_accuracy) for final evaluation...")
    best_model = tf.keras.models.load_model(best_model_path)
    loss, acc = best_model.evaluate(dataset.test_dataset(), verbose=0)
    cfg.logger.info(f"TEST RESULTS | accuracy: {acc * 100:.2f}% | loss: {loss:.4f}")

    # ===============================
    # Export deployment artifacts
    # ===============================
    _export_artifacts(best_model, cfg)

    # ===============================
    # Visualization
    # ===============================
    VisualizationUtils.plot_training_curves(history, str(cfg.fig_dir))
    VisualizationUtils.visualize_predictions(
        model=best_model,
        save_path=str(cfg.fig_dir),
        test_dataset=dataset.test_dataset(),
        class_names=cfg.class_names,
    )


if __name__ == "__main__":
    train(parse_args())
