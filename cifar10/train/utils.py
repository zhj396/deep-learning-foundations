# -*- coding: utf-8 -*-
"""
Visualization Utilities:
- Plot training curves (accuracy & loss)
- Visualize model predictions on test samples
"""

import logging
import os
from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

logger = logging.getLogger("train")

# Non-interactive backend: figures are saved to disk, never shown, so training
# also works on headless machines (Kaggle, CI, ssh sessions).
matplotlib.use("Agg")


class VisualizationUtils:
    """Static methods for training-curve plots and prediction previews."""

    # ========================
    # Internal Helper Methods
    # ========================
    @staticmethod
    def _smooth_curve(points: List[float], factor: float = 0.8) -> List[float]:
        """Apply an exponential moving average to smooth a curve."""
        smoothed = []
        for p in points:
            if smoothed:
                smoothed.append(smoothed[-1] * factor + p * (1 - factor))
            else:
                smoothed.append(p)
        return smoothed

    @staticmethod
    def _save_plot(fig: plt.Figure, path: str) -> None:
        """Save a matplotlib figure to disk and close it."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        logger.info(f"Plot saved: {path}")

    @staticmethod
    def _to_display_range(images: np.ndarray) -> np.ndarray:
        """Rescale each image to [0, 1] for display.

        Per-image min-max rescaling makes both unit-scaled inputs and
        ImageNet-preprocessed inputs (resnet50: mean-subtracted negative
        values; efficientnetb0: raw [0, 255]) viewable; the unit-scaled
        case is stretched slightly to span [0, 1].
        """
        flat = images.reshape(len(images), -1)
        mins = flat.min(axis=1).reshape(-1, 1, 1, 1)
        maxs = flat.max(axis=1).reshape(-1, 1, 1, 1)
        return (images - mins) / (maxs - mins + 1e-8)

    # ========================
    # Training Curves
    # ========================
    @classmethod
    def plot_training_curves(cls, history, save_path: str, smooth: bool = True,
                             factor: float = 0.8) -> None:
        """Plot and save training/validation accuracy & loss curves.

        Args:
            history: Keras training History object.
            save_path: Directory to save the plots into.
            smooth: Whether to apply EMA smoothing (raw values are kept in
                the log; smoothing is only cosmetic for the figures).
            factor: Smoothing factor (higher = smoother).
        """
        def maybe_smooth(values):
            return cls._smooth_curve(list(values), factor) if smooth else values

        for metric, ylabel, fname in [
            ("accuracy", "Accuracy", "accuracy.png"),
            ("loss", "Loss", "loss.png"),
        ]:
            fig = plt.figure(figsize=(10, 6))
            plt.plot(maybe_smooth(history.history[metric]), label=f"Train {ylabel}")
            plt.plot(maybe_smooth(history.history[f"val_{metric}"]),
                     label=f"Validation {ylabel}")
            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.title(f"Training and Validation {ylabel}" + (" (Smoothed)" if smooth else ""))
            plt.legend()
            plt.grid(True)
            cls._save_plot(fig, os.path.join(save_path, fname))

    # ========================
    # Prediction Visualization
    # ========================
    @classmethod
    def visualize_predictions(
        cls,
        model: tf.keras.Model,
        save_path: str,
        test_dataset: Optional[tf.data.Dataset] = None,
        x_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
        num_images: int = 10,
    ) -> None:
        """Visualize model predictions on a few test images and save the figure.

        Data can come either from a batched tf.data.Dataset (one batch is
        taken and sliced) or from raw numpy arrays.

        Args:
            model: Trained Keras model.
            save_path: Directory to save predictions.png into.
            test_dataset: Batched tf.data.Dataset yielding (images, one-hot labels).
            x_test / y_test: Numpy arrays (alternative to test_dataset).
            class_names: Class names for the title labels.
            num_images: Number of images to display (grid is 2 x 5).
        """
        class_names = class_names or [str(i) for i in range(10)]
        num_images = min(num_images, 10)

        if test_dataset is not None:
            # take(1) yields ONE BATCH; slice the first num_images items from it.
            for images, labels in test_dataset.take(1):
                images = images.numpy()[:num_images]
                labels = labels.numpy()[:num_images]
        elif x_test is not None and y_test is not None:
            images = np.asarray(x_test)[:num_images]
            labels = np.asarray(y_test)[:num_images]
        else:
            raise ValueError("Must provide either test_dataset or (x_test, y_test)")

        if len(images) == 0:
            logger.warning("No images available for prediction visualization.")
            return

        # Labels may be one-hot (dataset path) or integer (numpy path)
        label_ids = (np.argmax(labels, axis=-1) if np.ndim(labels) > 1
                     else labels.astype(int))

        logger.info(f"Generating predictions for {len(images)} test images...")
        preds = model.predict(images, verbose=0)
        pred_ids = np.argmax(preds, axis=-1)

        display_images = cls._to_display_range(images)

        cols = 5
        rows = int(np.ceil(len(images) / cols))
        fig = plt.figure(figsize=(15, 3.2 * rows))
        for i in range(len(images)):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(np.clip(display_images[i], 0, 1))

            true_label = class_names[label_ids[i]]
            pred_label = class_names[pred_ids[i]]
            confidence = preds[i][pred_ids[i]] * 100

            color = "green" if label_ids[i] == pred_ids[i] else "red"
            plt.title(f"True: {true_label}\nPred: {pred_label}\n({confidence:.1f}%)",
                      color=color, fontsize=10)
            plt.axis("off")

        plt.tight_layout()
        cls._save_plot(fig, os.path.join(save_path, "predictions.png"))
